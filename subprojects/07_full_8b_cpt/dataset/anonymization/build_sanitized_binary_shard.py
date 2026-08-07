#!/usr/bin/env python3
"""Build one decontaminated, anonymized, globally deduplicated binary shard."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from anonymization_common import (
    DEDUP_SCHEMA,
    SHARD_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    import_parent_builder,
    load_overlay_heldouts,
    load_parent,
    load_task_exclusions,
    read_json,
    sha256_file,
    utc_now,
    validate_file_receipt,
    validate_overlay,
    validate_task_input,
    write_json_atomic,
)
from bridge_common import iter_index_lengths, task_output_prefix, tokenizer_tree_receipt, write_index
from pii_masker import mask


PARENT: Any = None


def worker_init(tokenizer_dir: str, vocab_size: int, tokenizer_sha: str) -> None:
    PARENT._worker_init(tokenizer_dir, vocab_size, tokenizer_sha)


def encode(record: tuple[str, str]) -> dict[str, Any]:
    doc_id, text = record
    raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if PARENT._DECONTAM_MODULE is not None:
        action, reason, evidence = PARENT._DECONTAM_MODULE.match_document(text, PARENT._DECONTAM_INDEX)
        if action == "drop":
            return {"doc_id": doc_id, "drop": "greekmmlu", "reason": reason, "evidence": evidence, "raw_sha256": raw_sha}
    masked, pii = mask(text)
    masked_sha = hashlib.sha256(masked.encode("utf-8")).hexdigest()
    token_ids = PARENT._TOKENIZER.encode(masked, add_special_tokens=False).ids
    token_ids.append(PARENT._EOS_ID)
    if not token_ids or min(token_ids) < 0 or max(token_ids) >= PARENT._VOCAB_SIZE:
        raise ValueError(f"token id outside frozen vocabulary: {doc_id}")
    return {
        "doc_id": doc_id,
        "drop": False,
        "raw_sha256": raw_sha,
        "masked_sha256": masked_sha,
        "token_ids": token_ids,
        "pii": pii,
        "changed": masked != text,
    }


def load_drops(
    receipt: dict[str, Any], task_index: int
) -> tuple[collections.Counter[tuple[str, str]], dict[str, Any]]:
    matches = [row for row in receipt.get("task_drop_files", []) if int(row["task_index"]) == task_index]
    if not matches:
        return collections.Counter(), {"rows": 0}
    if len(matches) != 1:
        raise ValueError("duplicate task drop-file receipts")
    row = matches[0]
    path = validate_file_receipt(row)
    result: collections.Counter[tuple[str, str]] = collections.Counter()
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) != 3
                or len(fields[1]) != 64
                or fields[2]
                not in {"postmask_exact_duplicate", "validation_content_collision"}
            ):
                raise ValueError(f"malformed drop row: {path}")
            int(fields[1], 16)
            result[(fields[0], fields[1])] += 1
    if sum(result.values()) != int(row["rows"]):
        raise ValueError("task drop-file row accounting drift")
    return result, dict(row)


def consume_postmask_drop(
    remaining: collections.Counter[tuple[str, str]], doc_id: str, masked_sha256: str
) -> bool:
    """Consume one row-level drop while preserving any selected survivor."""
    key = (doc_id, masked_sha256)
    if remaining[key] <= 0:
        return False
    remaining[key] -= 1
    if remaining[key] == 0:
        del remaining[key]
    return True


def validate_resume(path: Path, task: dict[str, Any], overlay_sha: str, dedup_sha: str) -> bool:
    if not path.exists():
        return False
    value = read_json(path)
    expected = {
        "schema_version": SHARD_SCHEMA,
        "status": "completed",
        "kind": "training",
        "task_id": task["task_id"],
        "task_sha256": canonical_sha256(task),
        "anonymization_overlay_sha256": overlay_sha,
        "postmask_dedup_receipt_sha256": dedup_sha,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"existing sanitized shard binding drift ({key}): {path}")
    for key in ("bin", "idx", "dropped_ledger", "retained_ledger"):
        validate_file_receipt(value["outputs"][key])
    sequences, documents, tokens = iter_index_lengths(Path(value["outputs"]["idx"]["path"]))
    if (sequences, documents, tokens) != (
        value["counts"]["documents"], value["counts"]["document_index_entries"], value["counts"]["tokens"]
    ):
        raise ValueError("existing sanitized shard index drift")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--dedup-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("worker settings must be positive")

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    parent = load_parent(overlay)
    heldouts = load_overlay_heldouts(overlay, overlay_path, args.heldout_manifest)
    tasks = overlay["tasks"]
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError("task index outside frozen inventory")
    task = tasks[args.task_index]
    validate_task_input(task)
    exclusions, exclusion_binding = load_task_exclusions(task, heldouts)
    dedup_path = args.dedup_receipt.resolve()
    dedup = read_json(dedup_path)
    if (
        dedup.get("schema_version") != DEDUP_SCHEMA
        or dedup.get("status") != "completed"
        or dedup.get("overlay_sha256") != sha256_file(overlay_path)
    ):
        raise ValueError("post-mask deduplication receipt is not valid for this overlay")
    drop_counts, drop_binding = load_drops(dedup, args.task_index)
    remaining_drops = drop_counts.copy()
    expected_postmask_dropped_rows = sum(drop_counts.values())
    global PARENT
    PARENT = import_parent_builder()
    if task["decontaminate_greekmmlu"]:
        PARENT._install_decontaminator(parent)

    tokenizer = parent["tokenizer"]
    tokenizer_root = Path(tokenizer["root"])
    if tokenizer_tree_receipt(tokenizer_root)["tree_sha256"] != tokenizer["tree_sha256"]:
        raise ValueError("tokenizer tree drift")
    prefix = task_output_prefix(args.stage_root.resolve(), task)
    manifest = Path(str(prefix) + ".manifest.json")
    overlay_sha = sha256_file(overlay_path)
    dedup_sha = sha256_file(dedup_path)
    if validate_resume(manifest, task, overlay_sha, dedup_sha):
        print(json.dumps({"ok": True, "resumed": True, "task": task["task_id"]}, sort_keys=True))
        return 0
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bin": Path(str(prefix) + ".bin"),
        "idx": Path(str(prefix) + ".idx"),
        "dropped": Path(str(prefix) + ".dropped.jsonl"),
        "retained": Path(str(prefix) + ".retained.jsonl"),
    }
    for path in outputs.values():
        if path.is_symlink():
            raise ValueError(f"refusing generated symlink: {path}")
        path.unlink(missing_ok=True)
    temporary = {key: Path(str(path) + ".partial") for key, path in outputs.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    counters = {
        "input_rows": 0, "filtered_rows": 0, "empty_rows": 0, "heldout_rows": 0,
        "phase_excluded_rows": 0, "candidate_rows": 0, "contaminated_rows": 0,
        "policy_excluded_rows": 0,
        "postmask_dropped_rows": 0, "documents": 0, "tokens": 0,
        "masked_documents": 0, "email_matches": 0, "ip_matches": 0, "iban_matches": 0,
    }
    records = PARENT._iter_parquet_records(task, exclusions, counters)
    context = mp.get_context("fork")
    lengths: list[int] = []
    with (
        context.Pool(
            args.workers, initializer=worker_init,
            initargs=(str(tokenizer_root), int(tokenizer["vocab_size"]), tokenizer["tokenizer_json_sha256"]),
        ) as pool,
        temporary["bin"].open("wb") as binary,
        temporary["dropped"].open("w", encoding="utf-8") as dropped,
        temporary["retained"].open("w", encoding="utf-8") as retained,
    ):
        for result in pool.imap(encode, records, chunksize=args.chunksize):
            if result["drop"] == "greekmmlu":
                counters["contaminated_rows"] += 1
                dropped.write(json.dumps({
                    "doc_id": result["doc_id"], "reason": result["reason"],
                    "raw_text_sha256": result["raw_sha256"], "evidence": result["evidence"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
                continue
            for name in ("email", "ip", "iban"):
                counters[f"{name}_matches"] += int(result["pii"][name])
            counters["masked_documents"] += int(result["changed"])
            if consume_postmask_drop(
                remaining_drops, result["doc_id"], result["masked_sha256"]
            ):
                counters["postmask_dropped_rows"] += 1
                dropped.write(json.dumps({
                    "doc_id": result["doc_id"], "reason": "postmask_dedup_or_validation_collision",
                    "raw_text_sha256": result["raw_sha256"], "masked_text_sha256": result["masked_sha256"],
                }, sort_keys=True) + "\n")
                continue
            values = np.asarray(result["token_ids"], dtype=np.int32)
            binary.write(values.tobytes(order="C"))
            retained.write(json.dumps({
                "doc_id": result["doc_id"],
                "text_sha256": result["masked_sha256"],
                "raw_text_sha256": result["raw_sha256"],
                "masked_text_sha256": result["masked_sha256"],
                "tokens": int(values.size),
            }, sort_keys=True) + "\n")
            lengths.append(int(values.size))
            counters["documents"] += 1
            counters["tokens"] += int(values.size)
        for handle in (binary, dropped, retained):
            handle.flush(); os.fsync(handle.fileno())
    sequences, document_entries, exact_tokens = write_index(temporary["idx"], lengths)
    if sequences != counters["documents"] or exact_tokens != counters["tokens"]:
        raise RuntimeError("sanitized Megatron index accounting does not close")
    if temporary["bin"].stat().st_size != exact_tokens * 4:
        raise RuntimeError("sanitized binary byte accounting does not close")
    if counters["candidate_rows"] != counters["documents"] + counters["contaminated_rows"] + counters["postmask_dropped_rows"]:
        raise RuntimeError("sanitized candidate accounting does not close")
    if remaining_drops:
        raise RuntimeError("task post-mask drop identities do not close")
    if counters["postmask_dropped_rows"] != expected_postmask_dropped_rows:
        raise RuntimeError("task post-mask drop receipt does not close")
    for key, path in outputs.items():
        os.replace(temporary[key], path)
    payload = {
        "schema_version": SHARD_SCHEMA, "status": "completed", "completed_at": utc_now(),
        "task_id": task["task_id"], "task_sha256": canonical_sha256(task), "task_index": args.task_index,
        "kind": "training", "pool": task["pool"], "source_name": task["source_name"],
        "source_weight_within_pool": task.get("source_weight_within_pool"),
        "output_prefix": str(prefix.resolve()),
        "input": {"path": str(Path(task["input_path"]).resolve()), "sha256": task["input_sha256"], "bytes": task["input_bytes"], "rows": task["input_rows"]},
        "tokenizer_tree_sha256": tokenizer["tree_sha256"], "tokenizer": tokenizer,
        "anonymization_overlay": absolute_receipt(overlay_path),
        "anonymization_overlay_sha256": overlay_sha,
        "postmask_dedup_receipt": absolute_receipt(dedup_path),
        "postmask_dedup_receipt_sha256": dedup_sha,
        "heldout_exclusion": exclusion_binding, "task_drop_file": drop_binding,
        "anonymization": overlay["anonymization"],
        "counts": {**counters, "document_index_entries": document_entries},
        "outputs": {
            "bin": absolute_receipt(outputs["bin"]), "idx": absolute_receipt(outputs["idx"]),
            "dropped_ledger": absolute_receipt(outputs["dropped"], rows=counters["contaminated_rows"] + counters["postmask_dropped_rows"]),
            "retained_ledger": absolute_receipt(outputs["retained"], rows=counters["documents"]),
        },
    }
    write_json_atomic(manifest, payload)
    print(json.dumps({"ok": True, "task": task["task_id"], "documents": counters["documents"], "tokens": counters["tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a training-disjoint replacement for the legacy old_greek panel ID."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


INDEX_HEADER = b"MMIDIDX\x00\x00"
CATALOG_DTYPE = np.dtype(
    [("pool", "u1"), ("task_index", "<u4"), ("document_index", "<u4"), ("tokens", "<u4"), ("identity", "V16"), ("order", "V16")],
    align=False,
)
CANDIDATE_DTYPE = np.dtype(
    [("task_index", "<u4"), ("document_index", "<u4"), ("tokens", "<u4"), ("identity", "V16"), ("order", "V16"), ("content", "V32")],
    align=False,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def receipt(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if rows is not None:
        value["rows"] = rows
    return value


def write_index(path: Path, lengths: list[int]) -> None:
    values = np.asarray(lengths, dtype=np.int32)
    pointers = np.empty(len(values), dtype=np.int64)
    offset = 0
    for index, length in enumerate(values):
        pointers[index] = offset
        offset += int(length) * 4
    documents = np.arange(len(values) + 1, dtype=np.int64)
    with path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", len(values)))
        handle.write(struct.pack("<Q", len(documents)))
        handle.write(values.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(documents.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())


def document_key(source_dataset: object, source_doc_id: object) -> str:
    components = [["source_dataset", str(source_dataset)], ["source_doc_id", str(source_doc_id)]]
    payload = {
        "contract": "full-cpt-document-identity-v2",
        "source_name": "greek_replay_apertus_original",
        "identity_scope": "global",
        "components": components,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "docv2:" + hashlib.sha256(encoded).hexdigest()


def training_overlap(path: Path, training: np.ndarray) -> tuple[int, int]:
    total = overlap = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = np.asarray([hashlib.sha256(str(row["text"]).encode("utf-8")).digest()], dtype="V32")
            position = int(np.searchsorted(training, digest)[0])
            total += 1
            overlap += int(position < training.size and training[position] == digest[0])
    return total, overlap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--training-content-receipt", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--original-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pool = read_json(args.pool_receipt)
    tasks = {int(row["task_index"]): row for row in pool["tasks"]}
    training_receipt = read_json(args.training_content_receipt)
    training_path = Path(training_receipt["combined"]["path"])
    if sha256_file(training_path) != training_receipt["combined"]["sha256"]:
        raise ValueError("selected training content inventory drift")
    training = np.memmap(training_path, mode="r", dtype="V32")
    if training.size > 1 and np.any(training[1:] <= training[:-1]):
        raise ValueError("selected training content inventory is not unique and sorted")

    full_path = args.stage_root / "inventory/catalog/old_greek_replay.sorted.catalog45"
    selected_path = args.stage_root / "inventory/catalog/old_greek_replay.source_local_selected.catalog45"
    full = np.memmap(full_path, mode="r", dtype=CATALOG_DTYPE)
    selected = np.memmap(selected_path, mode="r", dtype=CATALOG_DTYPE)
    selected_identities = np.sort(np.array(selected["identity"], copy=True))
    positions = np.searchsorted(selected_identities, full["identity"])
    in_selected = positions < selected_identities.size
    in_selected[in_selected] &= selected_identities[positions[in_selected]] == full["identity"][in_selected]
    candidate_catalog = np.array(full[~in_selected], copy=True)
    candidate_catalog.sort(order=("task_index", "document_index"), kind="stable")
    candidate_info = np.empty(candidate_catalog.size, dtype=CANDIDATE_DTYPE)
    for name in ("task_index", "document_index", "tokens", "identity", "order"):
        candidate_info[name] = candidate_catalog[name]
    start = 0
    while start < candidate_info.size:
        task_index = int(candidate_info[start]["task_index"])
        end = start + 1
        while end < candidate_info.size and int(candidate_info[end]["task_index"]) == task_index:
            end += 1
        rows = candidate_info[start:end]
        indexes = rows["document_index"]
        manifest = read_json(Path(tasks[task_index]["source_manifest"]["path"]))
        ledger = Path(manifest["outputs"]["retained_ledger"]["path"])
        cursor = 0
        with ledger.open("r", encoding="utf-8") as handle:
            for document_index, line in enumerate(handle):
                if cursor == indexes.size:
                    break
                if document_index != int(indexes[cursor]):
                    continue
                value = json.loads(line)
                content = bytes.fromhex(str(value["text_sha256"]))
                identity = hashlib.sha256(str(value["doc_id"]).encode("utf-8") + b"\0" + content).digest()[:16]
                if identity != bytes(rows[cursor]["identity"]) or int(value["tokens"]) != int(rows[cursor]["tokens"]):
                    raise ValueError("candidate catalog/ledger drift")
                rows[cursor]["content"] = content
                cursor += 1
        if cursor != indexes.size:
            raise ValueError("candidate ledger is incomplete")
        start = end

    content_positions = np.searchsorted(training, candidate_info["content"])
    overlaps = content_positions < training.size
    overlaps[overlaps] &= training[content_positions[overlaps]] == candidate_info["content"][overlaps]
    clean = candidate_info[~overlaps]
    clean.sort(order="order", kind="stable")
    chosen: list[np.void] = []
    chosen_content: set[bytes] = set()
    tokens = 0
    for row in clean:
        content = bytes(row["content"])
        if content in chosen_content:
            continue
        chosen.append(row.copy())
        chosen_content.add(content)
        tokens += int(row["tokens"])
        if tokens >= args.target_tokens:
            break
    if tokens < args.target_tokens:
        raise ValueError(f"insufficient clean replay validation capacity: {tokens}")

    wanted_indexes: dict[int, dict[int, dict[str, Any]]] = {}
    for row in chosen:
        wanted_indexes.setdefault(int(row["task_index"]), {})[int(row["document_index"])] = {
            "tokens": int(row["tokens"]), "content": bytes(row["content"]), "identity": bytes(row["identity"])
        }
    wanted_by_doc_id: dict[str, dict[str, Any]] = {}
    for task_index, wanted in wanted_indexes.items():
        manifest = read_json(Path(tasks[task_index]["source_manifest"]["path"]))
        ledger = Path(manifest["outputs"]["retained_ledger"]["path"])
        with ledger.open("r", encoding="utf-8") as handle:
            for document_index, line in enumerate(handle):
                if document_index not in wanted:
                    continue
                value = json.loads(line)
                expected = wanted[document_index]
                if bytes.fromhex(value["text_sha256"]) != expected["content"]:
                    raise ValueError("chosen replay ledger drift")
                wanted_by_doc_id[str(value["doc_id"])] = {**expected, "task_index": task_index, "document_index": document_index}
    if len(wanted_by_doc_id) != len(chosen):
        raise ValueError("chosen replay identities are not unique")

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_root / "tokenizer.json"))
    tokenizer_config = read_json(args.tokenizer_root / "tokenizer_config.json")
    eos_value = tokenizer_config["eos_token"]
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    eos_id = tokenizer.token_to_id(str(eos_token))
    if eos_id is None:
        raise ValueError("tokenizer EOS is missing")
    raw_path = args.output_dir / "val_forget_old_greek_clean.jsonl"
    prefix = args.output_dir / "val_forget_old_greek_clean_text_document"
    bin_path = Path(str(prefix) + ".bin")
    idx_path = Path(str(prefix) + ".idx")
    lengths: list[int] = []
    source_counts: Counter[str] = Counter()
    found: set[str] = set()
    with raw_path.open("w", encoding="utf-8") as raw, bin_path.open("wb") as binary:
        parquet = pq.ParquetFile(args.source_parquet)
        columns = ["text", "source_dataset", "source_doc_id"]
        for batch in parquet.iter_batches(columns=columns, batch_size=4096, use_threads=False):
            values = batch.to_pydict()
            for offset in range(batch.num_rows):
                source_dataset = values["source_dataset"][offset]
                source_doc_id = values["source_doc_id"][offset]
                doc_id = document_key(source_dataset, source_doc_id)
                expected = wanted_by_doc_id.get(doc_id)
                if expected is None:
                    continue
                text = values["text"][offset]
                content = hashlib.sha256(text.encode("utf-8")).digest()
                ids = tokenizer.encode(text, add_special_tokens=False).ids + [int(eos_id)]
                if content != expected["content"] or len(ids) != expected["tokens"]:
                    raise ValueError(f"source reconstruction drift: {doc_id}")
                binary.write(np.asarray(ids, dtype=np.int32).tobytes(order="C"))
                lengths.append(len(ids))
                raw.write(json.dumps({
                    "doc_id": doc_id,
                    "source_dataset": source_dataset,
                    "source_doc_id": source_doc_id,
                    "text_sha256": content.hex(),
                    "text": text,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                source_counts[str(source_dataset)] += 1
                found.add(doc_id)
        raw.flush(); os.fsync(raw.fileno())
        binary.flush(); os.fsync(binary.fileno())
    if found != set(wanted_by_doc_id):
        raise ValueError(f"source parquet did not reconstruct {len(set(wanted_by_doc_id) - found)} chosen rows")
    write_index(idx_path, lengths)
    observed_tokens = sum(lengths)
    if observed_tokens != tokens:
        raise ValueError("rebuilt validation token accounting drift")
    original_rows, original_overlap = training_overlap(args.original_panel, training)
    clean_rows, clean_overlap = training_overlap(raw_path, training)
    if clean_rows != len(lengths) or clean_overlap:
        raise ValueError("replacement replay panel is not training-disjoint")
    payload = {
        "schema_version": "apertus_full_8b_clean_replay_validation_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "name": "old_greek",
        "display_name": "Greek replay retention",
        "legacy_id_disclosure": "old_greek is retained only as a compatibility ID; this source pool is not evidence of Ancient-Greek capability",
        "documents": len(lengths),
        "tokens": observed_tokens,
        "megatron_prefix": str(prefix.resolve()),
        "raw_jsonl": receipt(raw_path, rows=len(lengths)),
        "bin": receipt(bin_path),
        "idx": receipt(idx_path),
        "tokenizer_json_sha256": sha256_file(args.tokenizer_root / "tokenizer.json"),
        "source_parquet": receipt(args.source_parquet, rows=2_223_742),
        "selected_training_content_receipt": receipt(args.training_content_receipt),
        "catalogs": {"full": receipt(full_path, rows=int(full.size)), "selected": receipt(selected_path, rows=int(selected.size))},
        "overlap_audit": {
            "identity": "sha256_exact_utf8_text",
            "training_unique_texts": int(training.size),
            "original_panel": {"documents": original_rows, "overlapping_documents": original_overlap, "overlap_fraction": original_overlap / original_rows},
            "replacement_panel": {"documents": clean_rows, "overlapping_documents": clean_overlap, "overlap_fraction": 0.0},
        },
        "source_dataset_documents": dict(sorted(source_counts.items())),
        "selection": {"seed_order": 20260801, "unconsumed_catalog_only": True, "target_tokens_minimum": args.target_tokens, "exact_content_deduplicated_within_panel": True},
    }
    manifest = args.output_dir / "clean_replay_validation_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "documents": len(lengths), "tokens": observed_tokens, "original_overlap": original_overlap}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Mask one frozen training task and inventory its post-mask content hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from anonymization_common import (
    INVENTORY_SCHEMA,
    absolute_receipt,
    canonical_sha256,
    import_parent_builder,
    load_overlay_heldouts,
    load_parent,
    load_task_exclusions,
    sha256_file,
    task_manifest_path,
    task_payload_path,
    utc_now,
    validate_overlay,
    validate_task_input,
    write_json_atomic,
)
from pii_masker import mask


PARENT: Any = None


def inspect_record(record: tuple[str, str]) -> dict[str, Any]:
    doc_id, text = record
    raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if PARENT._DECONTAM_MODULE is not None:
        action, reason, _ = PARENT._DECONTAM_MODULE.match_document(
            text, PARENT._DECONTAM_INDEX
        )
        if action == "drop":
            return {
                "doc_id": doc_id,
                "drop": True,
                "reason": reason,
                "raw_sha256": raw_sha,
            }
    masked, counts = mask(text)
    return {
        "doc_id": doc_id,
        "drop": False,
        "raw_sha256": raw_sha,
        "masked_sha256": hashlib.sha256(masked.encode("utf-8")).hexdigest(),
        "pii": counts,
        "changed": masked != text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("worker settings must be positive")

    overlay_path = args.overlay.resolve()
    overlay = validate_overlay(overlay_path, Path(__file__))
    parent_receipt = load_parent(overlay)
    heldouts = load_overlay_heldouts(overlay, overlay_path, args.heldout_manifest)
    tasks = parent_receipt["tasks"]
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError("task index outside frozen inventory")
    task = tasks[args.task_index]
    validate_task_input(task)
    exclusions, exclusion_binding = load_task_exclusions(task, heldouts)

    global PARENT
    PARENT = import_parent_builder()
    if task["decontaminate_greekmmlu"]:
        PARENT._install_decontaminator(parent_receipt)
    counters = {
        "input_rows": 0,
        "filtered_rows": 0,
        "empty_rows": 0,
        "heldout_rows": 0,
        "phase_excluded_rows": 0,
        "candidate_rows": 0,
        "contaminated_rows": 0,
        "inventory_rows": 0,
        "masked_documents": 0,
        "email_matches": 0,
        "ip_matches": 0,
        "iban_matches": 0,
    }
    records = PARENT._iter_parquet_records(task, exclusions, counters)
    output = task_payload_path(args.stage_root.resolve(), args.task_index, "postmask.tsv")
    manifest = task_manifest_path(args.stage_root.resolve(), args.task_index, "inventory")
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != INVENTORY_SCHEMA
            or value.get("task_sha256") != canonical_sha256(task)
            or value.get("overlay_sha256") != sha256_file(overlay_path)
        ):
            raise ValueError(f"existing inventory manifest binding drift: {manifest}")
        receipt = value["output"]
        if absolute_receipt(Path(receipt["path"]), rows=receipt["rows"]) != receipt:
            raise ValueError(f"existing inventory payload drift: {output}")
        print(json.dumps({"ok": True, "resumed": True, "task": task["task_id"]}, sort_keys=True))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".partial")
    temporary.unlink(missing_ok=True)
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool, temporary.open("w", encoding="ascii") as handle:
        for result in pool.imap(inspect_record, records, chunksize=args.chunksize):
            if result["drop"]:
                counters["contaminated_rows"] += 1
                continue
            # Hashes and opaque document ids only. No matched PII is ever logged.
            handle.write(
                f"{result['masked_sha256']}\t{args.task_index:05d}\t{result['doc_id']}\n"
            )
            counters["inventory_rows"] += 1
            counters["masked_documents"] += int(result["changed"])
            for name in ("email", "ip", "iban"):
                counters[f"{name}_matches"] += int(result["pii"][name])
        handle.flush()
        os.fsync(handle.fileno())
    if counters["candidate_rows"] != counters["contaminated_rows"] + counters["inventory_rows"]:
        raise RuntimeError("inventory candidate accounting does not close")
    os.replace(temporary, output)
    payload = {
        "schema_version": INVENTORY_SCHEMA,
        "status": "completed",
        "completed_at": utc_now(),
        "task_index": args.task_index,
        "task_id": task["task_id"],
        "task_sha256": canonical_sha256(task),
        "pool": task["pool"],
        "source_name": task["source_name"],
        "overlay": absolute_receipt(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "heldout_exclusion": exclusion_binding,
        "anonymization": overlay["anonymization"],
        "counts": counters,
        "output": absolute_receipt(output, rows=counters["inventory_rows"]),
    }
    write_json_atomic(manifest, payload)
    print(json.dumps({"ok": True, "task": task["task_id"], **counters}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

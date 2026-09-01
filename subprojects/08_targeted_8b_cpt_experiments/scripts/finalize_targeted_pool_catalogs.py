#!/usr/bin/env python3
"""Freeze no-dedup modern catalogs and inherit vetted replay capacities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from contract_utils import file_binding, nearest_replay_targets, read_json, require, sha256_file, write_json_atomic


CATALOG_DTYPE = np.dtype(
    [("pool", "u1"), ("task_index", "<u4"), ("document_index", "<u4"), ("tokens", "<u4"), ("identity", "V16"), ("order", "V16")],
    align=False,
)
POOL_CODES = {"hplt_new_greek": 0, "non_hplt_new_greek": 1, "foreign_replay": 2, "old_greek_replay": 3}


def digest16(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()[:16]


def retained_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"malformed retained ledger: {path}")
                rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--academic-pool-receipt", type=Path, required=True)
    parser.add_argument("--poly-pool-receipt", type=Path, required=True)
    parser.add_argument("--parent-pool-receipt", type=Path, required=True)
    parser.add_argument("--schedule-seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable pool receipt exists: {args.output}")
    inputs = read_json(args.input_receipt)
    heldout = read_json(args.heldout_manifest)
    academic = read_json(args.academic_pool_receipt)
    poly = read_json(args.poly_pool_receipt)
    parent = read_json(args.parent_pool_receipt)
    require(heldout.get("input_receipt_sha256") == sha256_file(args.input_receipt), "heldout/input binding drift")
    catalogs: dict[str, list[np.ndarray]] = {"hplt_new_greek": [], "non_hplt_new_greek": []}
    manifests: list[dict[str, Any]] = []
    input_sha = sha256_file(args.input_receipt)
    heldout_sha = sha256_file(args.heldout_manifest)
    for task in inputs["tasks"]:
        pool = task["pool"]
        if pool not in catalogs or task.get("task_origin") != "targeted_new_modern":
            continue
        prefix = (args.stage_root / "megatron" / task["output_prefix"]).resolve()
        manifest_path = Path(str(prefix) + ".manifest.json")
        manifest = read_json(manifest_path)
        require(manifest.get("status") == "completed", f"binary task incomplete: {manifest_path}")
        require(manifest.get("input_receipt_sha256") == input_sha, "binary input binding drift")
        require(manifest.get("heldout_manifest_sha256") == heldout_sha, "binary heldout binding drift")
        require(int(manifest["task_index"]) == int(task["task_index"]), "binary task index drift")
        ledger_path = Path(manifest["outputs"]["retained_ledger"]["path"])
        require(sha256_file(ledger_path) == manifest["outputs"]["retained_ledger"]["sha256"], "retained ledger drift")
        ledger = retained_rows(ledger_path)
        require(len(ledger) == int(manifest["counts"]["documents"]), "retained ledger rows drift")
        rows = np.empty(len(ledger), dtype=CATALOG_DTYPE)
        for document_index, row in enumerate(ledger):
            identity = digest16("targeted8b-record-v1", task["task_index"], document_index, row["doc_id"], row["text_sha256"])
            order = hashlib.sha256(struct.pack("<Q", args.schedule_seed) + identity).digest()[:16]
            rows[document_index] = (POOL_CODES[pool], int(task["task_index"]), document_index, int(row["tokens"]), identity, order)
        catalogs[pool].append(rows)
        manifests.append(file_binding(manifest_path))
    catalog_root = args.output.parent / "catalog"
    require(not catalog_root.exists(), f"immutable catalog root exists: {catalog_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_catalog_root = Path(
        tempfile.mkdtemp(prefix=".catalog.", suffix=".partial", dir=args.output.parent)
    )
    receipts: dict[str, dict[str, Any]] = {}
    capacities: dict[str, int] = {}
    published_catalog = False
    try:
        for pool in ("hplt_new_greek", "non_hplt_new_greek"):
            require(catalogs[pool], f"no modern binary rows for {pool}")
            rows = np.concatenate(catalogs[pool])
            order = np.lexsort((rows["identity"], rows["order"]))
            rows = rows[order]
            require(len(np.unique(rows["identity"])) == len(rows), f"record identity collision in {pool}")
            temporary_path = temporary_catalog_root / f"{pool}.sorted.catalog45"
            final_path = catalog_root / temporary_path.name
            rows.tofile(temporary_path)
            binding = file_binding(temporary_path)
            binding["path"] = str(final_path.resolve())
            capacities[pool] = int(rows["tokens"].astype(np.uint64).sum())
            receipts[pool] = {
                **binding,
                "rows": int(rows.size),
                "tokens": capacities[pool],
                "record_bytes": CATALOG_DTYPE.itemsize,
                "sort_key": "sha256(seed_le64 || unique_record_identity)_prefix128",
                "schedule_seed": args.schedule_seed,
            }
        academic_tokens = int(academic["counts"]["final_training_tokens"])
        poly_tokens = int(poly["counts"]["final_training_tokens"])
        require(capacities["hplt_new_greek"] >= academic_tokens, "HPLT quarter capacity is below academic target")
        require(capacities["non_hplt_new_greek"] == academic_tokens + poly_tokens, "academic/poly binary token total drift")
        for pool in ("foreign_replay", "old_greek_replay"):
            receipt = dict(parent["sorted_training_catalogs"][pool])
            path = Path(receipt["path"])
            require(path.is_file() and path.stat().st_size == int(receipt["bytes"]), f"parent replay catalog missing: {path}")
            require(sha256_file(path) == receipt["sha256"], f"parent replay catalog drift: {path}")
            receipts[pool] = receipt
            capacities[pool] = int(receipt["tokens"])
        modern = 2 * academic_tokens + poly_tokens
        foreign_target, old_target = nearest_replay_targets(modern)
        active_total = modern + foreign_target + old_target
        require(capacities["foreign_replay"] >= foreign_target, "foreign replay capacity is below target")
        require(capacities["old_greek_replay"] >= old_target, "Old-Greek replay capacity is below target")
        payload = {
            "schema_version": "apertus_schedule_pool_corpus_v1",
            "status": "completed",
            "input_receipt": file_binding(args.input_receipt),
            "heldout_manifest": file_binding(args.heldout_manifest),
            "academic_pool_receipt": file_binding(args.academic_pool_receipt),
            "poly_pool_receipt": file_binding(args.poly_pool_receipt),
            "parent_pool_receipt": file_binding(args.parent_pool_receipt),
            "tokenizer": inputs["tokenizer"],
            "sorted_training_catalogs": receipts,
            "modern_greek": {
                "tokens": modern,
                "hplt_tokens": academic_tokens,
                "glossapi_non_hplt_tokens": academic_tokens + poly_tokens,
            },
            "available_capacities": capacities,
            "integer_79_20_1_geometry": {
                "active_tokens": active_total,
                "modern_greek": modern,
                "foreign_replay": foreign_target,
                "old_greek_replay": old_target,
                "realized_fractions": {
                    "modern_greek": str(modern / active_total),
                    "foreign_replay": str(foreign_target / active_total),
                    "old_greek_replay": str(old_target / active_total),
                },
            },
            "new_binary_task_manifests": manifests,
            "invariants": {
                "academic_and_hplt_active_token_ratio": "1:1",
                "release_polytonic_sources_passes": 1,
                "new_modern_rows_deduplicated": False,
                "new_modern_row_multiplicity_preserved": True,
                "replay_source_catalogs_inherited_byte_exactly": True,
            },
        }
        os.rename(temporary_catalog_root, catalog_root)
        published_catalog = True
        write_json_atomic(args.output, payload)
    except BaseException:
        if published_catalog:
            shutil.rmtree(catalog_root, ignore_errors=True)
        else:
            shutil.rmtree(temporary_catalog_root, ignore_errors=True)
        raise
    print(json.dumps({"ok": True, "modern_target": modern, "capacities": capacities}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

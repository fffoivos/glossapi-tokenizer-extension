#!/usr/bin/env python3
"""Freeze the packed sequence inventory and stable per-pool sequence IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEQUENCE_DTYPE = np.dtype(
    [
        ("sequence_id", "<u8"),
        ("packing_task_index", "<u4"),
        ("row_index", "<u4"),
        ("active_tokens", "<u2"),
    ],
    align=False,
)
SEQUENCE_ORDER_SEED = np.uint64(20260801)


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Bijective 64-bit mixing for the frozen packed-sequence permutation."""
    values = values.astype(np.uint64, copy=True) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packing-plan", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite packed corpus receipt: {args.output}")
    plan = read_json(args.packing_plan)
    plan_schema = plan.get("schema_version")
    if plan_schema not in {
        "apertus_mini_fixed_sequence_packing_plan_v1",
        "apertus_fixed_sequence_packing_plan_v1",
    }:
        raise ValueError("unsupported packing plan")
    generic_schema = plan_schema == "apertus_fixed_sequence_packing_plan_v1"
    bucket_schema = (
        "apertus_fixed_sequence_bucket_v1"
        if generic_schema
        else "apertus_mini_fixed_sequence_bucket_v1"
    )
    plan_sha = sha256_file(args.packing_plan)
    by_pool: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    task_receipts = []
    for task in plan["tasks"]:
        prefix = args.stage_root / "megatron" / task["output_prefix"]
        manifest_path = Path(str(prefix) + ".manifest.json")
        manifest = read_json(manifest_path)
        if (
            manifest.get("schema_version") != bucket_schema
            or manifest.get("status") != "completed"
            or int(manifest.get("task_index", -1)) != int(task["task_index"])
            or manifest.get("packing_plan_sha256") != plan_sha
            or int(manifest.get("active_tokens", -1)) != int(task["target_active_tokens"])
        ):
            raise ValueError(f"packed bucket manifest binding drift: {manifest_path}")
        for output in manifest["outputs"].values():
            path = Path(output["path"])
            if not path.is_file() or path.stat().st_size != int(output["bytes"]):
                raise ValueError(f"packed output size drift: {path}")
        active_path = Path(manifest["outputs"]["active_counts"]["path"])
        if sha256_file(active_path) != manifest["outputs"]["active_counts"]["sha256"]:
            raise ValueError(f"packed active-count checksum drift: {active_path}")
        by_pool[str(task["pool"])].append((task, manifest))
        task_receipts.append(
            {
                "task_index": task["task_index"],
                "pool": task["pool"],
                "bucket": task["bucket"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    sequence_root = args.stage_root / "sequences"
    sequence_root.mkdir(parents=True, exist_ok=True)
    pools: dict[str, dict[str, Any]] = {}
    all_ids: list[np.ndarray] = []
    for pool, rows in sorted(by_pool.items(), key=lambda item: item[1][0][0]["pool_code"]):
        rows.sort(key=lambda pair: int(pair[0]["bucket"]))
        arrays = []
        token_total = 0
        for task, manifest in rows:
            active = np.fromfile(
                manifest["outputs"]["active_counts"]["path"], dtype=np.uint16
            )
            if active.size != int(manifest["sequence_count"]):
                raise ValueError("packed active-count row drift")
            row_indices = np.arange(active.size, dtype=np.uint64)
            sequence_ids = (
                (np.uint64(task["pool_code"]) << np.uint64(62))
                | (np.uint64(task["bucket"]) << np.uint64(55))
                | row_indices
            )
            array = np.empty(active.size, dtype=SEQUENCE_DTYPE)
            array["sequence_id"] = sequence_ids
            array["packing_task_index"] = int(task["task_index"])
            array["row_index"] = row_indices.astype(np.uint32)
            array["active_tokens"] = active
            arrays.append(array)
            token_total += int(active.astype(np.uint64).sum())
        sequences = np.concatenate(arrays)
        if np.unique(sequences["sequence_id"]).size != sequences.size:
            raise ValueError(f"stable sequence ID collision: {pool}")
        permutation = np.argsort(
            splitmix64(sequences["sequence_id"] ^ SEQUENCE_ORDER_SEED), kind="stable"
        )
        sequences = sequences[permutation]
        path = sequence_root / f"{pool}.sequences18"
        temporary = Path(str(path) + ".partial")
        sequences.tofile(temporary)
        os.replace(temporary, path)
        expected = int(plan["pools"][pool]["target_active_tokens"])
        if token_total != expected:
            raise ValueError(f"packed pool active-token total drift: {pool}")
        pools[pool] = {
            "pool_code": int(rows[0][0]["pool_code"]),
            "sequence_count": int(sequences.size),
            "active_tokens": token_total,
            "sequence_catalog": {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": int(sequences.size),
                "record_bytes": SEQUENCE_DTYPE.itemsize,
            },
            "packing_tasks": len(rows),
            "sequence_order": "splitmix64(sequence_id XOR 20260801)_ascending",
        }
        all_ids.append(sequences["sequence_id"])

    combined_ids = np.concatenate(all_ids)
    if np.unique(combined_ids).size != combined_ids.size:
        raise ValueError("stable sequence IDs are not globally unique")
    payload = {
        "schema_version": (
            "apertus_packed_sequence_corpus_v1"
            if generic_schema
            else "apertus_mini_packed_sequence_corpus_v1"
        ),
        "status": "completed",
        "packing_plan": {"path": str(args.packing_plan.resolve()), "sha256": plan_sha},
        "sequence_record_format": {
            "record_bytes": SEQUENCE_DTYPE.itemsize,
            "fields": [
                "sequence_id_u64",
                "packing_task_index_u32",
                "row_index_u32",
                "active_tokens_u16",
            ],
            "sequence_id_layout": "pool_code_bits_63_62|bucket_bits_61_55|row_bits_54_0",
            "sequence_order_seed": int(SEQUENCE_ORDER_SEED),
            "sequence_order_algorithm": "splitmix64_bijective_sort_over_stable_sequence_ids",
        },
        "pools": pools,
        "global": {
            "sequence_count": int(combined_ids.size),
            "active_tokens": sum(int(row["active_tokens"]) for row in pools.values()),
            "duplicate_sequence_ids": 0,
        },
        "packing_task_manifests": task_receipts,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": True, "sequences": int(combined_ids.size), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

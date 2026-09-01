#!/usr/bin/env python3
"""Locate exact Modern-Greek content duplicates across validated source groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation_root = args.stage_root / "validation" / "partition_groups"

    arrays: list[np.ndarray] = []
    boundaries = [0]
    modern_groups: list[tuple[int, dict[str, Any]]] = []
    for group_index in range(args.expected_groups):
        receipt = read_json(validation_root / f"{group_index:04d}.json")
        if (
            receipt.get("status") != "completed"
            or receipt.get("content_uniqueness", {}).get("policy")
            != "required_globally_for_modern_greek"
        ):
            continue
        content = receipt["content_digest"]
        path = Path(content["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(content["bytes"])
            or sha256_file(path) != content["sha256"]
        ):
            raise ValueError(f"content digest drift: {path}")
        array = np.fromfile(path, dtype="V16")
        arrays.append(array)
        modern_groups.append((group_index, receipt))
        boundaries.append(boundaries[-1] + int(array.size))

    values = np.concatenate(arrays)
    order = np.argsort(values, kind="stable")
    adjacent_equal = values[order[1:]] == values[order[:-1]]
    duplicate_sorted_positions = np.flatnonzero(adjacent_equal)
    duplicate_digest_hex = sorted(
        {bytes(values[order[int(position)]]).hex() for position in duplicate_sorted_positions}
    )
    duplicate_locations: dict[str, list[dict[str, Any]]] = {
        digest: [] for digest in duplicate_digest_hex
    }
    for digest in duplicate_digest_hex:
        target = np.void(bytes.fromhex(digest))
        for global_position in np.flatnonzero(values == target):
            group_slot = bisect_right(boundaries, int(global_position)) - 1
            group_index, receipt = modern_groups[group_slot]
            duplicate_locations[digest].append(
                {
                    "group_index": group_index,
                    "row_within_group_content_digest": int(global_position) - boundaries[group_slot],
                    "source_name": receipt["input"]["source_name"],
                    "input_path": receipt["input"]["path"],
                }
            )

    ledger_matches: dict[str, list[dict[str, Any]]] = {
        digest: [] for digest in duplicate_digest_hex
    }
    affected_groups = {
        int(location["group_index"])
        for locations in duplicate_locations.values()
        for location in locations
    }
    for group_index in sorted(affected_groups):
        receipt = read_json(validation_root / f"{group_index:04d}.json")
        for shard in receipt["shards"]:
            manifest = read_json(Path(shard["manifest_path"]))
            for ledger_kind in ("retained_ledger", "dropped_ledger"):
                ledger_path = Path(manifest["outputs"][ledger_kind]["path"])
                with ledger_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        text_sha256 = str(row.get("text_sha256", ""))
                        prefix = text_sha256[:32]
                        if prefix not in ledger_matches:
                            continue
                        ledger_matches[prefix].append(
                            {
                                "group_index": group_index,
                                "task_id": shard["task_id"],
                                "logical_pool": shard["logical_pool"],
                                "phase": shard["phase"],
                                "ledger_kind": ledger_kind,
                                "ledger_path": str(ledger_path),
                                "doc_id": row.get("doc_id"),
                                "text_sha256": text_sha256,
                                "tokens": row.get("tokens"),
                            }
                        )

    payload = {
        "schema_version": "apertus_mini_modern_content_duplicate_diagnostic_v1",
        "status": "completed",
        "modern_content_rows": int(values.size),
        "duplicate_digest_count": len(duplicate_digest_hex),
        "duplicate_row_excess": int(duplicate_sorted_positions.size),
        "duplicates": [
            {
                "text_sha256_prefix_128": digest,
                "content_digest_locations": duplicate_locations[digest],
                "ledger_matches": ledger_matches[digest],
            }
            for digest in duplicate_digest_hex
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"ok": True, "output": str(args.output), "duplicates": len(duplicate_digest_hex)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

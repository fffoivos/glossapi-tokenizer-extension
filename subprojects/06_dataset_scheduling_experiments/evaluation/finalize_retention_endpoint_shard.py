#!/usr/bin/env python3
"""Freeze one disjoint shard of the retention endpoint wave."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import atomic_write_json, read_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.arm) != len(set(args.arm)) or not 1 <= len(args.arm) <= 4:
        raise ValueError("retention shard must contain one to four unique arms")
    wave = read_json(args.wave_manifest)
    tasks = {row["arm_id"]: row for row in wave.get("tasks", [])}
    if set(args.arm) - set(tasks):
        raise ValueError("retention shard arm inventory drift")
    rows = []
    for label in args.arm:
        receipt_path = Path(tasks[label]["output_root"]) / "receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema_version") != "apertus_mini_retention_endpoint_v1"
            or receipt.get("status") != "completed"
        ):
            raise ValueError(f"retention shard receipt drift: {label}")
        rows.append(
            {
                "model_label": label,
                "receipt": str(receipt_path.resolve()),
                "sha256": sha256_file(receipt_path),
            }
        )
    payload = {
        "schema_version": "apertus_mini_retention_endpoint_shard_v1",
        "status": "completed",
        "wave_manifest": str(args.wave_manifest.resolve()),
        "models": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "models": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

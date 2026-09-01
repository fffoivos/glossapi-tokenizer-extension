#!/usr/bin/env python3
"""Freeze the initial plus five-arm endpoint retention receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import ARMS, atomic_write_json, read_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wave = read_json(args.wave_manifest)
    tasks = {row["arm_id"]: row for row in wave.get("tasks", [])}
    expected = ("initial_shared",) + ARMS
    if tuple(sorted(tasks)) != tuple(sorted(expected)):
        raise ValueError("retention endpoint model inventory drift")
    rows = []
    for label in expected:
        receipt_path = Path(tasks[label]["output_root"]) / "receipt.json"
        receipt = read_json(receipt_path)
        if receipt.get("schema_version") != "apertus_mini_retention_endpoint_v1" or receipt.get("status") != "completed":
            raise ValueError(f"retention receipt drift: {label}")
        rows.append({"model_label": label, "receipt": str(receipt_path.resolve()), "sha256": sha256_file(receipt_path)})
    payload = {
        "schema_version": "apertus_mini_retention_endpoint_wave_v1",
        "status": "completed",
        "models": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "models": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

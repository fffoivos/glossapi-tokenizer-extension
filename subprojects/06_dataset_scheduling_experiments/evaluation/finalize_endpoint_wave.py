#!/usr/bin/env python3
"""Freeze all five Greek endpoint benchmark receipts into one campaign receipt."""

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
        raise ValueError("endpoint wave arm inventory drift")
    rows = []
    for arm in expected:
        receipt_path = Path(tasks[arm]["output_root"]) / "receipt.json"
        receipt = read_json(receipt_path)
        metrics = {row["benchmark"]: row for row in receipt.get("metrics", [])}
        if (
            receipt.get("schema_version") != "apertus_mini_greek_endpoint_benchmarks_v1"
            or receipt.get("status") != "completed"
            or set(metrics) != {"greek_belebele", "demosqa"}
            or int(metrics["greek_belebele"]["n"]) != 900
            or int(metrics["demosqa"]["n"]) != 600
        ):
            raise ValueError(f"endpoint receipt drift: {arm}")
        rows.append(
            {
                "arm_id": arm,
                "metrics": receipt["metrics"],
                "receipt": str(receipt_path.resolve()),
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
    payload = {
        "schema_version": "apertus_mini_greek_endpoint_wave_receipt_v1",
        "status": "completed",
        "wave_manifest": str(args.wave_manifest.resolve()),
        "wave_manifest_sha256": sha256_file(args.wave_manifest),
        "arms": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "models": len(rows), "benchmarks_per_model": 2}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

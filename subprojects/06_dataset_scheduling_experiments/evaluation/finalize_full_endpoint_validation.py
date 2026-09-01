#!/usr/bin/env python3
"""Freeze the 5-arm by 13-source full endpoint validation panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.collect_validation_trajectory import parse_log
from production.campaign_contract import ARMS, TOTAL_ITERATIONS, atomic_write_json, read_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = read_json(args.campaign_manifest)
    panels = tuple(campaign["evaluation"]["validation_panels"])
    rows = []
    logs = []
    for arm in ARMS:
        log = args.output_root / arm / "driver.out"
        parsed = [row for row in parse_log(log) if row["iteration"] == TOTAL_ITERATIONS]
        by_panel = {row["panel"]: row for row in parsed}
        if tuple(sorted(by_panel)) != tuple(sorted(panels)):
            raise ValueError(f"full endpoint validation panel drift for {arm}")
        rows.extend({"arm_id": arm, **by_panel[panel]} for panel in panels)
        logs.append({"arm_id": arm, "path": str(log.resolve()), "sha256": sha256_file(log)})
    payload = {
        "schema_version": "apertus_mini_full_endpoint_validation_v1",
        "status": "completed",
        "iteration": TOTAL_ITERATIONS,
        "eval_iters": 5,
        "scheduled_tokens_per_panel": 5 * 512 * 4096,
        "checkpoint_receipt": str(args.checkpoint_receipt.resolve()),
        "checkpoint_receipt_sha256": sha256_file(args.checkpoint_receipt),
        "row_count": len(rows),
        "expected_row_count": len(ARMS) * len(panels),
        "rows": rows,
        "logs": logs,
    }
    if payload["row_count"] != payload["expected_row_count"]:
        raise ValueError("full endpoint validation row-count drift")
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

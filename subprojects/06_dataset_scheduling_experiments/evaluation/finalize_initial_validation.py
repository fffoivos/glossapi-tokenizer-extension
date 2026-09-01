#!/usr/bin/env python3
"""Freeze the shared iteration-0 validation metrics used by all five arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.collect_validation_trajectory import parse_log
from production.campaign_contract import atomic_write_json, read_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--driver-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = read_json(args.campaign_manifest)
    expected = set(campaign["evaluation"]["validation_panels"])
    rows = [row for row in parse_log(args.driver_log) if row["iteration"] == 0]
    by_panel = {row["panel"]: row for row in rows}
    if set(by_panel) != expected:
        raise ValueError(f"initial validation panel drift: {sorted(set(by_panel) ^ expected)}")
    payload = {
        "schema_version": "apertus_mini_initial_validation_receipt_v1",
        "status": "completed",
        "campaign_manifest": str(args.campaign_manifest.resolve()),
        "campaign_manifest_sha256": sha256_file(args.campaign_manifest),
        "shared_by_all_five_arms": True,
        "iteration": 0,
        "driver_log": str(args.driver_log.resolve()),
        "driver_log_sha256": sha256_file(args.driver_log),
        "panels": [by_panel[name] for name in sorted(expected)],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "panels": len(by_panel)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


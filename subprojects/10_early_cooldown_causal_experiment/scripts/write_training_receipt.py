#!/usr/bin/env python3
"""Write an atomic, classified early-cooldown training terminal receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from contract_utils import atomic_json, file_binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--phase", choices=("branch",), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--control-receipt", type=Path)
    parser.add_argument("--leaf-switch", required=True)
    parser.add_argument("--started-epoch", type=int, required=True)
    parser.add_argument("--finished-epoch", type=int, required=True)
    args = parser.parse_args()
    latest = None
    if args.log and args.log.is_file():
        for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"iteration\s+(\d+)\s*/", line)
            if match:
                latest = int(match.group(1))
    leaf_switches = [value for value in args.leaf_switch.split(":") if value and value != "none"]
    payload = {
        "schema_version": "apertus_full8_early_cooldown_training_v2",
        "status": args.status,
        "phase": args.phase,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": args.job_id,
        "last_logged_iteration": latest,
        "elapsed_seconds": args.finished_epoch - args.started_epoch,
        "allocation": {
            "nodes": 16,
            "leaf_switch": leaf_switches[0] if len(leaf_switches) == 1 else None,
            "leaf_switches": leaf_switches,
            "single_leaf": len(leaf_switches) == 1,
        },
        "control_receipt": (
            file_binding(args.control_receipt)
            if args.control_receipt and args.control_receipt.is_file()
            else None
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "status": args.status, "phase": args.phase}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

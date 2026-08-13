#!/usr/bin/env python3
"""Freeze the successful Slurm test-only result for the exact production command."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("replay", "branch_holder"), required=True)
    args = parser.parse_args()
    require("job" in args.result.lower() and "start" in args.result.lower(), "Slurm test-only did not return a predicted job start")
    role_fragments = {
        "replay": ("--time=05:00:00", "EARLY_PHASE=replay"),
        "branch_holder": ("--time=12:00:00", "EARLY_PHASE=branch", "--dependency=after:"),
    }
    for fragment in ("--partition=normal", "--nodes=16", "--switches=1", "EARLY_RECOVERY_MODE=0", *role_fragments[args.role]):
        require(fragment in args.command, f"test-only command missing {fragment}")
    payload = {
        "schema_version": "apertus_early_cooldown_slurm_test_only_v1",
        "status": "passed",
        "role": args.role,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": args.command,
        "result": args.result,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "result": args.result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

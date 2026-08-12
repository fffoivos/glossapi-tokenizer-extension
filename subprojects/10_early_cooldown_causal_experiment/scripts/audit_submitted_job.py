#!/usr/bin/env python3
"""Freeze the exact resources and submit command of one Slurm job."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from contract_utils import atomic_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", args.job_id], text=True).strip()
    fields = {}
    for part in raw.split():
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    require(fields.get("JobId") == args.job_id, "Slurm job identity drift")
    nodes = int(fields.get("NumNodes", "0"))
    partition = fields.get("Partition")
    if args.role.startswith("evaluation") or args.role == "native_endpoint":
        require(partition == "debug" and nodes == 4, "evaluation resource drift")
    elif "supervisor" in args.role:
        require(partition == "debug" and nodes == 1, "supervisor resource drift")
    elif args.role in {"training", "recovery_train"}:
        require(partition == "normal" and nodes == 16, "training resource drift")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_slurm_job_v1",
        "status": "audited",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": args.job_id,
        "role": args.role,
        "partition": partition,
        "nodes": nodes,
        "time_limit": fields.get("TimeLimit"),
        "dependency": fields.get("Dependency"),
        "command": fields.get("Command"),
        "work_dir": fields.get("WorkDir"),
        "raw_scontrol": raw,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "job": args.job_id, "role": args.role}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

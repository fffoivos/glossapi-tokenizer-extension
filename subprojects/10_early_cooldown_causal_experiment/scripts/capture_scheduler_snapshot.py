#!/usr/bin/env python3
"""Capture the live Clariden queue/partition state on a debug node."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

from contract_utils import atomic_json, require


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug", "scheduler snapshot must run on debug")
    payload = {
        "schema_version": "apertus_early_cooldown_scheduler_snapshot_v1",
        "status": "captured",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": "debug",
        "queue": command("squeue", "-u", "fffoivos", "-o", "%.18i %.9P %.28j %.2t %.10M %.10l %.6D %R"),
        "partitions": command("sinfo", "-o", "%P|%a|%l|%D|%t|%N"),
        "qos": command("sacctmgr", "-n", "-P", "show", "qos", "normal,debug-qos", "format=Name,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPerUser"),
        "selected_training_leaf": "group29",
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "partition": "debug", "job": payload["job_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

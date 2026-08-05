#!/usr/bin/env python3
"""Freeze proof that B:USR1 saved a checkpoint and that exact resume advanced."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


ITERATION = re.compile(r"iteration\s+(\d+)\s*/")


def last_iteration(path: Path) -> int | None:
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITERATION.search(line)
        if match:
            latest = int(match.group(1))
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interrupted-root", type=Path, required=True)
    parser.add_argument("--restart-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interrupted_log = args.interrupted_root / "training.log"
    interrupted_receipt = args.interrupted_root / "training_job_receipt.json"
    marker = args.interrupted_root / "graceful_stop_requested"
    first = read_json(interrupted_receipt)
    latest = last_iteration(interrupted_log)
    if first.get("status") != "gracefully_stopped" or not marker.is_file() or latest is None:
        raise ValueError("interrupted attempt was not a receipted graceful stop")
    if int(first.get("last_logged_iteration", -1)) != latest or latest >= int(first["requested_end_iteration"]):
        raise ValueError("interrupted attempt boundary drift")
    tracker = args.run_root / "checkpoints/latest_checkpointed_iteration.txt"
    checkpoint_iteration = int(tracker.read_text().strip())
    checkpoint = args.run_root / "checkpoints" / f"iter_{checkpoint_iteration:07d}"
    if not 0 < checkpoint_iteration <= latest or not (checkpoint / ".metadata").is_file():
        raise ValueError("graceful stop did not persist a recoverable checkpoint")
    restart_log = args.restart_root / "training.log"
    restart_receipt = args.restart_root / "training_job_receipt.json"
    resumed = read_json(restart_receipt)
    resumed_iteration = last_iteration(restart_log)
    if (
        resumed.get("status") != "completed"
        or int(resumed.get("start_iteration", -1)) != checkpoint_iteration
        or int(resumed.get("end_iteration", -1)) != checkpoint_iteration + 1
        or resumed_iteration != checkpoint_iteration + 1
    ):
        raise ValueError("graceful checkpoint exact resume failed")
    payload = {
        "schema_version": "apertus_full_8b_graceful_stop_smoke_v1",
        "status": "passed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile_id": first["profile_id"],
        "signal": "B:USR1",
        "interrupted": {
            "requested_end_iteration": int(first["requested_end_iteration"]),
            "last_logged_iteration": latest,
            "receipt": file_binding(interrupted_receipt),
            "log": file_binding(interrupted_log),
            "marker": file_binding(marker),
        },
        "checkpoint": {
            "iteration": checkpoint_iteration,
            "root": str(checkpoint.resolve()),
            "metadata_sha256": hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest(),
            "tracker": file_binding(tracker),
        },
        "resume": {
            "passed": True,
            "start_iteration": checkpoint_iteration,
            "end_iteration": checkpoint_iteration + 1,
            "receipt": file_binding(restart_receipt),
            "log": file_binding(restart_log),
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "checkpoint": checkpoint_iteration, "resumed_to": checkpoint_iteration + 1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

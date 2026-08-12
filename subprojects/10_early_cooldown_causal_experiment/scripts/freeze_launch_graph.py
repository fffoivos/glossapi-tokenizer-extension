#!/usr/bin/env python3
"""Freeze the audited initial training/supervisor submission graph."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-job", required=True)
    parser.add_argument("--supervisor-job", required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--supervisor-audit", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, supervisor = read_json(args.train_audit), read_json(args.supervisor_audit)
    gate = read_json(args.operational_gate)
    require(train.get("job_id") == args.train_job and train.get("partition") == "normal" and train.get("nodes") == 16, "training audit drift")
    require(supervisor.get("job_id") == args.supervisor_job and supervisor.get("partition") == "debug" and supervisor.get("nodes") == 1, "supervisor audit drift")
    require(gate.get("status") == "passed", "operational gate drift")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_launch_graph_v1",
        "status": "submitted",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": {
            "training": args.train_job,
            "afterany_supervisor": args.supervisor_job,
        },
        "training_audit": file_binding(args.train_audit),
        "supervisor_audit": file_binding(args.supervisor_audit),
        "operational_gate": file_binding(args.operational_gate),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "train": args.train_job, "supervisor": args.supervisor_job}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

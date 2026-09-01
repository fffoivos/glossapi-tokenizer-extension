#!/usr/bin/env python3
"""Freeze the audited one-allocation training and supervisor graph."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--supervisor-job", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--supervisor-audit", type=Path, required=True)
    parser.add_argument("--test-only", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = read_json(args.training_audit)
    supervisor = read_json(args.supervisor_audit)
    test = read_json(args.test_only)
    gate = read_json(args.operational_gate)
    require(training.get("job_id") == args.training_job and training.get("partition") == "normal" and training.get("nodes") == 16, "training audit drift")
    require(training.get("time_limit") == "12:00:00", "training time-limit drift")
    require(supervisor.get("job_id") == args.supervisor_job and supervisor.get("partition") == "debug" and supervisor.get("nodes") == 1, "supervisor audit drift")
    require(args.training_job in (supervisor.get("dependency") or ""), "supervisor dependency drift")
    require(test.get("status") == "passed" and test.get("role") == "training", "training test-only drift")
    require(gate.get("status") == "passed", "operational gate drift")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_launch_graph_v3",
        "status": "submitted",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": {"training": args.training_job, "afterany_supervisor": args.supervisor_job},
        "training_audit": file_binding(args.training_audit),
        "supervisor_audit": file_binding(args.supervisor_audit),
        "test_only": file_binding(args.test_only),
        "operational_gate": file_binding(args.operational_gate),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "training": args.training_job, "supervisor": args.supervisor_job}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

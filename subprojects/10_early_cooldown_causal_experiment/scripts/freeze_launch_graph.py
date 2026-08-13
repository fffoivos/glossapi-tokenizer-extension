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
    parser.add_argument("--replay-job", required=True)
    parser.add_argument("--branch-holder-job", required=True)
    parser.add_argument("--replay-supervisor-job", required=True)
    parser.add_argument("--branch-supervisor-job", required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--branch-holder-audit", type=Path, required=True)
    parser.add_argument("--replay-supervisor-audit", type=Path, required=True)
    parser.add_argument("--branch-supervisor-audit", type=Path, required=True)
    parser.add_argument("--branch-test-only", type=Path, required=True)
    parser.add_argument("--operational-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay, branch = read_json(args.replay_audit), read_json(args.branch_holder_audit)
    replay_supervisor, branch_supervisor = read_json(args.replay_supervisor_audit), read_json(args.branch_supervisor_audit)
    branch_test = read_json(args.branch_test_only)
    gate = read_json(args.operational_gate)
    require(replay.get("job_id") == args.replay_job and replay.get("partition") == "normal" and replay.get("nodes") == 16, "replay audit drift")
    require(branch.get("job_id") == args.branch_holder_job and branch.get("partition") == "normal" and branch.get("nodes") == 16, "branch-holder audit drift")
    require(replay.get("time_limit") == "05:00:00", "replay time-limit drift")
    require(branch.get("time_limit") == "12:00:00" and args.replay_job in (branch.get("dependency") or ""), "branch-holder dependency/time drift")
    require(replay_supervisor.get("job_id") == args.replay_supervisor_job and replay_supervisor.get("partition") == "debug" and replay_supervisor.get("nodes") == 1, "replay supervisor audit drift")
    require(branch_supervisor.get("job_id") == args.branch_supervisor_job and branch_supervisor.get("partition") == "debug" and branch_supervisor.get("nodes") == 1, "branch supervisor audit drift")
    require(branch_test.get("status") == "passed" and branch_test.get("role") == "branch_holder", "branch-holder test-only drift")
    require(gate.get("status") == "passed", "operational gate drift")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_launch_graph_v2",
        "status": "submitted",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": {
            "replay": args.replay_job,
            "delayed_branch_holder": args.branch_holder_job,
            "replay_afterany_supervisor": args.replay_supervisor_job,
            "branch_afterany_supervisor": args.branch_supervisor_job,
        },
        "replay_audit": file_binding(args.replay_audit),
        "branch_holder_audit": file_binding(args.branch_holder_audit),
        "replay_supervisor_audit": file_binding(args.replay_supervisor_audit),
        "branch_supervisor_audit": file_binding(args.branch_supervisor_audit),
        "branch_test_only": file_binding(args.branch_test_only),
        "operational_gate": file_binding(args.operational_gate),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "replay": args.replay_job, "branch_holder": args.branch_holder_job}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

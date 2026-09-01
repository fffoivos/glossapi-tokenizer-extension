#!/usr/bin/env python3
"""Prove a completed 16-node smoke occupied exactly one Clariden leaf."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def hosts(expression: str) -> set[str]:
    return set(run(["scontrol", "show", "hostnames", expression]).splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--restart-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    row = run(["sacct", "-j", args.job_id, "-X", "-n", "-P", "-o", "State,NodeList,AllocNodes,ExitCode"])
    rows = [line for line in row.splitlines() if line.strip()]
    require(len(rows) == 1, "smoke sacct row count drift")
    state, node_expression, allocated, exit_code = rows[0].split("|")
    require(state == "COMPLETED" and exit_code == "0:0", "smoke job did not complete cleanly")
    nodes = hosts(node_expression)
    require(int(allocated) == len(nodes) == 16, "smoke allocation is not exactly 16 nodes")

    matches: list[str] = []
    topology = run(["scontrol", "show", "topo"])
    for line in topology.splitlines():
        match = re.search(r"SwitchName=(group\d+).*?Nodes=(\S+)", line)
        if match and nodes <= hosts(match.group(2)):
            matches.append(match.group(1))
    require(len(matches) == 1, "smoke nodes do not resolve to exactly one leaf")

    smoke = read_json(args.restart_smoke)
    require(
        smoke.get("schema_version") == "targeted_8b_restart_smoke_v1"
        and smoke.get("status") == "passed"
        and smoke.get("profile_id") == "dp32_16node"
        and smoke.get("allocation", {}).get("normal_nodes") == 16
        and smoke.get("allocation", {}).get("single_leaf") is True,
        "restart-smoke receipt does not prove the allocation contract",
    )
    payload = {
        "schema_version": "targeted_8b_completed_smoke_leaf_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": args.job_id,
        "slurm_state": state,
        "exit_code": exit_code,
        "node_expression": node_expression,
        "nodes": sorted(nodes),
        "node_count": len(nodes),
        "leaf_switch": matches[0],
        "restart_smoke": file_binding(args.restart_smoke),
        "checks": {
            "completed_cleanly": True,
            "exactly_16_nodes": True,
            "exactly_one_level0_leaf": True,
            "restart_receipt_single_leaf": True,
        },
    }
    write_json_atomic(args.output, payload)
    print(matches[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

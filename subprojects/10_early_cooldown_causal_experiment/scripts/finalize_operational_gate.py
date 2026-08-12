#!/usr/bin/env python3
"""Bind the scientific gate to a successful exact Slurm test-only command."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--test-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate, test = read_json(args.launch_gate), read_json(args.test_only)
    require(gate.get("schema_version") == "apertus_full8_early_cooldown_launch_gate_v1" and gate.get("status") == "passed" and all(gate.get("checks", {}).values()), "scientific launch gate failed")
    require(test.get("schema_version") == "apertus_early_cooldown_slurm_test_only_v1" and test.get("status") == "passed", "Slurm test-only failed")
    payload = {
        "schema_version": "apertus_full8_early_cooldown_operational_gate_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "launch_gate": file_binding(args.launch_gate),
        "slurm_test_only": file_binding(args.test_only),
        "checks": {"scientific_gate_passed": True, "exact_normal_16_node_command_accepted": True},
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "operational_gate": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

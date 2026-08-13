#!/usr/bin/env python3
"""Re-hash every artifact bound into a passed early-cooldown launch gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract_utils import read_json, require, sha256_file


def verify(binding: dict) -> None:
    path = Path(binding["path"]).resolve()
    require(path.is_file(), f"missing launch-gate artifact: {path}")
    require(path.stat().st_size == int(binding["bytes"]), f"launch-gate byte drift: {path}")
    require(sha256_file(path) == binding["sha256"], f"launch-gate hash drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    gate = read_json(args.gate)
    require(gate.get("schema_version") == "apertus_full8_early_cooldown_launch_gate_v1" and gate.get("status") == "passed" and all(gate.get("checks", {}).values()), "launch gate is not passed")
    for key in ("contract", "branch_recipe", "code_bundle_receipt", "scheduler_snapshot", "parent_recipe", "schedule_manifest", "validation_manifest"):
        verify(gate[key])
    evidence = gate.get("runtime_evidence", {})
    for key, value in evidence.items():
        if key == "per_document_baselines":
            require(len(value) == 13, "baseline receipt count drift")
            for binding in value:
                verify(binding)
        else:
            verify(value)
    scalar_evidence = sum(key != "per_document_baselines" for key in evidence)
    print(json.dumps({"ok": True, "runtime_bindings": 7 + scalar_evidence + len(evidence.get("per_document_baselines", []))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

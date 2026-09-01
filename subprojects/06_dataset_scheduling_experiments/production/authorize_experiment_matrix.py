#!/usr/bin/env python3
"""Create an authorized matrix only from one passing receipt per declared gate."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

from campaign_contract import atomic_write_json, file_receipt, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-matrix", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.experiment_matrix)
    if matrix.get("launch_authorized") is not False:
        raise ValueError("source matrix must be the frozen unauthorized matrix")
    required = tuple(matrix.get("launch_gates", []))
    if not required or len(required) != len(set(required)):
        raise ValueError("launch gate inventory is empty or duplicated")
    observed: dict[str, dict] = {}
    matrix_receipt = file_receipt(args.experiment_matrix)
    validator_path = Path(__file__).with_name("finalize_launch_gate_set.py")
    validator_sha256 = sha256_file(validator_path)
    for path in args.gate_receipt:
        receipt = read_json(path)
        gate_id = str(receipt.get("gate_id", ""))
        if (
            receipt.get("schema_version") != "apertus_mini_launch_gate_receipt_v1"
            or receipt.get("status") != "passed"
            or receipt.get("launch_authorized") is not True
            or gate_id not in required
        ):
            raise ValueError(f"non-passing or undeclared launch gate receipt: {path}")
        semantics = receipt.get("semantic_validation", {})
        if (
            semantics.get("schema_version")
            != "apertus_mini_launch_gate_semantics_v1"
            or semantics.get("validator")
            != "production/finalize_launch_gate_set.py"
            or semantics.get("validator_sha256") != validator_sha256
            or semantics.get("all_gate_specific_checks_passed") is not True
        ):
            raise ValueError(f"gate receipt lacks the semantic validator proof: {gate_id}")
        if receipt.get("experiment_matrix") != matrix_receipt:
            raise ValueError(f"gate receipt is not bound to the source matrix: {gate_id}")
        if gate_id in observed:
            raise ValueError(f"duplicate launch gate receipt: {gate_id}")
        evidence = receipt.get("evidence", [])
        if not evidence or any(
            not isinstance(row, dict)
            or not row.get("path")
            or not row.get("sha256")
            or "bytes" not in row
            for row in evidence
        ):
            raise ValueError(f"gate receipt lacks hash-bound evidence: {gate_id}")
        for row in evidence:
            evidence_path = Path(row["path"])
            if (
                not evidence_path.is_file()
                or evidence_path.stat().st_size != int(row["bytes"])
                or sha256_file(evidence_path) != row["sha256"]
            ):
                raise ValueError(f"gate evidence drift: {gate_id}/{evidence_path}")
        observed[gate_id] = file_receipt(path)
    missing = sorted(set(required) - set(observed))
    extra = sorted(set(observed) - set(required))
    if missing or extra:
        raise ValueError(f"launch gate coverage mismatch: missing={missing}, extra={extra}")
    output = copy.deepcopy(matrix)
    output["launch_authorized"] = True
    output["status"] = "launch_authorized_all_declared_gate_receipts_passed"
    output["launch_authorization"] = {
        "authorized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_matrix": file_receipt(args.experiment_matrix),
        "gate_receipts": [observed[gate_id] for gate_id in required],
        "gate_ids": list(required),
    }
    atomic_write_json(args.output, output)
    print(json.dumps({"ok": True, "gates": len(required), "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze a non-authorizing draft of hash-bound launch-gate evidence.

Only ``finalize_launch_gate_set.py`` performs the gate-specific semantic checks
and emits receipts accepted by ``authorize_experiment_matrix.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from campaign_contract import atomic_write_json, file_receipt, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-matrix", type=Path, required=True)
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--assertion", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = read_json(args.experiment_matrix)
    if args.gate_id not in matrix.get("launch_gates", []):
        raise ValueError(f"undeclared launch gate: {args.gate_id}")
    if not args.assertion or any(not value.strip() for value in args.assertion):
        raise ValueError("at least one nonempty gate assertion is required")
    evidence = [file_receipt(path) for path in args.evidence]
    payload = {
        "schema_version": "apertus_mini_launch_gate_evidence_draft_v1",
        "status": "evidence_frozen_not_semantically_validated",
        "launch_authorized": False,
        "gate_id": args.gate_id,
        "passed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_matrix": file_receipt(args.experiment_matrix),
        "assertions": args.assertion,
        "evidence": evidence,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "gate_id": args.gate_id,
                "launch_authorized": False,
                "next_step": "production/finalize_launch_gate_set.py",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

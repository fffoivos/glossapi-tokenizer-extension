#!/usr/bin/env python3
"""Turn the reused one-checkpoint matrix aggregate into an endpoint receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = read_json(args.matrix_root / "matrix_receipt.json")
    contract = read_json(args.contract)
    require(matrix.get("status") == "completed" and len(matrix.get("checkpoint_receipts", [])) == 1, "endpoint matrix incomplete")
    require(contract.get("schema_version") == "apertus_full8_native_greek_endpoint_contract_v1", "endpoint contract drift")
    receipt_path = Path(matrix["checkpoint_receipts"][0]["path"])
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "completed" and receipt.get("model") == "iter_0013193", "endpoint aggregate drift")
    payload = {
        "schema_version": "apertus_full8_native_greek_endpoint_receipt_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": 13193,
        "benchmarks": ["DemosQA", "Medical_MCQA", "ASEP_MCQA", "GPCR", "OYXOY"],
        "protipa_status": "blocked_non_blocking",
        "contract": file_binding(args.contract),
        "manifest": file_binding(args.manifest),
        "aggregate": file_binding(receipt_path),
        "matrix_receipt": file_binding(args.matrix_root / "matrix_receipt.json"),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"ok": True, "iteration": 13193, "benchmark_families": 5}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

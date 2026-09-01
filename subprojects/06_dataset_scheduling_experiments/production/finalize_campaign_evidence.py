#!/usr/bin/env python3
"""Freeze completion of training, loss trajectories, and endpoint benchmark jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from campaign_contract import atomic_write_json, file_receipt, read_json


def require(path: Path, schema: str, *, count_key: str | None = None, count: int | None = None) -> dict:
    value = read_json(path)
    if value.get("schema_version") != schema or value.get("status") not in {"completed", "passed"}:
        raise ValueError(f"incomplete campaign evidence: {path}")
    if count_key is not None and int(value.get(count_key, -1)) != int(count):
        raise ValueError(f"campaign evidence count drift: {path}/{count_key}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--full-endpoint-validation-receipt", type=Path)
    parser.add_argument("--greek-endpoint-receipt", type=Path)
    parser.add_argument("--retention-endpoint-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    greek_endpoint_receipt = (
        args.greek_endpoint_receipt.resolve()
        if args.greek_endpoint_receipt is not None
        else root / "greek_endpoint_benchmarks" / "endpoint_wave_receipt.json"
    )
    if root not in greek_endpoint_receipt.parents:
        raise ValueError("Greek endpoint recovery receipt must remain inside the run root")
    full_endpoint_validation_receipt = (
        args.full_endpoint_validation_receipt.resolve()
        if args.full_endpoint_validation_receipt is not None
        else root / "full_endpoint_validation" / "full_endpoint_validation_receipt.json"
    )
    if root not in full_endpoint_validation_receipt.parents:
        raise ValueError("full endpoint recovery receipt must remain inside the run root")
    retention_endpoint_receipt = (
        args.retention_endpoint_receipt.resolve()
        if args.retention_endpoint_receipt is not None
        else root / "retention_endpoint_benchmarks" / "retention_wave_receipt.json"
    )
    if root not in retention_endpoint_receipt.parents:
        raise ValueError("retention endpoint recovery receipt must remain inside the run root")
    paths = {
        "training": root / "training_completion_receipt.json",
        "full_endpoint_validation": full_endpoint_validation_receipt,
        "validation_trajectory": root / "analysis" / "validation_trajectory.json",
        "greekmmlu_trajectory": root / "analysis" / "greekmmlu_trajectory.json",
        "core_summary": root / "analysis" / "core_campaign_summary.json",
        "greek_endpoints": greek_endpoint_receipt,
        "retention_endpoints": retention_endpoint_receipt,
    }
    require(paths["training"], "apertus_mini_training_completion_v1")
    require(paths["full_endpoint_validation"], "apertus_mini_full_endpoint_validation_v1", count_key="row_count", count=65)
    require(paths["validation_trajectory"], "apertus_mini_validation_trajectory_v1", count_key="row_count", count=5395)
    require(paths["greekmmlu_trajectory"], "apertus_mini_greekmmlu_trajectory_v1", count_key="row_count", count=415)
    require(paths["core_summary"], "apertus_mini_core_campaign_summary_v1")
    greek = require(paths["greek_endpoints"], "apertus_mini_greek_endpoint_wave_receipt_v1")
    retention = require(paths["retention_endpoints"], "apertus_mini_retention_endpoint_wave_v1")
    if len(greek.get("arms", [])) != 6 or len(retention.get("models", [])) != 6:
        raise ValueError("initial-plus-five endpoint inventory drift")
    payload = {
        "schema_version": "apertus_mini_campaign_evidence_completion_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_trajectories": 5,
        "checkpoint_averaging": False,
        "validation_bindings": 5395,
        "native_greekmmlu_bindings": 415,
        "full_endpoint_validation_bindings": 65,
        "endpoint_models_including_initial": 6,
        "evidence": {name: file_receipt(path) for name, path in paths.items()},
        "winner_selected": False,
        "next_gate": "paired_uncertainty_and_predeclared_retention_constraints",
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "evidence_files": len(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

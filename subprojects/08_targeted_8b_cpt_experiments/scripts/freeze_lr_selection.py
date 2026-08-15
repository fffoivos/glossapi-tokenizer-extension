#!/usr/bin/env python3
"""Freeze the approved fixed learning-rate choice for either model scale."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from build_training_run_permit import LR_CHECKS


FLOORS = {
    "5.5e-5": "5.5e-6",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--historical-decision", type=Path)
    parser.add_argument("--arm", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable LR-selection receipt exists: {args.output}")
    experiment = read_json(args.experiment)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    current = executing_code_bundle()
    if args.scale == "8b":
        require(args.historical_decision is not None and args.historical_decision.is_file() and not args.arm, "8B LR selection requires only the historical decision")
        text = args.historical_decision.read_text(encoding="utf-8")
        require("Use **peak LR `5.5e-5`**" in text and "Use `LR_PEAK=5.5e-5`" in text, "historical 8B LR decision drift")
        peak = "5.5e-5"
        evidence = [file_binding(args.historical_decision)]
        candidates = ["2.75e-5", "5.5e-5", "8.25e-5", "1.1e-4"]
        decision = {"method": "historical_completed_8b_loss_first_sweep", "selected_peak_lr": peak}
    else:
        require(args.historical_decision is None and not args.arm, "1.5B fixed LR selection does not accept pilot arms")
        policy = experiment["learning_rate_1p5b"]
        require(
            policy == {
                "mode": "fixed_matched_8b_recipe",
                "peak_lr": "5.5e-5",
                "floor_lr": "5.5e-6",
                "terminal_lr_ratio": "0.1",
                "source": "same_approved_scientific_recipe_as_8b",
                "lr_pilot_runs": 0,
                "benchmark_values_used_for_selection": False,
            },
            "1.5B fixed LR policy drift",
        )
        peak = policy["peak_lr"]
        evidence = [file_binding(args.experiment)]
        candidates = [peak]
        decision = {
            "method": policy["mode"],
            "source": policy["source"],
            "lr_pilot_runs": policy["lr_pilot_runs"],
            "benchmark_values_used_for_selection": policy["benchmark_values_used_for_selection"],
            "selected_peak_lr": peak,
        }
    floor = FLOORS[peak]
    payload = {
        "schema_version": "apertus_hard_h_to_g_lr_selection_v1",
        "status": "selected",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "executing_code_bundle": current,
        "experiment": file_binding(args.experiment),
        "candidates": candidates,
        "peak_lr": peak,
        "floor_lr": floor,
        "decision": decision,
        "checks": {name: True for name in LR_CHECKS},
        "evidence": evidence,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

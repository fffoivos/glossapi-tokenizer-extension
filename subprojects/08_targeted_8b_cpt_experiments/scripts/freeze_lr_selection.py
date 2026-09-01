#!/usr/bin/env python3
"""Freeze the historical 8B LR or select the 1.5B loss-only pilot winner."""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Any

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic
from build_training_run_permit import LR_CHECKS


GRID = ["7.5e-5", "1.0e-4", "1.25e-4"]
FLOORS = {
    "5.5e-5": "5.5e-6",
    "7.5e-5": "7.5e-6",
    "1.0e-4": "1.0e-5",
    "1.25e-4": "1.25e-5",
}


def relative_improvement(initial: float, final: float) -> float:
    require(initial > 0 and final > 0, "loss values must be positive")
    return (initial - final) / initial


def validate_arm(path: Path, current: dict[str, Any]) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema_version") == "apertus_hard_h_to_g_lr_pilot_arm_v1", f"LR arm schema drift: {path}")
    require(value.get("status") == "completed" and value.get("scale") == "1p5b", f"LR arm identity drift: {path}")
    require(str(value.get("peak_lr")) in GRID and str(value.get("floor_lr")) == FLOORS[str(value["peak_lr"])], f"LR arm value drift: {path}")
    require(int(value.get("updates", -1)) == 238, f"LR arm horizon drift: {path}")
    require(value.get("greekmmlu_accessed") is False and value.get("downstream_benchmarks_accessed") is False, f"LR arm accessed forbidden benchmark: {path}")
    require(value.get("skipped_updates") == 0 and value.get("nonfinite_updates") == 0, f"LR arm instability: {path}")
    require(value.get("optimizer_state_id") and value.get("initialization_binding") and value.get("data_prefix_trajectory_sha256"), f"LR arm isolation binding missing: {path}")
    bundle = value.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], f"LR arm code-bundle drift: {path}")
    losses = value.get("panel_losses")
    require(isinstance(losses, dict) and set(losses) == {"initial", "final", "roles"}, f"LR arm panel-loss block drift: {path}")
    initial = losses["initial"]
    final = losses["final"]
    roles = losses["roles"]
    require(isinstance(initial, dict) and isinstance(final, dict) and set(initial) == set(final), f"LR arm panel inventory drift: {path}")
    require(roles.get("hplt") and roles.get("foreign") and roles.get("old_greek"), f"LR arm panel-role inventory drift: {path}")
    require(set(roles["hplt"]) | set(roles["foreign"]) | set(roles["old_greek"]) == set(initial), f"LR arm panel roles do not cover exact panel set: {path}")
    require(len(roles["hplt"]) == 1 and len(roles["old_greek"]) == 1, f"LR arm HPLT/Old-Greek role cardinality drift: {path}")
    require(all(float(initial[name]) > 0 and float(final[name]) > 0 for name in initial), f"LR arm has invalid loss: {path}")
    return value


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
        require(args.historical_decision is None and args.arm is not None and len(args.arm) == 3, "1.5B LR selection requires exactly three pilot arms")
        arms = [validate_arm(path, current) for path in args.arm]
        require(sorted(str(row["peak_lr"]) for row in arms) == sorted(GRID), "1.5B LR candidate grid incomplete")
        require(len({str(row["optimizer_state_id"]) for row in arms}) == 3, "1.5B LR candidates share optimizer state")
        require(len({str(row["initialization_binding"]) for row in arms}) == 1, "1.5B LR candidates use different initialization")
        require(len({str(row["data_prefix_trajectory_sha256"]) for row in arms}) == 1, "1.5B LR candidates use different data prefixes")
        policy = experiment["learning_rate_pilot_1p5b"]
        margin = float(policy["retention_noninferiority_absolute_loss_margin"])
        scored = []
        for row in arms:
            losses = row["panel_losses"]
            initial = {key: float(value) for key, value in losses["initial"].items()}
            final = {key: float(value) for key, value in losses["final"].items()}
            roles = losses["roles"]
            hplt = statistics.fmean(relative_improvement(initial[name], final[name]) for name in roles["hplt"])
            foreign = statistics.fmean(relative_improvement(initial[name], final[name]) for name in roles["foreign"])
            old = statistics.fmean(relative_improvement(initial[name], final[name]) for name in roles["old_greek"])
            replay_names = [*roles["foreign"], *roles["old_greek"]]
            retention_pass = all(final[name] - initial[name] <= margin for name in replay_names)
            scored.append({"row": row, "score": statistics.fmean((hplt, foreign, old)), "retention_pass": retention_pass, "components": {"hplt": hplt, "foreign_macro": foreign, "old_greek": old}, "maximum_replay_loss_delta": max(final[name] - initial[name] for name in replay_names)})
        eligible = [row for row in scored if row["retention_pass"]]
        require(eligible, "no 1.5B LR candidate passes the predeclared replay-retention margin")
        best = max(row["score"] for row in eligible)
        ties = [row for row in eligible if best - row["score"] <= 1e-6]
        selected = min(ties, key=lambda row: Decimal(str(row["row"]["peak_lr"])))
        peak = str(selected["row"]["peak_lr"])
        evidence = [file_binding(path) for path in args.arm]
        candidates = GRID
        decision = {
            "method": policy["selection_rule"],
            "selection_score": policy["selection_score"],
            "retention_margin": margin,
            "tie_break": policy["tie_break"],
            "arms": {str(row["row"]["peak_lr"]): {key: value for key, value in row.items() if key != "row"} for row in scored},
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

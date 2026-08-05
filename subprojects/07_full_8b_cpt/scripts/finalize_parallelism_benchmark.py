#!/usr/bin/env python3
"""Promote DP64 only after numerical, restart, cursor and speed gates pass."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


ITERATION = re.compile(r"iteration\s+(\d+)\s*/")
METRIC = re.compile(r"(?:^|\|)\s*([^|:]+?)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
STEP_MS = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_log(path: Path, *, require_full: bool = True) -> dict:
    rows: dict[int, dict[str, float]] = {}
    step_ms: list[float] = []
    skipped = 0
    nonfinite = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        timing = STEP_MS.search(line)
        if timing:
            step_ms.append(float(timing.group(1)))
        iteration = ITERATION.search(line)
        if not iteration:
            continue
        metrics = {normalize(key): float(raw) for key, raw in METRIC.findall(line)}
        row = {}
        for source, target in (("lm loss", "loss"), ("grad norm", "grad"), ("params norm", "params")):
            if source in metrics:
                row[target] = metrics[source]
        if row:
            rows[int(iteration.group(1))] = row
        skipped += int(metrics.get("number of skipped iterations", 0))
        if any(not math.isfinite(value) for value in row.values()):
            nonfinite += 1
    if require_full and (len(step_ms) < 288 or len(rows) < 288):
        raise ValueError(f"incomplete benchmark log: {path} timings={len(step_ms)} rows={len(rows)}")
    if not rows:
        raise ValueError(f"benchmark log contains no iteration rows: {path}")
    measured = step_ms[32:288] if require_full else step_ms
    return {
        "rows": rows,
        "median_step_seconds": statistics.median(measured) / 1000 if measured else None,
        "p90_step_seconds": percentile(measured, 0.90) / 1000 if measured else None,
        "observations": len(measured),
        "skipped": skipped,
        "nonfinite": nonfinite,
        "binding": file_binding(path),
    }


def close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return abs(left - right) <= atol + rtol * abs(left)


def exact_restart(uninterrupted: dict, restarted: dict, iteration: int) -> dict:
    left = uninterrupted["rows"].get(iteration)
    right = restarted["rows"].get(iteration)
    if left is None or right is None:
        return {"passed": False, "reason": "comparison update absent"}
    fields = sorted(set(left) | set(right))
    return {
        "passed": left == right,
        "iteration": iteration,
        "uninterrupted": left,
        "restarted": right,
        "absolute_deltas": {field: abs(left.get(field, math.inf) - right.get(field, math.inf)) for field in fields},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = read_json(args.profiles)
    thresholds = profiles["benchmark"]["promotion"]
    contract = read_json(args.benchmark_contract)
    if contract.get("status") != "frozen" or contract.get("updates") != 288:
        raise ValueError("benchmark contract drift")
    control = parse_log(args.control_root / "segments/updates_0_288/training.log")
    candidate = parse_log(args.candidate_root / "segments/updates_0_288/training.log")
    control_restart = parse_log(args.control_root / "segments/updates_160_161/training.log", require_full=False)
    candidate_restart = parse_log(args.candidate_root / "segments/updates_160_161/training.log", require_full=False)
    control_job = read_json(args.control_root / "segments/updates_0_288/training_job_receipt.json")
    candidate_job = read_json(args.candidate_root / "segments/updates_0_288/training_job_receipt.json")
    if control_job.get("scientific_digest") != candidate_job.get("scientific_digest"):
        raise ValueError("control/candidate scientific digest drift")
    first_control = control["rows"][1]
    first_candidate = candidate["rows"][1]
    first_checks = {
        "loss": close(first_control["loss"], first_candidate["loss"], atol=thresholds["loss_atol"], rtol=thresholds["loss_rtol"]),
        "gradient_norm": close(first_control["grad"], first_candidate["grad"], atol=thresholds["gradient_norm_atol"], rtol=thresholds["gradient_norm_rtol"]),
        "parameter_norm": close(first_control["params"], first_candidate["params"], atol=thresholds["parameter_norm_atol"], rtol=thresholds["parameter_norm_rtol"]),
    }
    common = list(range(33, 289))
    control_losses = [control["rows"][index]["loss"] for index in common]
    deltas = [candidate["rows"][index]["loss"] - control["rows"][index]["loss"] for index in common]
    control_std = statistics.stdev(control_losses)
    if control_std == 0:
        rmse_ratio = 0.0 if all(value == 0 for value in deltas) else math.inf
        signed_ratio = 0.0 if statistics.mean(deltas) == 0 else math.inf
    else:
        rmse_ratio = math.sqrt(statistics.mean(value * value for value in deltas)) / control_std
        signed_ratio = abs(statistics.mean(deltas)) / control_std
    speedup = control["median_step_seconds"] / candidate["median_step_seconds"]
    gpu_hour_ratio = 2.0 / speedup
    candidate_wall = float(candidate_job["elapsed_seconds"]) / 288
    amortized_p90 = max(candidate["p90_step_seconds"], candidate_wall)
    checks = {
        "same_scientific_digest": True,
        "same_frozen_sequence_and_goldfish_contract": bool(contract["sequence_ids"]["prefix_sha256"] and contract["goldfish"]["implementation"]["sha256"]),
        "first_batch_loss_gradient_parameter_parity": all(first_checks.values()),
        "trajectory_rmse_within_bound": rmse_ratio <= thresholds["trajectory_rmse_over_control_std_max"],
        "trajectory_signed_mean_within_bound": signed_ratio <= thresholds["trajectory_abs_mean_over_control_std_max"],
        "control_restart_exact": exact_restart(control, control_restart, 161)["passed"],
        "candidate_restart_exact": exact_restart(candidate, candidate_restart, 161)["passed"],
        "zero_skipped_updates": control["skipped"] == candidate["skipped"] == 0,
        "zero_nonfinite_updates": control["nonfinite"] == candidate["nonfinite"] == 0,
        "median_speedup_at_least_1p6": speedup >= thresholds["median_throughput_speedup_min"],
        "gpu_hour_ratio_at_most_1p25": gpu_hour_ratio <= thresholds["gpu_hour_ratio_max"],
        "amortized_p90_wall_at_most_5p89_seconds": amortized_p90 <= thresholds["p90_amortized_wall_seconds_per_update_max"],
    }
    promoted = all(checks.values())
    selected = "dp64_32node" if promoted else "dp32_16node"
    payload = {
        "schema_version": "apertus_full_8b_parallelism_benchmark_v1",
        "status": "promoted" if promoted else "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_profile": selected,
        "candidate_promoted": promoted,
        "checks": checks,
        "thresholds": thresholds,
        "first_batch_checks": first_checks,
        "trajectory": {"rmse_over_control_std": rmse_ratio, "absolute_mean_over_control_std": signed_ratio},
        "performance": {
            "control_median_step_seconds": control["median_step_seconds"],
            "control_p90_step_seconds": control["p90_step_seconds"],
            "candidate_median_step_seconds": candidate["median_step_seconds"],
            "candidate_p90_step_seconds": candidate["p90_step_seconds"],
            "candidate_amortized_wall_seconds_per_update": candidate_wall,
            "candidate_p90_amortized_wall_seconds_per_update": amortized_p90,
            "median_throughput_speedup": speedup,
            "gpu_hour_ratio": gpu_hour_ratio,
            "selected_compute_hours": 19248 * (candidate["median_step_seconds"] if promoted else control["median_step_seconds"]) / 3600,
        },
        "restart": {
            "control": exact_restart(control, control_restart, 161),
            "candidate": exact_restart(candidate, candidate_restart, 161),
        },
        "scientific_digest": control_job["scientific_digest"],
        "inputs": {
            "profiles": file_binding(args.profiles),
            "benchmark_contract": file_binding(args.benchmark_contract),
            "control_log": control["binding"],
            "candidate_log": candidate["binding"],
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "selected_profile": selected, "candidate_promoted": promoted, "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

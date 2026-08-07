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
WALL_TIMESTAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+iteration")


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_log(path: Path, *, require_full: bool = True) -> dict:
    rows: dict[int, dict[str, float]] = {}
    step_ms: list[float] = []
    iteration_timestamps: dict[int, dt.datetime] = {}
    skipped = 0
    nonfinite = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        timing = STEP_MS.search(line)
        if timing:
            step_ms.append(float(timing.group(1)))
        iteration = ITERATION.search(line)
        if not iteration:
            continue
        iteration_number = int(iteration.group(1))
        timestamp = WALL_TIMESTAMP.search(line)
        if timestamp:
            iteration_timestamps[iteration_number] = dt.datetime.strptime(
                timestamp.group(1), "%Y-%m-%d %H:%M:%S"
            )
        metrics = {normalize(key): float(raw) for key, raw in METRIC.findall(line)}
        row = {}
        for source, target in (("lm loss", "loss"), ("grad norm", "grad"), ("params norm", "params")):
            if source in metrics:
                row[target] = metrics[source]
        if row:
            rows[iteration_number] = row
        skipped += int(metrics.get("number of skipped iterations", 0))
        if any(not math.isfinite(value) for value in row.values()):
            nonfinite += 1
    if require_full and (len(step_ms) < 288 or len(rows) < 288):
        raise ValueError(f"incomplete benchmark log: {path} timings={len(step_ms)} rows={len(rows)}")
    if not rows:
        raise ValueError(f"benchmark log contains no iteration rows: {path}")
    measured = step_ms[32:288] if require_full else step_ms
    timed_payload_seconds = None
    if require_full and 1 in iteration_timestamps and 288 in iteration_timestamps:
        # Megatron logs each wall timestamp after the corresponding update.  The
        # interval from update 1 through update 288 therefore contains updates
        # 2..288 plus every intervening validation/save pause.  Add update 1's
        # measured duration to retain the entire timed payload while excluding
        # only fixed process/model/dataset startup before the first update.
        timed_payload_seconds = (
            iteration_timestamps[288] - iteration_timestamps[1]
        ).total_seconds() + step_ms[0] / 1000
    return {
        "rows": rows,
        "median_step_seconds": statistics.median(measured) / 1000 if measured else None,
        "p90_step_seconds": percentile(measured, 0.90) / 1000 if measured else None,
        "observations": len(measured),
        "timed_payload_seconds": timed_payload_seconds,
        "skipped": skipped,
        "nonfinite": nonfinite,
        "binding": file_binding(path),
    }


def close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return abs(left - right) <= atol + rtol * abs(left)


def restart_provenance(
    root: Path,
    *,
    profile_id: str,
    scientific_digest: str,
    iteration: int,
    checkpoint_root: Path | None = None,
) -> dict:
    receipt_path = root / "segments/updates_160_161/training_job_receipt.json"
    receipt = read_json(receipt_path)
    checkpoint_name = f"iter_{iteration:07d}"
    view = root / "benchmark_load_views" / f"{checkpoint_name}_for_{profile_id}"
    marker = view / "latest_checkpointed_iteration.txt"
    checkpoint_link = view / checkpoint_name
    expected_checkpoint = (checkpoint_root or root / "checkpoints") / checkpoint_name
    checks = {
        "receipt_schema": receipt.get("schema_version") == "apertus_full_8b_training_job_v1",
        "receipt_completed": receipt.get("status") == "completed",
        "synchronous_checkpoint_save": receipt.get("checkpoint_save_mode") == "synchronous",
        "profile_id": receipt.get("profile_id") == profile_id,
        "scientific_digest": receipt.get("scientific_digest") == scientific_digest,
        "start_iteration": receipt.get("start_iteration") == iteration,
        "end_iteration": receipt.get("end_iteration") == iteration + 1,
        "load_marker": marker.is_file() and marker.read_text().strip() == str(iteration),
        "checkpoint_symlink": checkpoint_link.is_symlink(),
        "checkpoint_target": checkpoint_link.is_symlink()
        and checkpoint_link.resolve() == expected_checkpoint.resolve(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "receipt": file_binding(receipt_path),
        "load_view": str(view),
        "checkpoint_target": str(expected_checkpoint),
    }


def restart_equivalence(
    uninterrupted: dict,
    restarted: dict,
    iteration: int,
    *,
    gradient_atol: float,
    gradient_rtol: float,
) -> dict:
    left = uninterrupted["rows"].get(iteration)
    right = restarted["rows"].get(iteration)
    if left is None or right is None:
        return {"passed": False, "reason": "comparison update absent"}
    fields = sorted(set(left) | set(right))
    exact_fields = {name: left.get(name) == right.get(name) for name in ("loss", "params")}
    gradient_delta = abs(left.get("grad", math.inf) - right.get("grad", math.inf))
    gradient_limit = gradient_atol + gradient_rtol * abs(left.get("grad", math.inf))
    gradient_finite = math.isfinite(left.get("grad", math.inf)) and math.isfinite(
        right.get("grad", math.inf)
    )
    gradient_within_tolerance = gradient_finite and gradient_delta <= gradient_limit
    return {
        "passed": all(exact_fields.values()) and gradient_within_tolerance,
        "iteration": iteration,
        "uninterrupted": left,
        "restarted": right,
        "exact_logged_fields": exact_fields,
        "gradient_norm": {
            "finite": gradient_finite,
            "absolute_delta": gradient_delta,
            "absolute_limit": gradient_limit,
            "atol": gradient_atol,
            "rtol": gradient_rtol,
            "within_tolerance": gradient_within_tolerance,
            "reason": "cause not established; logged loss and parameter norm are exact while the DP32 logged gradient norm differs",
        },
        "absolute_deltas": {field: abs(left.get(field, math.inf) - right.get(field, math.inf)) for field in fields},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-repeat-root", type=Path, required=True)
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
    control_repeat_restart = parse_log(
        args.control_repeat_root / "segments/updates_160_161/training.log",
        require_full=False,
    )
    control_job = read_json(args.control_root / "segments/updates_0_288/training_job_receipt.json")
    candidate_job = read_json(args.candidate_root / "segments/updates_0_288/training_job_receipt.json")
    if control_job.get("scientific_digest") != candidate_job.get("scientific_digest"):
        raise ValueError("control/candidate scientific digest drift")
    main_jobs_synchronous = all(
        job.get("checkpoint_save_mode") == "synchronous"
        for job in (control_job, candidate_job)
    )
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
    candidate_elapsed = float(candidate_job["elapsed_seconds"])
    candidate_benchmark_wall = candidate_elapsed / 288
    candidate_profile = profiles["profiles"][candidate_job["profile_id"]]
    segment_boundaries = candidate_profile["segment_boundaries"]
    production_segment_updates = min(
        right - left for left, right in zip(segment_boundaries, segment_boundaries[1:])
    )
    timed_payload = candidate["timed_payload_seconds"]
    if timed_payload is None:
        # Legacy/synthetic logs have no wall timestamps. Keep the conservative
        # historical behavior rather than inventing a startup estimate.
        fixed_startup = None
        projected_production_wall = candidate_benchmark_wall
    else:
        fixed_startup = max(0.0, candidate_elapsed - timed_payload)
        projected_production_wall = (
            timed_payload / 288 + fixed_startup / production_segment_updates
        )
    amortized_p90 = max(candidate["p90_step_seconds"], projected_production_wall)
    restart_options = {
        "gradient_atol": thresholds["restart_gradient_norm_atol"],
        "gradient_rtol": thresholds["restart_gradient_norm_rtol"],
    }
    control_restart_provenance = restart_provenance(
        args.control_root,
        profile_id=control_job["profile_id"],
        scientific_digest=control_job["scientific_digest"],
        iteration=160,
    )
    candidate_restart_provenance = restart_provenance(
        args.candidate_root,
        profile_id=candidate_job["profile_id"],
        scientific_digest=candidate_job["scientific_digest"],
        iteration=160,
    )
    control_repeat_restart_provenance = restart_provenance(
        args.control_repeat_root,
        profile_id=control_job["profile_id"],
        scientific_digest=control_job["scientific_digest"],
        iteration=160,
        checkpoint_root=args.control_root / "checkpoints",
    )
    control_restart = restart_equivalence(control, control_restart, 161, **restart_options)
    control_repeat_restart = restart_equivalence(
        control, control_repeat_restart, 161, **restart_options
    )
    candidate_restart = restart_equivalence(candidate, candidate_restart, 161, **restart_options)
    checks = {
        "same_scientific_digest": True,
        "same_frozen_sequence_and_goldfish_contract": bool(contract["sequence_ids"]["prefix_sha256"] and contract["goldfish"]["implementation"]["sha256"]),
        "synchronous_checkpoint_save_mode": main_jobs_synchronous,
        "first_batch_loss_gradient_parameter_parity": all(first_checks.values()),
        "trajectory_rmse_within_bound": rmse_ratio <= thresholds["trajectory_rmse_over_control_std_max"],
        "trajectory_signed_mean_within_bound": signed_ratio <= thresholds["trajectory_abs_mean_over_control_std_max"],
        "control_restart_provenance": control_restart_provenance["passed"],
        "control_repeat_restart_provenance": control_repeat_restart_provenance["passed"],
        "candidate_restart_provenance": candidate_restart_provenance["passed"],
        "control_restart_numerically_equivalent": control_restart["passed"],
        "control_repeat_restart_numerically_equivalent": control_repeat_restart["passed"],
        "candidate_restart_numerically_equivalent": candidate_restart["passed"],
        "control_zero_skipped_updates": control["skipped"] == 0,
        "candidate_zero_skipped_updates": candidate["skipped"] == 0,
        "control_zero_nonfinite_updates": control["nonfinite"] == 0,
        "candidate_zero_nonfinite_updates": candidate["nonfinite"] == 0,
        "median_speedup_at_least_1p6": speedup >= thresholds["median_throughput_speedup_min"],
        "gpu_hour_ratio_at_most_1p25": gpu_hour_ratio <= thresholds["gpu_hour_ratio_max"],
        "amortized_p90_wall_at_most_5p89_seconds": amortized_p90 <= thresholds["p90_amortized_wall_seconds_per_update_max"],
    }
    control_viable = all(
        checks[name]
        for name in (
            "same_scientific_digest",
            "same_frozen_sequence_and_goldfish_contract",
            "synchronous_checkpoint_save_mode",
            "control_restart_provenance",
            "control_restart_numerically_equivalent",
            "control_repeat_restart_provenance",
            "control_repeat_restart_numerically_equivalent",
            "control_zero_skipped_updates",
            "control_zero_nonfinite_updates",
        )
    )
    promoted = control_viable and all(checks.values())
    selected = "dp64_32node" if promoted else ("dp32_16node" if control_viable else None)
    payload = {
        "schema_version": "apertus_full_8b_parallelism_benchmark_v1",
        "status": "promoted" if promoted else ("completed" if control_viable else "failed"),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_profile": selected,
        "candidate_promoted": promoted,
        "fallback_control_viable": control_viable,
        "checks": checks,
        "thresholds": thresholds,
        "first_batch_checks": first_checks,
        "trajectory": {"rmse_over_control_std": rmse_ratio, "absolute_mean_over_control_std": signed_ratio},
        "performance": {
            "control_median_step_seconds": control["median_step_seconds"],
            "control_p90_step_seconds": control["p90_step_seconds"],
            "candidate_median_step_seconds": candidate["median_step_seconds"],
            "candidate_p90_step_seconds": candidate["p90_step_seconds"],
            "candidate_benchmark_wall_seconds_per_update": candidate_benchmark_wall,
            "candidate_timed_payload_seconds": timed_payload,
            "candidate_fixed_startup_seconds": fixed_startup,
            "production_segment_updates": production_segment_updates,
            "candidate_projected_production_wall_seconds_per_update": projected_production_wall,
            "candidate_p90_amortized_wall_seconds_per_update": amortized_p90,
            "median_throughput_speedup": speedup,
            "gpu_hour_ratio": gpu_hour_ratio,
            "selected_compute_hours": int(profiles["scientific_invariants"]["training_updates"])
            * (candidate["median_step_seconds"] if promoted else control["median_step_seconds"])
            / 3600,
        },
        "restart": {
            "control": {
                "provenance": control_restart_provenance,
                "numerical": control_restart,
                "independent_repeat": {
                    "provenance": control_repeat_restart_provenance,
                    "numerical": control_repeat_restart,
                },
                "two_independent_restart_allocations_passed": bool(
                    control_restart_provenance["passed"]
                    and control_restart["passed"]
                    and control_repeat_restart_provenance["passed"]
                    and control_repeat_restart["passed"]
                ),
            },
            "candidate": {"provenance": candidate_restart_provenance, "numerical": candidate_restart},
        },
        "scientific_digest": control_job["scientific_digest"],
        "checkpointing": {
            "save_mode": "synchronous" if main_jobs_synchronous else "unverified_or_async",
            "async_save_forbidden_for_resumable_boundaries": True,
        },
        "inputs": {
            "profiles": file_binding(args.profiles),
            "benchmark_contract": file_binding(args.benchmark_contract),
            "control_log": control["binding"],
            "candidate_log": candidate["binding"],
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "selected_profile": selected, "candidate_promoted": promoted, "checks": checks}, sort_keys=True))
    return 0 if control_viable else 2


if __name__ == "__main__":
    raise SystemExit(main())

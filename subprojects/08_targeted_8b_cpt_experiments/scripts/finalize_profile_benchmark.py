#!/usr/bin/env python3
"""Finalize one profile candidate against the frozen reference trajectory."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import statistics
from pathlib import Path

from build_training_run_permit import PROFILE_CHECKS
from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from materialize_phase_cache import validate_overlay_receipt

FIELD_PATTERNS = {
    "iteration": re.compile(r"iteration\s+(\d+)/"),
    "samples": re.compile(r"consumed samples:\s*(\d+)"),
    "milliseconds": re.compile(r"elapsed time per iteration \(ms\):\s*([+\-0-9.Ee]+)"),
    "loss": re.compile(r"lm loss:\s*([+\-0-9.Ee]+)"),
    "gradient_norm": re.compile(r"grad norm:\s*([+\-0-9.Ee]+)"),
    "parameter_norm": re.compile(r"params norm:\s*([+\-0-9.Ee]+)"),
    "skipped": re.compile(r"number of skipped iterations:\s*(\d+)"),
    "nonfinite": re.compile(r"number of nan iterations:\s*(\d+)"),
}


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    require(path.is_file(), f"training log missing: {path}")
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        matches = {name: pattern.search(line) for name, pattern in FIELD_PATTERNS.items()}
        if not matches["iteration"]:
            continue
        require(all(matches.values()), f"incomplete optimizer-update log row: {line[:300]}")
        iteration = int(matches["iteration"].group(1))
        row = {
            "samples": float(matches["samples"].group(1)),
            "seconds": float(matches["milliseconds"].group(1)) / 1000,
            "loss": float(matches["loss"].group(1)),
            "gradient_norm": float(matches["gradient_norm"].group(1)),
            "parameter_norm": float(matches["parameter_norm"].group(1)),
            "skipped": float(matches["skipped"].group(1)),
            "nonfinite": float(matches["nonfinite"].group(1)),
        }
        require(all(math.isfinite(value) for value in row.values()), f"non-finite optimizer-update row: {iteration}")
        require(int(row["samples"]) == iteration * 1024, f"global sample cursor drift at update {iteration}")
        require(row["skipped"] == 0 and row["nonfinite"] == 0, f"skipped/non-finite update at {iteration}")
        require(iteration not in rows, f"duplicate optimizer update {iteration}: {path}")
        rows[iteration] = row
    require(rows, f"no optimizer updates parsed: {path}")
    return rows


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--throughput-log", type=Path, required=True)
    parser.add_argument("--reference-benchmark-contract", type=Path, required=True)
    parser.add_argument("--reference-throughput-log", type=Path, required=True)
    parser.add_argument("--uninterrupted-log", type=Path, required=True)
    parser.add_argument("--resumed-log", type=Path, required=True)
    parser.add_argument("--phase2-uninterrupted-log", type=Path, required=True)
    parser.add_argument("--phase2-resumed-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable profile benchmark receipt exists: {args.output}")
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    contract = read_json(args.benchmark_contract)
    require(experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2", "experiment contract drift")
    require(allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1", "allocation contract drift")
    require(contract.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1", "benchmark contract schema drift")
    require(contract.get("status") == "frozen" and contract.get("kind") == "profile" and contract.get("scale") == args.scale, "benchmark contract identity drift")
    require(contract.get("profile_id") == args.profile_id and contract.get("nodes") == args.nodes, "benchmark contract profile drift")
    require(contract.get("tensor_parallel") == args.tensor_parallel and contract.get("microbatch") == args.microbatch, "benchmark contract decomposition drift")
    current = executing_code_bundle()
    bundle = contract.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], "benchmark contract code-bundle drift")
    output_root = Path(str(contract.get("output_root", ""))).resolve()
    expected_paths = {
        args.preflight.resolve(): output_root / "preflight.json",
        args.throughput_log.resolve(): output_root / "throughput/driver.out",
        args.uninterrupted_log.resolve(): output_root / "restart/uninterrupted/driver.out",
        args.resumed_log.resolve(): output_root / "restart/resumed/driver.out",
        args.phase2_uninterrupted_log.resolve(): output_root / "restart/phase2_uninterrupted/driver.out",
        args.phase2_resumed_log.resolve(): output_root / "restart/phase2_resumed/driver.out",
    }
    require(all(actual == expected.resolve() for actual, expected in expected_paths.items()), "profile benchmark evidence path drift")
    preflight = read_json(args.preflight)
    require(preflight.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_preflight_v1" and preflight.get("status") == "passed", "profile benchmark preflight drift")
    require(preflight.get("contract") == file_binding(args.benchmark_contract), "profile benchmark preflight contract binding drift")
    overlay_audits: dict[str, dict[str, object]] = {}
    for phase, binding_field, root_field in (
        (1, "phase_cache_overlay_receipt", "phase_cache_root"),
        (2, "phase2_cache_overlay_receipt", "phase2_cache_root"),
    ):
        binding = contract.get(binding_field)
        require(isinstance(binding, dict), f"Phase-{phase} cache overlay binding missing")
        receipt_path = Path(str(binding.get("path", "")))
        require(receipt_path.is_file() and binding == file_binding(receipt_path), f"Phase-{phase} cache overlay binding drift")
        receipt = read_json(receipt_path)
        overlay_audits[str(phase)] = validate_overlay_receipt(
            receipt,
            phase=phase,
            overlay_root=Path(str(contract.get(root_field, ""))),
            require_pristine=False,
        )
    reference_contract = read_json(args.reference_benchmark_contract)
    require(reference_contract.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1", "reference benchmark contract schema drift")
    require(reference_contract.get("status") == "frozen" and reference_contract.get("kind") == "profile" and reference_contract.get("scale") == args.scale, "reference benchmark contract identity drift")
    reference_output_root = Path(str(reference_contract.get("output_root", ""))).resolve()
    require(args.reference_throughput_log.resolve() == (reference_output_root / "throughput/driver.out").resolve(), "reference throughput path drift")
    require(reference_contract.get("executing_code_bundle") == contract.get("executing_code_bundle"), "reference benchmark code-bundle drift")
    require(reference_contract.get("initialization_permit") == contract.get("initialization_permit"), "reference benchmark initialization drift")
    require(reference_contract.get("phase_cache_tree_sha256") == contract.get("phase_cache_tree_sha256"), "reference benchmark data trajectory drift")
    require(reference_contract.get("phase2_cache_tree_sha256") == contract.get("phase2_cache_tree_sha256"), "reference Phase-2 benchmark data trajectory drift")
    require(reference_contract.get("peak_lr") == contract.get("peak_lr") and reference_contract.get("floor_lr") == contract.get("floor_lr"), "reference benchmark LR drift")
    if args.scale == "8b":
        require(args.reference_benchmark_contract.resolve() == args.benchmark_contract.resolve(), "8B frozen profile must self-reference")
        reference_mode = "frozen_dp32_profile_self_reference_no_geometry_change"
    else:
        require(reference_contract.get("profile_id") == "1p5b_tp1_1node", "1.5B profile reference is not the frozen one-node candidate")
        reference_mode = "one_node_fixed_batch_reference"
    throughput = parse_log(args.throughput_log)
    reference = parse_log(args.reference_throughput_log)
    require(set(range(1, 257)) <= set(throughput) and set(range(1, 257)) <= set(reference), "profile benchmark lacks updates 1..256")
    thresholds = experiment["profile_selection"]["parity_thresholds"]
    loss_diffs = [throughput[update]["loss"] - reference[update]["loss"] for update in range(1, 257)]
    gradient_relative = [
        abs(throughput[update]["gradient_norm"] - reference[update]["gradient_norm"])
        / max(abs(reference[update]["gradient_norm"]), 1e-12)
        for update in range(1, 257)
    ]
    loss_rmse = math.sqrt(statistics.fmean(value * value for value in loss_diffs))
    loss_signed_mean = statistics.fmean(loss_diffs)
    gradient_relative_median = statistics.median(gradient_relative)
    uninterrupted = parse_log(args.uninterrupted_log)
    resumed = parse_log(args.resumed_log)
    require(2 in uninterrupted and 2 in resumed, "restart comparison update 2 missing")
    restart = {
        "loss_abs": abs(uninterrupted[2]["loss"] - resumed[2]["loss"]),
        "parameter_norm_abs": abs(uninterrupted[2]["parameter_norm"] - resumed[2]["parameter_norm"]),
        "gradient_norm_abs": abs(uninterrupted[2]["gradient_norm"] - resumed[2]["gradient_norm"]),
    }
    restart_gradient_limit = float(thresholds["restart_gradient_norm_atol"]) + float(thresholds["restart_gradient_norm_rtol"]) * abs(uninterrupted[2]["gradient_norm"])
    phase2_uninterrupted = parse_log(args.phase2_uninterrupted_log)
    phase2_resumed = parse_log(args.phase2_resumed_log)
    require(3 in phase2_uninterrupted and 3 in phase2_resumed, "Phase-2 restart comparison update 3 missing")
    require(
        "phase_local_samples=0" in args.phase2_uninterrupted_log.read_text(encoding="utf-8", errors="replace")
        and "phase_local_samples=1024" in args.phase2_resumed_log.read_text(encoding="utf-8", errors="replace"),
        "Phase-2 entry/resume cursor evidence missing",
    )
    phase2_restart = {
        "loss_abs": abs(phase2_uninterrupted[3]["loss"] - phase2_resumed[3]["loss"]),
        "parameter_norm_abs": abs(phase2_uninterrupted[3]["parameter_norm"] - phase2_resumed[3]["parameter_norm"]),
        "gradient_norm_abs": abs(phase2_uninterrupted[3]["gradient_norm"] - phase2_resumed[3]["gradient_norm"]),
    }
    phase2_gradient_limit = float(thresholds["restart_gradient_norm_atol"]) + float(thresholds["restart_gradient_norm_rtol"]) * abs(phase2_uninterrupted[3]["gradient_norm"])
    checks = {
        "fixed_batch_loss_parity": loss_rmse <= float(thresholds["trajectory_loss_rmse_max"]) and abs(loss_signed_mean) <= float(thresholds["trajectory_loss_signed_mean_abs_max"]),
        "fixed_batch_gradient_parity": gradient_relative_median <= float(thresholds["trajectory_gradient_relative_median_max"]),
        "restart_next_step_parity": restart["loss_abs"] <= float(thresholds["restart_loss_abs_max"]) and restart["parameter_norm_abs"] <= float(thresholds["restart_parameter_norm_abs_max"]) and restart["gradient_norm_abs"] <= restart_gradient_limit,
        "phase2_entry_and_restart_parity": phase2_restart["loss_abs"] <= float(thresholds["restart_loss_abs_max"]) and phase2_restart["parameter_norm_abs"] <= float(thresholds["restart_parameter_norm_abs_max"]) and phase2_restart["gradient_norm_abs"] <= phase2_gradient_limit,
        "sample_and_mask_cursor_continuity": all(int(throughput[update]["samples"]) == update * 1024 for update in range(1, 257)) and contract.get("phase_cache_receipt") is not None and contract.get("phase2_cache_receipt") is not None,
        "zero_skipped_or_nonfinite_updates": True,
        "tokens_per_gpu_hour_measured": True,
    }
    require(set(checks) == set(PROFILE_CHECKS), "profile check set drift")
    passed = all(checks.values())
    times = [throughput[update]["seconds"] for update in range(33, 257)]
    throughput_wall_path = output_root / "throughput/wall_seconds.txt"
    require(throughput_wall_path.is_file(), "profile benchmark wall-clock evidence missing")
    throughput_wall_seconds = int(throughput_wall_path.read_text(encoding="utf-8").strip())
    require(throughput_wall_seconds > 0, "profile benchmark wall-clock evidence invalid")
    median_step = statistics.median(times)
    p90_step = percentile(times, 0.90)
    tokens_per_gpu_hour = 4_194_304 / median_step * 3600 / (args.nodes * 4)
    world = args.nodes * 4
    require(world % args.tensor_parallel == 0, "profile world/TP drift")
    data_parallel = world // args.tensor_parallel
    payload = {
        "schema_version": "apertus_hard_h_to_g_profile_benchmark_v1",
        "status": "passed" if passed else "rejected",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "executing_code_bundle": current,
        "profile": {
            "profile_id": args.profile_id, "nodes": args.nodes, "gpus_per_node": 4,
            "tensor_parallel": args.tensor_parallel, "pipeline_parallel": 1,
            "context_parallel": 1, "data_parallel": data_parallel,
            "microbatch": args.microbatch,
            "gradient_accumulation_microbatches": 1024 // (data_parallel * args.microbatch),
        },
        "benchmark_contract": file_binding(args.benchmark_contract),
        "measurement": {
            "updates": 256, "discarded_warmup_updates": 32,
            "median_step_seconds": median_step, "p90_step_seconds": p90_step,
            "tokens_per_gpu_hour": tokens_per_gpu_hour,
            "production_cadence_wall_seconds": throughput_wall_seconds,
            "production_cadence_save_interval": 119,
            "production_cadence_eval_interval": 25,
        },
        "trajectory_parity": {
            "loss_rmse": loss_rmse, "loss_signed_mean": loss_signed_mean,
            "gradient_relative_median": gradient_relative_median,
            "thresholds": thresholds, "reference_mode": reference_mode,
        },
        "restart_parity": {**restart, "gradient_norm_limit": restart_gradient_limit, "thresholds": thresholds},
        "phase2_entry_and_restart_parity": {**phase2_restart, "gradient_norm_limit": phase2_gradient_limit, "entry_phase_local_samples": 0, "resume_phase_local_samples": 1024, "thresholds": thresholds},
        "qualification_cache_overlay_audits": overlay_audits,
        "checks": checks,
        "evidence": [
            file_binding(args.benchmark_contract), file_binding(args.throughput_log),
            file_binding(args.reference_benchmark_contract), file_binding(args.reference_throughput_log), file_binding(args.uninterrupted_log),
            file_binding(args.resumed_log), file_binding(args.phase2_uninterrupted_log),
            file_binding(args.phase2_resumed_log), file_binding(throughput_wall_path),
            file_binding(args.preflight),
        ],
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

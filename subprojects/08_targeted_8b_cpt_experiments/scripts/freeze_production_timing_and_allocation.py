#!/usr/bin/env python3
"""Freeze measured segment timing and the bounded one-successor schedule."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

from build_training_run_permit import validate_permit
from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from producer_bundle_compatibility import load_authority, require_accepted_producer

BOUNDARIES = [0, 952, 1904, 2261, 3218, 3456, 3694]
ALLOCATION_SECONDS = 43_200
RESERVE_SECONDS = 1_200
BENCHMARK_UPDATES = 256
CONSERVATIVE_MULTIPLIER = 1.15
MINIMUM_RUNTIME_FRACTION = 0.75
FIRST_ALLOCATION_1P5B_STEP_SECONDS = 26.5
FIRST_ALLOCATION_1P5B_CADENCE_OVERHEAD_SECONDS = 600


def future_json_binding(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def same_bundle(value: dict[str, Any], current: dict[str, Any], label: str) -> None:
    bundle = value.get("executing_code_bundle")
    require(
        isinstance(bundle, dict)
        and bundle.get("root") == current["root"]
        and bundle.get("tree_sha256") == current["tree_sha256"],
        f"{label} code-bundle drift",
    )


def selected_profile_benchmark(
    promotion: dict[str, Any], accepted_producers: set[tuple[str, str, str, int, str]]
) -> tuple[dict[str, Any], Path]:
    selection = promotion.get("selection")
    require(isinstance(selection, dict), "profile selection missing")
    profile_id = str(selection.get("profile_id", ""))
    bindings = promotion.get("candidate_receipts")
    require(
        isinstance(bindings, list) and bindings, "profile candidate bindings missing"
    )
    matches: list[tuple[dict[str, Any], Path]] = []
    for binding in bindings:
        require(isinstance(binding, dict), "profile candidate binding malformed")
        path = Path(str(binding.get("path", "")))
        require(
            path.is_file() and binding == file_binding(path),
            "profile candidate binding drift",
        )
        candidate = read_json(path)
        require_accepted_producer(candidate, accepted_producers, "profile candidate")
        if candidate.get("profile", {}).get("profile_id") == profile_id:
            matches.append((candidate, path))
    require(len(matches) == 1, "selected profile candidate is not uniquely bound")
    candidate, path = matches[0]
    require(
        candidate.get("status") == "passed", "selected profile benchmark did not pass"
    )
    measurement = candidate.get("measurement")
    require(isinstance(measurement, dict), "selected profile measurement missing")
    require(
        measurement.get("updates") == BENCHMARK_UPDATES,
        "profile benchmark update horizon drift",
    )
    require(
        measurement.get("production_cadence_save_interval") == 119,
        "profile benchmark save cadence drift",
    )
    require(
        measurement.get("production_cadence_eval_interval") == 25,
        "profile benchmark evaluation cadence drift",
    )
    require(
        float(measurement.get("p90_step_seconds", 0)) > 0,
        "profile p90 step time missing",
    )
    require(
        float(measurement.get("median_step_seconds", 0)) > 0,
        "profile median step time missing",
    )
    require(
        int(measurement.get("production_cadence_wall_seconds", 0)) > 0,
        "profile wall time missing",
    )
    return candidate, path


def validate_run_permit(
    path: Path, scale: str, current: dict[str, Any]
) -> dict[str, Any]:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_hard_h_to_g_training_run_permit_v1"
        and value.get("status") == "passed"
        and value.get("scale") == scale,
        f"{scale} run-permit identity drift",
    )
    profile = value["profile"]
    lr = value["learning_rate"]
    validate_permit(
        value,
        scale=scale,
        nodes=int(profile["nodes"]),
        tensor_parallel=int(profile["tensor_parallel"]),
        microbatch=int(profile["microbatch"]),
        peak_lr=str(lr["peak"]),
        floor_lr=str(lr["floor"]),
    )
    same_bundle(value, current, f"{scale} run permit")
    return value


def measured_scale(
    scale: str,
    promotion_path: Path,
    run_permit_path: Path,
    current: dict[str, Any],
    accepted_producers: set[tuple[str, str, str, int, str]],
) -> dict[str, Any]:
    promotion = read_json(promotion_path)
    permit = validate_run_permit(run_permit_path, scale, current)
    if scale == "8b":
        require(
            promotion.get("schema_version") == "apertus_hard_h_to_g_profile_promotion_v1"
            and promotion.get("status") == "promoted"
            and promotion.get("scale") == scale,
            "8b profile-promotion drift",
        )
        require_accepted_producer(promotion, accepted_producers, "8b profile promotion")
        candidate, candidate_path = selected_profile_benchmark(
            promotion, accepted_producers
        )
        require(
            permit.get("profile") == promotion.get("selection"),
            "8b run permit/profile selection drift",
        )
        measurement = candidate["measurement"]
        timing_basis = "measured_production_equivalent_profile"
        qualification_overhead_seconds = 0
        profile_authority = file_binding(promotion_path)
        benchmark_authority = file_binding(candidate_path)
    else:
        require(
            promotion.get("schema_version")
            == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1"
            and promotion.get("status") == "frozen"
            and promotion.get("kind") == "profile"
            and promotion.get("scale") == "1p5b"
            and promotion.get("profile_id") == "1p5b_tp1_1node",
            "1.5B first-allocation profile contract drift",
        )
        require_accepted_producer(
            promotion, accepted_producers, "1.5B profile candidate"
        )
        require(
            (
                permit["profile"]["profile_id"],
                permit["profile"]["nodes"],
                permit["profile"]["tensor_parallel"],
                permit["profile"]["microbatch"],
            )
            == ("1p5b_tp1_1node", 1, 1, 8),
            "1.5B run permit/profile candidate drift",
        )
        cadence_wall = math.ceil(
            BENCHMARK_UPDATES * FIRST_ALLOCATION_1P5B_STEP_SECONDS
            + FIRST_ALLOCATION_1P5B_CADENCE_OVERHEAD_SECONDS
        )
        measurement = {
            "updates": BENCHMARK_UPDATES,
            "discarded_warmup_updates": 32,
            "median_step_seconds": FIRST_ALLOCATION_1P5B_STEP_SECONDS,
            "p90_step_seconds": FIRST_ALLOCATION_1P5B_STEP_SECONDS,
            "production_cadence_wall_seconds": cadence_wall,
            "production_cadence_save_interval": 119,
            "production_cadence_eval_interval": 25,
            "status": "predeclared_conservative_startup_bound_pending_in_allocation_measurement",
        }
        timing_basis = "predeclared_first_allocation_bound_then_measured_eta_refresh"
        qualification_overhead_seconds = cadence_wall
        profile_authority = file_binding(promotion_path)
        benchmark_authority = None
    benchmark_wall = int(measurement["production_cadence_wall_seconds"])
    median_step = float(measurement["median_step_seconds"])
    p90_step = float(measurement["p90_step_seconds"])
    segments = []
    for segment_id, (start, end) in enumerate(pairwise(BOUNDARIES)):
        updates = end - start
        blocks = math.ceil(updates / BENCHMARK_UPDATES)
        conservative = math.ceil(blocks * benchmark_wall * CONSERVATIVE_MULTIPLIER)
        if scale == "1p5b" and segment_id == 0:
            conservative += qualification_overhead_seconds
        require(
            conservative + RESERVE_SECONDS < ALLOCATION_SECONDS,
            f"{scale} segment {segment_id} does not fit a 12-hour allocation conservatively",
        )
        segments.append(
            {
                "segment_id": segment_id,
                "start_update": start,
                "exit_update": end,
                "updates": updates,
                "benchmark_equivalent_blocks": blocks,
                "compute_only_p90_seconds": math.ceil(updates * p90_step),
                "minimum_train_seconds": max(
                    60,
                    math.floor(updates * median_step * MINIMUM_RUNTIME_FRACTION),
                ),
                "conservative_wall_seconds": conservative,
            }
        )
    return {
        "scale": scale,
        "profile": permit["profile"],
        "learning_rate": permit["learning_rate"],
        "profile_authority": profile_authority,
        "selected_profile_benchmark": benchmark_authority,
        "training_run_permit": file_binding(run_permit_path),
        "timing_basis": timing_basis,
        "qualification_overhead_seconds_in_segment_0": qualification_overhead_seconds,
        "benchmark_measurement": measurement,
        "segments": segments,
        "compute_only_p90_seconds": sum(
            row["compute_only_p90_seconds"] for row in segments
        ),
        "conservative_training_seconds": sum(
            row["conservative_wall_seconds"] for row in segments
        ),
    }


def successor_rows(scales: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    require(scales and set(scales) <= {"8b", "1p5b"}, "invalid timing scale inventory")
    for scale in scales:
        segments = scales[scale]["segments"]
        for source, target in pairwise(segments):
            maximum_hold = (
                ALLOCATION_SECONDS
                - target["conservative_wall_seconds"]
                - RESERVE_SECONDS
            )
            source_trigger = max(60, source["conservative_wall_seconds"] - maximum_hold)
            require(
                0 < source_trigger < source["conservative_wall_seconds"],
                "invalid source trigger",
            )
            require(
                source["conservative_wall_seconds"] - source_trigger <= maximum_hold,
                "holder can outlive source budget",
            )
            rows.append(
                {
                    "scale": scale,
                    "source_segment_id": source["segment_id"],
                    "target_segment_id": target["segment_id"],
                    "source_job_dependency_template": f"after:<source-job>+{math.ceil(source_trigger / 60)}",
                    "source_trigger_seconds": source_trigger,
                    "maximum_hold_seconds": maximum_hold,
                    "target_runtime_seconds": target["conservative_wall_seconds"],
                    "reserve_seconds": RESERVE_SECONDS,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--profile-promotion", type=Path, required=True)
    parser.add_argument("--run-permit", type=Path, required=True)
    parser.add_argument("--submission-dry-run-receipt", type=Path, required=True)
    parser.add_argument("--producer-compatibility", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--allocation-output", type=Path, required=True)
    args = parser.parse_args()
    require(
        not args.timing_output.exists(),
        f"immutable timing output exists: {args.timing_output}",
    )
    require(
        not args.allocation_output.exists(),
        f"immutable allocation output exists: {args.allocation_output}",
    )
    experiment = read_json(args.experiment)
    allocation = read_json(args.allocation)
    require(
        experiment.get("schema_version") == "apertus_hard_h_to_g_replication_v2",
        "experiment contract drift",
    )
    require(
        allocation.get("schema_version") == "apertus_hard_h_to_g_allocation_v1",
        "allocation contract drift",
    )
    require(
        allocation.get("handoff_formula", {}).get("values_status")
        == "blocked_until_production_equivalent_measurements",
        "allocation handoff policy drift",
    )
    current = executing_code_bundle()
    _, accepted_producers = load_authority(args.producer_compatibility, current)
    dry_run = read_json(args.submission_dry_run_receipt)
    require(
        dry_run.get("schema_version") == "apertus_hard_h_to_g_submission_dry_run_v1"
        and dry_run.get("status") == "passed"
        and dry_run.get("scale") == args.scale
        and isinstance(dry_run.get("checks"), dict)
        and all(dry_run["checks"].values()),
        "submission dry-run receipt drift",
    )
    same_bundle(dry_run, current, "submission dry run")
    scales = {
        args.scale: measured_scale(
            args.scale,
            args.profile_promotion,
            args.run_permit,
            current,
            accepted_producers,
        )
    }
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    common = {
        "created_at": created,
        "executing_code_bundle": current,
        "experiment": file_binding(args.experiment),
        "allocation_contract": file_binding(args.allocation),
        "producer_bundle_compatibility": file_binding(args.producer_compatibility),
        "submission_dry_run_receipt": file_binding(args.submission_dry_run_receipt),
    }
    timing = {
        "schema_version": "apertus_hard_h_to_g_production_timing_v1",
        "status": "passed",
        "scale": args.scale,
        **common,
        "method": {
            "production_equivalent_benchmark_updates": BENCHMARK_UPDATES,
            "save_interval": 119,
            "evaluation_interval": 25,
            "conservative_formula": "ceil(ceil(segment_updates/256)*measured_256_update_wall_seconds*1.15)",
            "minimum_runtime_formula": "max(60,floor(segment_updates*measured_median_step_seconds*0.75))",
            "minimum_runtime_is_lower_completion_tripwire": True,
            "conservative_wall_is_upper_allocation_budget": True,
            "queue_wait_excluded": True,
            "timing_basis": scales[args.scale]["timing_basis"],
        },
        "scales": scales,
        "clocks": {
            "compute_only_is_measured": args.scale == "8b",
            "compute_only_is_predeclared_until_first_allocation": args.scale == "1p5b",
            "training_complete_excludes_queue_wait": True,
            "evidence_complete_pending_live_evaluation_backlog": True,
        },
    }
    schedule = {
        "schema_version": "apertus_hard_h_to_g_allocation_schedule_v1",
        "status": "passed",
        "scale": args.scale,
        **common,
        "timing_receipt": future_json_binding(args.timing_output, timing),
        "allocation_seconds": ALLOCATION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "initial_segments_may_be_submitted_independently": True,
        "maximum_pending_delayed_successors": 1,
        "successors": successor_rows(scales),
        "invariants": {
            "direct_normal_holder": True,
            "debug_timer_for_normal_holder": False,
            "source_trigger_uses_conservative_source_wall_time": True,
            "holder_verifies_checkpoint_permit_and_target_cache": True,
            "holder_requires_target_runtime_plus_reserve": True,
            "at_most_one_delayed_successor": True,
            "sbatch_test_only_passed_without_manifest_mutation": True,
        },
    }
    write_json_atomic(args.timing_output, timing)
    write_json_atomic(args.allocation_output, schedule)
    print(args.timing_output)
    print(args.allocation_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze a two-update uninterrupted-vs-resume proof for targeted DP32 CPT."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

from contract_utils import file_binding, read_json, require, write_json_atomic


def completed_job(path: Path, *, start: int, end: int, digest: str) -> dict:
    value = read_json(path)
    require(
        value.get("schema_version") == "apertus_full_8b_training_job_v1"
        and value.get("status") == "completed"
        and value.get("profile_id") == "dp32_16node"
        and value.get("scientific_digest") == digest
        and int(value.get("start_iteration", -1)) == start
        and int(value.get("end_iteration", -1)) == end
        and value.get("checkpoint_save_mode") == "synchronous"
        and value.get("allocation", {}).get("single_leaf_switch") is True,
        f"training-job receipt drift: {path}",
    )
    return value


def validation_panels(log_path: Path, iteration: int) -> set[str]:
    pattern = re.compile(
        rf"validation loss at iteration\s+{iteration}\s+\[([^\]]+)\]"
    )
    return {
        match.group(1)
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if (match := pattern.search(line))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--base-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable restart-smoke receipt exists: {args.output}")
    expected_base = 0 if args.experiment == "A" else 9536
    require(args.base_iteration == expected_base, "experiment/base-iteration drift")
    checkpoint_iteration = expected_base + 1
    comparison_iteration = expected_base + 2

    recipe = read_json(args.recipe)
    profiles = read_json(args.profiles)
    selected = read_json(args.selected_profile)
    schedule = read_json(args.schedule_manifest)
    validation = read_json(args.validation_manifest)
    require(selected.get("status") == "frozen", "selected profile is not frozen")
    selection = selected.get("selection", {})
    require(
        selection.get("profile_id") == "dp32_16node"
        and int(selection.get("nodes", -1)) == 16
        and int(selection.get("data_parallel", -1)) == 32,
        "restart smoke is not bound to DP32/16-node geometry",
    )
    digest = str(selection.get("scientific_digest", ""))
    require(len(digest) == 64, "selected scientific digest missing")
    thresholds = profiles.get("restart_parity", {})
    require(
        thresholds.get("loss_and_parameter_norm_require_exact_logged_equality") is True
        and float(thresholds.get("gradient_norm_atol", math.nan)) == 0.001
        and float(thresholds.get("gradient_norm_rtol", math.nan)) == 0.02,
        "predeclared restart thresholds drift",
    )
    arms = {row["arm_id"]: row for row in schedule.get("arms", [])}
    require("D0_mixed" in arms, "D0 schedule missing")
    arm = arms["D0_mixed"]
    require(
        int(arm.get("optimizer_updates", -1)) >= comparison_iteration
        and int(recipe["batch_and_parallelism"]["global_batch_sequences"]) == 1024,
        "schedule is too short for restart smoke",
    )
    expected_panels = {row["name"] for row in validation.get("panels", [])}
    require(
        validation.get("status") == "frozen" and len(expected_panels) == 13,
        "frozen 13-panel validation manifest drift",
    )

    control_segment = args.control_root / f"segments/updates_{expected_base}_{comparison_iteration}"
    resumed_segment = args.resume_root / f"segments/updates_{checkpoint_iteration}_{comparison_iteration}"
    control_receipt = control_segment / "training_job_receipt.json"
    resumed_receipt = resumed_segment / "training_job_receipt.json"
    completed_job(control_receipt, start=expected_base, end=comparison_iteration, digest=digest)
    completed_job(resumed_receipt, start=checkpoint_iteration, end=comparison_iteration, digest=digest)

    sys.path.insert(0, str(args.code_root / "subprojects/07_full_8b_cpt/scripts"))
    from finalize_parallelism_benchmark import parse_log, restart_equivalence  # pylint: disable=import-error,import-outside-toplevel

    control = parse_log(control_segment / "training.log", require_full=False)
    resumed = parse_log(resumed_segment / "training.log", require_full=False)
    numerical = restart_equivalence(
        control,
        resumed,
        comparison_iteration,
        gradient_atol=float(thresholds["gradient_norm_atol"]),
        gradient_rtol=float(thresholds["gradient_norm_rtol"]),
    )
    checkpoint_name = f"iter_{checkpoint_iteration:07d}"
    control_checkpoint = args.control_root / "checkpoints" / checkpoint_name
    view = args.resume_root / "benchmark_load_views" / f"{checkpoint_name}_for_dp32_16node"
    marker = view / "latest_checkpointed_iteration.txt"
    checkpoint_link = view / checkpoint_name
    cursor_checks = {
        "synchronous_control_checkpoint_metadata_exists": (control_checkpoint / ".metadata").is_file(),
        "load_marker_is_exact": marker.is_file() and marker.read_text().strip() == str(checkpoint_iteration),
        "load_view_targets_exact_control_checkpoint": checkpoint_link.is_symlink() and checkpoint_link.resolve() == control_checkpoint.resolve(),
        "control_job_spans_checkpoint": (
            read_json(control_receipt).get("start_iteration") == expected_base
            and read_json(control_receipt).get("end_iteration") == comparison_iteration
        ),
        "resumed_job_started_at_checkpoint": read_json(resumed_receipt).get("start_iteration") == checkpoint_iteration,
        "schedule_slot_cursor_is_exact": checkpoint_iteration * 1024 < int(arm["training_slots"]),
    }
    control_boundary_panels = validation_panels(control_segment / "training.log", checkpoint_iteration)
    boundary_checks = {
        "control_checkpoint_metadata_exists": (control_checkpoint / ".metadata").is_file(),
        "all_frozen_panels_ran_before_control_checkpoint": (
            bool(expected_panels) and control_boundary_panels == expected_panels
        ),
        "resume_loads_exact_uninterrupted_control_checkpoint": (
            checkpoint_link.is_symlink() and checkpoint_link.resolve() == control_checkpoint.resolve()
        ),
    }
    uninterrupted_ok = (
        expected_base + 1 in control["rows"]
        and comparison_iteration in control["rows"]
        and control["skipped"] == 0
        and control["nonfinite"] == 0
    )
    resume_ok = (
        comparison_iteration in resumed["rows"]
        and resumed["skipped"] == 0
        and resumed["nonfinite"] == 0
        and all(cursor_checks.values())
    )
    checks = {
        "uninterrupted_two_update_run_passed": uninterrupted_ok,
        "resume_from_exact_control_checkpoint_passed": resume_ok,
        "control_checkpoint_boundary_proven": all(boundary_checks.values()),
        "checkpoint_sample_cursor_exact": all(cursor_checks.values()),
        "first_post_checkpoint_update_within_frozen_bounds": numerical["passed"],
        "nonfinite_updates": control["nonfinite"] + resumed["nonfinite"],
        "skipped_updates": control["skipped"] + resumed["skipped"],
    }
    passed = (
        uninterrupted_ok
        and resume_ok
        and all(boundary_checks.values())
        and numerical["passed"]
        and checks["nonfinite_updates"] == 0
        and checks["skipped_updates"] == 0
    )
    payload = {
        "schema_version": "targeted_8b_restart_smoke_v1",
        "status": "passed" if passed else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": args.experiment,
        "profile_id": "dp32_16node",
        "base_iteration": expected_base,
        "checkpoint_iteration": checkpoint_iteration,
        "comparison_iteration": comparison_iteration,
        "code_bundle_receipt": file_binding(args.code_bundle_receipt),
        "recipe": file_binding(args.recipe),
        "profiles": file_binding(args.profiles),
        "selected_profile": file_binding(args.selected_profile),
        "schedule_manifest": file_binding(args.schedule_manifest),
        "validation_manifest": file_binding(args.validation_manifest),
        "checks": checks,
        "cursor_checks": cursor_checks,
        "boundary_checks": boundary_checks,
        "boundary_validation_panels": {
            "expected": sorted(expected_panels),
            "control": sorted(control_boundary_panels),
        },
        "comparison_design": "resume_from_exact_uninterrupted_control_checkpoint",
        "numerical_equivalence": numerical,
        "thresholds": thresholds,
        "inputs": {
            "control_job_receipt": file_binding(control_receipt),
            "resumed_job_receipt": file_binding(resumed_receipt),
            "control_log": control["binding"],
            "resumed_log": resumed["binding"],
        },
        "allocation": {
            "normal_nodes": 16,
            "single_leaf": True,
            "single_allocation_for_control_and_resume": True,
            "total_optimizer_updates_executed": 3,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"ok": passed, "checks": checks}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze the GreekMMLU plateau set and its predeclared full-panel confirmation."""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from canonical_evidence import completed_result
from contract_utils import (
    file_binding,
    read_json,
    require,
    require_file_binding,
    write_json_atomic,
)
from run_greekmmlu_evaluator import (
    ALL_UPDATES,
    FULL_CLEAN_UPDATES,
    execute_frozen_greekmmlu,
)
from run_greekmmlu_fallback_evaluator import FALLBACK_UPDATES
from validate_greekmmlu_sentinels import bootstrap_mean, prediction_map


def fallback_id(iteration: int) -> str:
    return f"greekmmlu_full_fallback_i{iteration:07d}"


def selected_predictions(
    run_root: Path,
    *,
    scale: str,
    iteration: int,
    full_panel_required: bool,
    selected_size: int | None,
) -> tuple[Path, dict[str, float], dict[str, object]]:
    if full_panel_required and iteration in FALLBACK_UPDATES:
        canonical_iteration = max(3218, iteration)
        result_path, result = completed_result(
            run_root,
            evaluator_id=fallback_id(iteration),
            iteration=canonical_iteration,
            schema="apertus_hard_h_to_g_greekmmlu_fallback_evaluation_v1",
            scale=scale,
        )
        require(
            result.get("target_iteration") == iteration
            and result.get("action") == "full_panel_scored",
            f"{iteration}: full fallback did not run",
        )
        view_name = "full_clean"
    else:
        result_path, result = completed_result(
            run_root,
            evaluator_id="greekmmlu",
            iteration=iteration,
            schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
            scale=scale,
        )
        view_name = "full_clean" if full_panel_required else f"sentinel_{selected_size}"
    evaluation_path = require_file_binding(result["evaluation"])
    evaluation = read_json(evaluation_path)
    prediction_path = require_file_binding(evaluation["views"][view_name])
    expected_count = 16_159 if view_name == "full_clean" else int(selected_size or 0)
    return (
        prediction_path,
        prediction_map(prediction_path, expected_count),
        {
            "canonical_result": file_binding(result_path),
            "evaluation": file_binding(evaluation_path),
            "predictions": file_binding(prediction_path),
            "view": view_name,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--clean-examples", type=Path, required=True)
    parser.add_argument("--sentinel-manifest", type=Path, required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--joint-calibration-authority", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.iteration == 3694, "plateau confirmation must run at update 3694")
    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")

    calibration_result_path, calibration_result = completed_result(
        args.run_root,
        evaluator_id="greekmmlu_calibration",
        iteration=3218,
        schema="apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1",
        scale=args.scale,
    )
    calibration_path = require_file_binding(calibration_result["calibration"])
    calibration = read_json(calibration_path)
    require(
        calibration.get("schema_version") == "apertus_greekmmlu_sentinel_calibration_v1"
        and calibration.get("status") == "passed"
        and calibration.get("scale") == args.scale,
        "plateau calibration drift",
    )
    joint = read_json(args.joint_calibration_authority)
    require(
        joint.get("schema_version")
        == "apertus_greekmmlu_sentinel_calibration_authority_v1"
        and joint.get("status") == "passed"
        and joint.get("scope") == "both_scales"
        and joint.get("calibrations", {}).get(args.scale, {}).get("canonical_result")
        == file_binding(calibration_result_path),
        "joint calibration authority drift",
    )
    joint_trajectory = joint.get("cross_scale_trajectory")
    require(
        isinstance(joint_trajectory, dict)
        and joint_trajectory.get("mode") in {"sentinel_pair", "full_clean"},
        "joint trajectory decision missing",
    )
    full_panel_required = joint_trajectory["mode"] == "full_clean"
    selected_size = joint_trajectory.get("selected_size")
    if full_panel_required:
        require(selected_size is None, "full fallback retained a selected sentinel")
    else:
        require(selected_size in {4096, 8192}, "selected sentinel size drift")

    predictions: dict[int, dict[str, float]] = {}
    source_bindings: dict[str, object] = {}
    for update in sorted(ALL_UPDATES):
        _path, values, binding = selected_predictions(
            args.run_root,
            scale=args.scale,
            iteration=update,
            full_panel_required=full_panel_required,
            selected_size=selected_size,
        )
        predictions[update] = values
        source_bindings[str(update)] = binding
    reference_ids = set(predictions[0])
    require(
        all(set(values) == reference_ids for values in predictions.values()),
        "plateau prediction id sets drift",
    )
    mean_nll = {
        update: statistics.fmean(values.values()) for update, values in predictions.items()
    }
    minimum_update = min(mean_nll, key=lambda update: (mean_nll[update], update))
    comparisons: dict[str, object] = {}
    plateau_members: list[int] = []
    for update in sorted(ALL_UPDATES):
        paired = [
            predictions[update][key] - predictions[minimum_update][key]
            for key in sorted(reference_ids)
        ]
        se, low, high = bootstrap_mean(paired, 10_000, 20260814 + update)
        includes_zero = low <= 0 <= high
        if includes_zero:
            plateau_members.append(update)
        comparisons[str(update)] = {
            "choice_nll": mean_nll[update],
            "delta_from_minimum": statistics.fmean(paired),
            "bootstrap_se": se,
            "ci95": [low, high],
            "includes_zero": includes_zero,
        }
    require(plateau_members, "plateau set is empty")

    confirmation_root = args.output / "plateau_full_confirmation"
    confirmation: dict[str, object] | None = None
    try:
        if full_panel_required:
            action = "full_panel_trajectory_already_required"
        elif plateau_members == [3218]:
            action = "singleton_3218_requires_no_additional_full_score"
        else:
            target = plateau_members[0]
            action = "earliest_plateau_member_full_panel_confirmed"
            if target in FULL_CLEAN_UPDATES:
                source_path, source = completed_result(
                    args.run_root,
                    evaluator_id="greekmmlu",
                    iteration=target,
                    schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
                    scale=args.scale,
                )
                require(source.get("mode") == "full_clean", "existing full confirmation drift")
                confirmation = {
                    "target_iteration": target,
                    "reused_existing_full_score": True,
                    "canonical_result": file_binding(source_path),
                    "evaluation": source["evaluation"],
                    "summary": source["summary"],
                    "views": source["views"],
                }
            else:
                artifacts = execute_frozen_greekmmlu(
                    scale=args.scale,
                    iteration=target,
                    run_root=args.run_root,
                    output_root=confirmation_root,
                    code_root=args.code_root,
                    code_receipt=args.code_receipt,
                    clean_examples=args.clean_examples,
                    sentinel_manifest=args.sentinel_manifest,
                    eval_venv=args.eval_venv,
                    mode="full_clean",
                )
                confirmation = {
                    "target_iteration": target,
                    "reused_existing_full_score": False,
                    **artifacts,
                }
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_greekmmlu_plateau_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": args.scale,
                "calibration_result": file_binding(calibration_result_path),
                "calibration": file_binding(calibration_path),
                "joint_calibration_authority": file_binding(
                    args.joint_calibration_authority
                ),
                "cross_scale_trajectory": joint_trajectory,
                "trajectory_view": "full_clean" if full_panel_required else f"sentinel_{selected_size}",
                "bootstrap": {
                    "replicates": 10_000,
                    "seed_base": 20260814,
                    "interval": "percentile_95",
                    "paired_by_example_id": True,
                },
                "minimum_choice_nll_iteration": minimum_update,
                "plateau_definition": "paired_95_percent_choice_nll_difference_ci_against_minimum_includes_zero",
                "plateau_members": plateau_members,
                "comparisons": comparisons,
                "source_evaluations": source_bindings,
                "confirmation_action": action,
                "full_panel_confirmation": confirmation,
            },
        )
    except BaseException:
        if confirmation_root.exists():
            shutil.rmtree(confirmation_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

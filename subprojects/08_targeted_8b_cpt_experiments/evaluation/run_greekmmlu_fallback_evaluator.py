#!/usr/bin/env python3
"""Run a mandatory full GreekMMLU fallback when sentinel calibration rejects."""

from __future__ import annotations

import argparse
import shutil
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

FALLBACK_UPDATES = tuple(sorted(ALL_UPDATES - FULL_CLEAN_UPDATES))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--target-iteration", type=int, required=True)
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

    target = args.target_iteration
    require(target in FALLBACK_UPDATES, f"non-predeclared fallback target: {target}")
    require(
        args.iteration == max(3218, target),
        "fallback eligibility milestone drift",
    )
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
        and calibration.get("scale") == args.scale
        and calibration_result.get("decision_state") == calibration.get("decision_state"),
        "canonical calibration binding drift",
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
    source_path, source = completed_result(
        args.run_root,
        evaluator_id="greekmmlu",
        iteration=target,
        schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
        scale=args.scale,
    )
    require(source.get("mode") == "sentinel_pair", "fallback source was not a sentinel score")

    action: str
    artifacts: dict[str, object]
    full_root = args.output / "full_fallback"
    try:
        if joint_trajectory["mode"] == "full_clean":
            action = "full_panel_scored"
            artifacts = execute_frozen_greekmmlu(
                scale=args.scale,
                iteration=target,
                run_root=args.run_root,
                output_root=full_root,
                code_root=args.code_root,
                code_receipt=args.code_receipt,
                clean_examples=args.clean_examples,
                sentinel_manifest=args.sentinel_manifest,
                eval_venv=args.eval_venv,
                mode="full_clean",
            )
        else:
            selected_size = int(joint_trajectory.get("selected_size", 0))
            require(selected_size in {4096, 8192}, "authorized sentinel size drift")
            evaluation_path = require_file_binding(source["evaluation"])
            evaluation = read_json(evaluation_path)
            view_name = f"sentinel_{selected_size}"
            view_path = require_file_binding(evaluation["views"][view_name])
            action = "sentinel_authorized_no_full_score_required"
            artifacts = {
                "checkpoint_export": source["checkpoint_export"],
                "evaluation": file_binding(evaluation_path),
                "summary": source["summary"],
                "views": {view_name: file_binding(view_path)},
            }
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_greekmmlu_fallback_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "target_iteration": target,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": args.scale,
                "action": action,
                "source_sentinel_result": file_binding(source_path),
                "calibration_result": file_binding(calibration_result_path),
                "calibration": file_binding(calibration_path),
                "joint_calibration_authority": file_binding(
                    args.joint_calibration_authority
                ),
                "decision_state": calibration["decision_state"],
                "cross_scale_trajectory": joint_trajectory,
                "selected_size": joint_trajectory.get("selected_size"),
                **artifacts,
            },
        )
    except BaseException:
        if full_root.exists():
            shutil.rmtree(full_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

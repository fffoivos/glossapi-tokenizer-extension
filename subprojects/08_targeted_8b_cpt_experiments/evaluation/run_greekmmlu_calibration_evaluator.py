#!/usr/bin/env python3
"""Canonical evaluator that freezes same-stack GreekMMLU sentinel calibration."""

from __future__ import annotations

import argparse
import os
import subprocess
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
from validate_greekmmlu_sentinels import CALIBRATION_UPDATES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--sentinel-manifest", type=Path, required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.iteration == 3218, "sentinel calibration must run at update 3218")
    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")

    prediction_args: list[str] = []
    sources = {}
    for update in CALIBRATION_UPDATES:
        source_path, source = completed_result(
            args.run_root,
            evaluator_id="greekmmlu",
            iteration=update,
            schema="apertus_hard_h_to_g_greekmmlu_evaluation_v1",
            scale=args.scale,
        )
        require(source.get("mode") == "full_clean", f"{update}: full-clean score missing")
        evaluation_path = require_file_binding(source["evaluation"])
        evaluation = read_json(evaluation_path)
        require(
            evaluation.get("schema_version") == "apertus_frozen_greekmmlu_evaluation_v1"
            and evaluation.get("status") == "completed"
            and evaluation.get("scale") == args.scale
            and int(evaluation.get("iteration", -1)) == update
            and evaluation.get("mode") == "full_clean",
            f"{update}: frozen GreekMMLU receipt drift",
        )
        prediction_path = require_file_binding(evaluation["views"]["full_clean"])
        prediction_args.extend(["--prediction", f"{update}={prediction_path}"])
        sources[str(update)] = {
            "canonical_result": file_binding(source_path),
            "evaluation": file_binding(evaluation_path),
            "predictions": file_binding(prediction_path),
        }

    calibration_path = args.output / "calibration.json"
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run(
        [
            str((args.eval_venv / "bin/python").resolve()),
            str((args.code_root / "subprojects/08_targeted_8b_cpt_experiments/evaluation/validate_greekmmlu_sentinels.py").resolve()),
            "--scale",
            args.scale,
            "--manifest",
            str(args.sentinel_manifest.resolve()),
            *prediction_args,
            "--bootstrap-replicates",
            "10000",
            "--bootstrap-seed",
            "20260814",
            "--expected-full-count",
            "16159",
            "--output",
            str(calibration_path),
        ],
        env=env,
        check=True,
    )
    calibration = read_json(calibration_path)
    require(
        calibration.get("schema_version") == "apertus_greekmmlu_sentinel_calibration_v1"
        and calibration.get("status") == "passed"
        and calibration.get("scale") == args.scale
        and calibration.get("decision_state")
        in {"4096_pass", "8192_pass", "full_panel_required"}
        and calibration.get("selection_authorized")
        is (not calibration.get("full_panel_required")),
        "sentinel calibration completion drift",
    )
    write_json_atomic(
        result_path,
        {
            "schema_version": "apertus_hard_h_to_g_greekmmlu_calibration_evaluation_v1",
            "status": "completed",
            "campaign_id": args.campaign_id,
            "evaluator_id": args.evaluator_id,
            "iteration": args.iteration,
            "attempt": args.attempt,
            "contract_digest": args.contract_digest,
            "scale": args.scale,
            "sentinel_manifest": file_binding(args.sentinel_manifest),
            "source_evaluations": sources,
            "calibration": file_binding(calibration_path),
            "decision_state": calibration["decision_state"],
            "selected_size": calibration.get("selected_size"),
            "full_panel_required": calibration["full_panel_required"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Canonical evaluator for the frozen full-clean or nested GreekMMLU panel."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import (
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from run_offline_panels_evaluator import resolve_export

FULL_CLEAN_UPDATES = {0, 238, 476, 714, 2618, 2856, 3094, 3218, 3694}
ALL_UPDATES = {
    0,
    238,
    476,
    714,
    952,
    1190,
    1428,
    1666,
    1904,
    2142,
    2261,
    2380,
    2618,
    2856,
    3094,
    3218,
    3456,
    3694,
}


def mode_for_iteration(iteration: int) -> str:
    require(iteration in ALL_UPDATES, f"GreekMMLU iteration is not predeclared: {iteration}")
    return "full_clean" if iteration in FULL_CLEAN_UPDATES else "sentinel_pair"


def execute_frozen_greekmmlu(
    *,
    scale: str,
    iteration: int,
    run_root: Path,
    output_root: Path,
    code_root: Path,
    code_receipt: Path,
    clean_examples: Path,
    sentinel_manifest: Path,
    eval_venv: Path,
    mode: str,
) -> dict[str, Any]:
    """Run one exact frozen panel and return its verified artifact bindings."""

    require(mode in {"full_clean", "sentinel_pair"}, "invalid GreekMMLU mode")
    _hf_root, export_receipt = resolve_export(
        run_root, scale=scale, iteration=iteration
    )
    wrapper = (
        code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/run_frozen_greekmmlu_1node_debug.sbatch"
    )
    require(wrapper.is_file(), "frozen GreekMMLU wrapper missing")
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(code_root.resolve()),
            "H2G_CODE_RECEIPT": str(code_receipt.resolve()),
            "H2G_SCALE": scale,
            "H2G_ITERATION": str(iteration),
            "H2G_CHECKPOINT_EXPORT": str(export_receipt),
            "H2G_GREEKMMLU_MODE": mode,
            "H2G_GREEKMMLU_CLEAN_EXAMPLES": str(clean_examples.resolve()),
            "H2G_GREEKMMLU_SENTINEL_MANIFEST": str(sentinel_manifest.resolve()),
            "H2G_GREEKMMLU_OUTPUT": str(output_root),
            "EVAL_VENV": str(eval_venv.resolve()),
        }
    )
    subprocess.run(["bash", str(wrapper)], env=env, check=True)
    receipt_path = output_root / "aggregate/receipt.json"
    summary_path = output_root / "aggregate/summary.json"
    receipt = read_json(receipt_path)
    summary = read_json(summary_path)
    expected_rows = 16_159 if mode == "full_clean" else 8192
    expected_views = (
        {"full_clean", "sentinel_4096", "sentinel_8192"}
        if mode == "full_clean"
        else {"sentinel_4096", "sentinel_8192"}
    )
    require(
        receipt.get("schema_version") == "apertus_frozen_greekmmlu_evaluation_v1"
        and receipt.get("status") == "completed"
        and receipt.get("scale") == scale
        and int(receipt.get("iteration", -1)) == iteration
        and receipt.get("mode") == mode
        and int(summary.get("scored_rows", -1)) == expected_rows
        and set(receipt.get("views", {})) == expected_views
        and receipt.get("summary") == file_binding(summary_path),
        "frozen GreekMMLU completion drift",
    )
    return {
        "checkpoint_export": file_binding(export_receipt),
        "evaluation": file_binding(receipt_path),
        "summary": file_binding(summary_path),
        "views": receipt["views"],
    }


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
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")
    mode = mode_for_iteration(args.iteration)
    evaluation_root = args.output / "greekmmlu"
    try:
        artifacts = execute_frozen_greekmmlu(
            scale=args.scale,
            iteration=args.iteration,
            run_root=args.run_root,
            output_root=evaluation_root,
            code_root=args.code_root,
            code_receipt=args.code_receipt,
            clean_examples=args.clean_examples,
            sentinel_manifest=args.sentinel_manifest,
            eval_venv=args.eval_venv,
            mode=mode,
        )
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_greekmmlu_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": args.scale,
                "mode": mode,
                **artifacts,
            },
        )
    except BaseException:
        if evaluation_root.exists():
            shutil.rmtree(evaluation_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

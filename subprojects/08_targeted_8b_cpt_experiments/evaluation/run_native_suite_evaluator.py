#!/usr/bin/env python3
"""Canonical-campaign adapter for one contamination-filtered native-Greek suite."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import file_binding, read_json, require, write_json_atomic
from run_offline_panels_evaluator import resolve_export


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--eval-code-root", type=Path, required=True)
    parser.add_argument("--eval-code-receipt", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-execution-gate", type=Path, required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")
    _hf_root, export_receipt = resolve_export(
        args.run_root, scale=args.scale, iteration=args.iteration
    )
    native_root = args.output / "native_suite"
    wrapper = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/run_native_suite_checkpoint_4node_debug.sbatch"
    )
    require(wrapper.is_file(), "frozen native-suite wrapper missing")
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_SCALE": args.scale,
            "H2G_ITERATION": str(args.iteration),
            "H2G_CHECKPOINT_EXPORT": str(export_receipt),
            "H2G_NATIVE_OUTPUT": str(native_root),
            "EVAL_CODE_ROOT": str(args.eval_code_root.resolve()),
            "EVAL_CODE_RECEIPT": str(args.eval_code_receipt.resolve()),
            "EVAL_SOURCE_CONTRACT": str(args.source_contract.resolve()),
            "EVAL_SOURCE_MANIFEST": str(args.source_manifest.resolve()),
            "EVAL_SOURCE_GATE": str(args.source_execution_gate.resolve()),
            "EVAL_VENV": str(args.eval_venv.resolve()),
            "EVAL_EXCLUSIONS": str(args.exclusions.resolve()),
        }
    )
    try:
        subprocess.run(["bash", str(wrapper)], env=env, check=True)
        matrix_path = native_root / "matrix_receipt.json"
        filtered_path = native_root / "contamination_filtered/receipt.json"
        rebind_path = native_root / "assets/rebind_receipt.json"
        matrix = read_json(matrix_path)
        filtered = read_json(filtered_path)
        rebind = read_json(rebind_path)
        require(
            matrix.get("status") == "completed"
            and len(matrix.get("checkpoint_receipts", [])) == 1,
            "native-suite matrix drift",
        )
        require(
            filtered.get("status") == "passed"
            and len(filtered.get("checkpoints", [])) == 1,
            "native-suite filtered result drift",
        )
        require(
            rebind.get("status") == "passed" and all(rebind.get("checks", {}).values()),
            "native-suite rebind drift",
        )
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_native_suite_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": args.scale,
                "checkpoint_export": file_binding(export_receipt),
                "rebind": file_binding(rebind_path),
                "matrix": file_binding(matrix_path),
                "contamination_filtered": file_binding(filtered_path),
                "exclusions": file_binding(args.exclusions),
            },
        )
    except BaseException:
        if native_root.exists():
            shutil.rmtree(native_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

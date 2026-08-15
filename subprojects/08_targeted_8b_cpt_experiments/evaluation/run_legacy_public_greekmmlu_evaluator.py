#!/usr/bin/env python3
"""Canonical adapter for the frozen 8B update-3218 historical compatibility score."""

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
    parser.add_argument("--scale", choices=("8b",), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--legacy-contract", type=Path, required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.iteration == 3218, "legacy public evaluator is fixed at update 3218")
    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")
    _hf_root, export_receipt = resolve_export(
        args.run_root, scale="8b", iteration=args.iteration
    )
    legacy_root = args.output / "legacy_public"
    wrapper = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/run_legacy_public_greekmmlu_debug.sbatch"
    )
    require(wrapper.is_file(), "legacy public wrapper missing")
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_SCALE": "8b",
            "H2G_ITERATION": "3218",
            "H2G_CHECKPOINT_EXPORT": str(export_receipt),
            "H2G_LEGACY_CONTRACT": str(args.legacy_contract.resolve()),
            "H2G_LEGACY_OUTPUT": str(legacy_root),
            "EVAL_VENV": str(args.eval_venv.resolve()),
        }
    )
    try:
        subprocess.run(["bash", str(wrapper)], env=env, check=True)
        compatibility_path = legacy_root / "result_receipt.json"
        compatibility = read_json(compatibility_path)
        require(
            compatibility.get("schema_version")
            == "apertus_legacy_public_greekmmlu_result_v1"
            and compatibility.get("status") == "completed"
            and compatibility.get("scope")
            == "8b_update_3218_historical_compatibility_only"
            and compatibility.get("scientific_primary") is False
            and compatibility.get("decision") in {"pass", "fail", "inconclusive"},
            "legacy public compatibility result drift",
        )
        write_json_atomic(
            result_path,
            {
                "schema_version": "apertus_hard_h_to_g_legacy_public_evaluation_v1",
                "status": "completed",
                "campaign_id": args.campaign_id,
                "evaluator_id": args.evaluator_id,
                "iteration": args.iteration,
                "attempt": args.attempt,
                "contract_digest": args.contract_digest,
                "scale": "8b",
                "scientific_primary": False,
                "checkpoint_export": file_binding(export_receipt),
                "legacy_contract": file_binding(args.legacy_contract),
                "compatibility_result": file_binding(compatibility_path),
                "decision": compatibility["decision"],
            },
        )
    except BaseException:
        if legacy_root.exists():
            shutil.rmtree(legacy_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

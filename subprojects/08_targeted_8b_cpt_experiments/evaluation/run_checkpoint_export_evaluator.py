#!/usr/bin/env python3
"""Canonical-campaign evaluator adapter for exact checkpoint HF export."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import file_binding, read_json, require, write_json_atomic
from run_canonical_train_segment import load_checkpoint_reference


def resolve_checkpoint_root(run_root: Path, *, scale: str, iteration: int) -> Path:
    matches = []
    for path in run_root.resolve().rglob("checkpoint_reference.json"):
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if value.get("scale") == scale and int(value.get("update", -1)) == iteration:
            matches.append(path)
    require(
        len(matches) == 1,
        f"expected exactly one checkpoint reference for {scale}@{iteration}; found {len(matches)}",
    )
    reference = load_checkpoint_reference(matches[0], scale=scale, update=iteration)
    return Path(str(reference["load_root"])).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--contract-digest", required=True)
    args = parser.parse_args()

    require(args.output.is_dir(), "canonical evaluation attempt root missing")
    result_path = args.output / "result.json"
    require(not result_path.exists(), "canonical evaluation result already exists")
    source_root = (
        args.initial_checkpoint_root.resolve()
        if args.iteration == 0
        else resolve_checkpoint_root(
            args.run_root, scale=args.scale, iteration=args.iteration
        )
    )
    checkpoint_name = "release" if args.iteration == 0 else f"iter_{args.iteration:07d}"
    require(
        (source_root / checkpoint_name).is_dir(), "evaluation source checkpoint missing"
    )
    export_root = args.output / "export"
    wrapper = (
        args.code_root
        / "subprojects/08_targeted_8b_cpt_experiments/clariden/export_checkpoint_for_evaluation_debug.sbatch"
    )
    require(wrapper.is_file(), "frozen checkpoint export wrapper missing")
    env = os.environ.copy()
    env.update(
        {
            "H2G_CODE_ROOT": str(args.code_root.resolve()),
            "H2G_CODE_RECEIPT": str(args.code_receipt.resolve()),
            "H2G_SCALE": args.scale,
            "H2G_SOURCE_CHECKPOINT_ROOT": str(source_root),
            "H2G_SOURCE_ITERATION": str(args.iteration),
            "H2G_MEGATRON_DIR": str(args.megatron_root.resolve()),
            "H2G_TOKENIZER_DIR": str(args.tokenizer_dir.resolve()),
            "H2G_MODEL_CONTRACT": str(args.model_contract.resolve()),
            "H2G_EXPORT_ROOT": str(export_root),
            "H2G_EVAL_PYTHON": str(args.eval_python.resolve()),
        }
    )
    subprocess.run(["bash", str(wrapper)], env=env, check=True)
    receipt_path = export_root / "checkpoint_export_receipt.json"
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema_version") == "apertus_hard_h_to_g_checkpoint_export_v1"
        and receipt.get("status") == "completed"
        and receipt.get("scale") == args.scale
        and int(receipt.get("iteration", -1)) == args.iteration
        and receipt.get("ready_for_frozen_evaluators") is True,
        "checkpoint export completion receipt drift",
    )
    write_json_atomic(
        result_path,
        {
            "schema_version": "apertus_hard_h_to_g_checkpoint_export_evaluation_v1",
            "status": "completed",
            "campaign_id": args.campaign_id,
            "evaluator_id": args.evaluator_id,
            "iteration": args.iteration,
            "attempt": args.attempt,
            "contract_digest": args.contract_digest,
            "scale": args.scale,
            "checkpoint_export": file_binding(receipt_path),
            "hf_export_root": str((export_root / "hf").resolve()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

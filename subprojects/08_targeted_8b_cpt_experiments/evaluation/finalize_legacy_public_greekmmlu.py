#!/usr/bin/env python3
"""Finalize the single historical BF16 public-GreekMMLU compatibility point."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import file_binding, read_json, require, write_json_atomic

QUESTIONS = 16_632
REFERENCE_CORRECT = 9_969
REFERENCE_ACCURACY = REFERENCE_CORRECT / QUESTIONS
MARGIN = 0.015
Z_90 = statistics.NormalDist().inv_cdf(0.95)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(all(isinstance(row, dict) for row in rows), "legacy predictions malformed")
    return rows


def wilson_interval(correct: int, total: int, z: float = Z_90) -> tuple[float, float]:
    require(total > 0 and 0 <= correct <= total and z > 0, "invalid Wilson input")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def equivalence_decision(interval: tuple[float, float]) -> str:
    lower, upper = interval
    band = REFERENCE_ACCURACY - MARGIN, REFERENCE_ACCURACY + MARGIN
    if lower >= band[0] and upper <= band[1]:
        return "pass"
    if upper < band[0] or lower > band[1]:
        return "fail"
    return "inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-export", type=Path, required=True)
    parser.add_argument("--legacy-contract-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output.exists(), f"immutable legacy result exists: {args.output}")
    predictions_path = (
        args.evaluation_root / f"{args.model_label}_native_mcq_predictions.jsonl"
    )
    rows = read_jsonl(predictions_path)
    ids = [str(row.get("example_id", "")) for row in rows]
    require(
        len(rows) == len(set(ids)) == QUESTIONS,
        "legacy public GreekMMLU prediction count/id drift",
    )
    require(
        all(row.get("benchmark") == "greekmmlu" for row in rows),
        "legacy result contains another benchmark",
    )
    contract = read_json(args.legacy_contract_receipt)
    require(
        contract.get("schema_version") == "apertus_legacy_public_greekmmlu_receipt_v1"
        and contract.get("status") == "frozen"
        and contract.get("code_revision")
        == "cfdd0e7b00761a736be660867bf3d09733e24a92"
        and contract.get("loader_change_scope") == "dataset_loading_only"
        and contract.get("clean_panel_is_scientific_primary") is True,
        "legacy public evaluator contract drift",
    )
    config = read_json(args.model / "config.json")
    require(int(config.get("vocab_size", -1)) == 148_480, "legacy tokenizer/vocab drift")
    correct = sum(int(bool(row["correct"])) for row in rows)
    interval = wilson_interval(correct, QUESTIONS)
    decision = equivalence_decision(interval)
    write_json_atomic(
        args.output,
        {
            "schema_version": "apertus_legacy_public_greekmmlu_result_v1",
            "status": "completed",
            "scope": "8b_update_3218_historical_compatibility_only",
            "scientific_primary": False,
            "named_reconstruction_difference": (
                "historical_June_dataset_revision_unrecoverable; scorer uses the "
                "pinned 2026-07-31-equivalent snapshot through the proven loader-only adapter"
            ),
            "model": str(args.model.resolve()),
            "checkpoint_export": file_binding(args.checkpoint_export),
            "legacy_contract": file_binding(args.legacy_contract_receipt),
            "predictions": file_binding(predictions_path),
            "scoring": {
                "dtype": "bfloat16",
                "candidate_batch_size": 16,
                "example_batch_size": 16,
                "questions": QUESTIONS,
            },
            "observed": {
                "correct": correct,
                "accuracy": correct / QUESTIONS,
                "wilson_90_percent": list(interval),
            },
            "reference": {
                "correct": REFERENCE_CORRECT,
                "questions": QUESTIONS,
                "accuracy": REFERENCE_ACCURACY,
                "absolute_equivalence_margin": MARGIN,
                "equivalence_band": [
                    REFERENCE_ACCURACY - MARGIN,
                    REFERENCE_ACCURACY + MARGIN,
                ],
            },
            "decision": decision,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

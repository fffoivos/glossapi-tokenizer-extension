#!/usr/bin/env python3
"""Aggregate matched full-clean GreekMMLU checkpoint trajectories."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


TOKENS_PER_UPDATE = 4_194_304


def binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2 + 1
        for position in order[start:end]:
            result[position] = average
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    if len(set(a)) == 1 or len(set(b)) == 1:
        return None
    return statistics.correlation(a, b)


def load_predictions(path: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in result:
                raise ValueError(f"duplicate example id in {path}: {example_id}")
            scores = [float(choice["avg_logprob"]) for choice in row["choice_scores"]]
            predicted = max(range(len(scores)), key=scores.__getitem__)
            result[example_id] = predicted == int(row["answer_index"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows: list[dict[str, Any]] = []
    prediction_maps: dict[tuple[str, int], dict[str, bool]] = {}
    with args.sources.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle, delimiter="\t"):
            scale = source["scale"]
            update = int(source["update"])
            iteration_dir = f"iter_{update:07d}"
            if source["existing_result_root"] != "-":
                result_root = Path(source["existing_result_root"])
            else:
                result_root = args.trajectory_root / "results" / scale / iteration_dir / "full_clean"
            receipt_path = result_root / "aggregate" / "receipt.json"
            summary_path = result_root / "aggregate" / "summary.json"
            predictions_path = result_root / "aggregate" / "predictions.jsonl"
            receipt = read_json(receipt_path)
            summary = read_json(summary_path)
            if not (
                receipt.get("schema_version") == "apertus_frozen_greekmmlu_evaluation_v1"
                and receipt.get("status") == "completed"
                and receipt.get("scale") == scale
                and int(receipt.get("iteration", -1)) == update
                and receipt.get("mode") == "full_clean"
            ):
                raise ValueError(f"aggregate receipt identity drift: {receipt_path}")
            view = summary["views"]["full_clean"]
            metrics = view["metrics"]
            overall = metrics["overall"]
            if int(overall["n"]) != 16_159:
                raise ValueError(f"full-clean question count drift: {summary_path}")
            prediction_maps[(scale, update)] = load_predictions(predictions_path)
            rows.append(
                {
                    "scale": scale,
                    "update": update,
                    "token_slots": update * TOKENS_PER_UPDATE,
                    "accuracy": float(overall["accuracy"]),
                    "correct": int(overall["correct"]),
                    "n": int(overall["n"]),
                    "choice_nll": float(overall["choice_nll"]),
                    "correct_answer_bpb": float(overall["correct_answer_bpb"]),
                    "by_subject": metrics["by_subject"],
                    "by_educational_level": metrics["by_educational_level"],
                    "receipt": binding(receipt_path),
                    "summary": binding(summary_path),
                    "predictions": binding(predictions_path),
                }
            )

    by_scale = {
        scale: sorted((row for row in rows if row["scale"] == scale), key=lambda row: row["update"])
        for scale in ("1p5b", "8b")
    }
    expected_updates = [row["update"] for row in by_scale["1p5b"]]
    if expected_updates != [row["update"] for row in by_scale["8b"]] or len(expected_updates) != 17:
        raise ValueError("cross-scale checkpoint grid drift")

    comparisons: list[dict[str, Any]] = []
    for update in expected_updates:
        one = prediction_maps[("1p5b", update)]
        eight = prediction_maps[("8b", update)]
        if set(one) != set(eight) or len(one) != 16_159:
            raise ValueError(f"cross-scale question identity drift at update {update}")
        both_correct = sum(one[key] and eight[key] for key in one)
        one_only = sum(one[key] and not eight[key] for key in one)
        eight_only = sum(eight[key] and not one[key] for key in one)
        both_wrong = len(one) - both_correct - one_only - eight_only
        comparisons.append(
            {
                "update": update,
                "token_slots": update * TOKENS_PER_UPDATE,
                "both_correct": both_correct,
                "one_p5b_only_correct": one_only,
                "eight_b_only_correct": eight_only,
                "both_wrong": both_wrong,
                "answer_correctness_agreement": (both_correct + both_wrong) / len(one),
            }
        )

    metric_shape: dict[str, Any] = {}
    for metric in ("accuracy", "choice_nll", "correct_answer_bpb"):
        one = [float(row[metric]) for row in by_scale["1p5b"]]
        eight = [float(row[metric]) for row in by_scale["8b"]]
        one_delta = [right - left for left, right in zip(one, one[1:])]
        eight_delta = [right - left for left, right in zip(eight, eight[1:])]
        metric_shape[metric] = {
            "level_pearson": statistics.correlation(one, eight),
            "adjacent_first_difference_pearson": statistics.correlation(one_delta, eight_delta),
            "adjacent_first_difference_spearman": spearman(one_delta, eight_delta),
        }

    peaks = {}
    for scale, values in by_scale.items():
        peak = max(values, key=lambda row: row["accuracy"])
        minimum_nll = min(values, key=lambda row: row["choice_nll"])
        peaks[scale] = {
            "accuracy": {"update": peak["update"], "value": peak["accuracy"]},
            "choice_nll": {"update": minimum_nll["update"], "value": minimum_nll["choice_nll"]},
        }

    payload = {
        "schema_version": "apertus_h2g_cross_scale_greekmmlu_trajectory_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "panel": {"name": "decontaminated_full_clean", "n": 16_159},
        "tokens_per_update": TOKENS_PER_UPDATE,
        "updates": expected_updates,
        "rows": rows,
        "cross_scale_question_comparisons": comparisons,
        "cross_scale_shape": metric_shape,
        "peaks": peaks,
        "sources": binding(args.sources),
    }
    if not all(math.isfinite(float(row[key])) for row in rows for key in ("accuracy", "choice_nll", "correct_answer_bpb")):
        raise ValueError("non-finite trajectory metric")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Promote BF16 evaluation only when it tracks the FP32 scorer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["benchmark"], row["example_id"])
            if key in rows:
                raise ValueError(f"duplicate prediction {key}")
            rows[key] = row
    return rows


def choice_nll(row: dict) -> float:
    import math

    scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(value - maximum) for value in scores)) - scores[int(row["answer_index"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32", type=Path, required=True)
    parser.add_argument("--bfloat16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-prediction-agreement", type=float, default=0.995)
    parser.add_argument("--max-per-benchmark-accuracy-delta", type=float, default=0.005)
    parser.add_argument("--max-mean-choice-nll-delta", type=float, default=0.005)
    args = parser.parse_args()
    left, right = load(args.float32), load(args.bfloat16)
    if left.keys() != right.keys():
        raise ValueError("FP32/BF16 example identities differ")

    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for key in sorted(left):
        grouped[key[0]].append((left[key], right[key]))
    rows = []
    for benchmark, pairs in sorted(grouped.items()):
        agreement = sum(a["pred_index"] == b["pred_index"] for a, b in pairs) / len(pairs)
        accuracy_delta = abs(
            sum(bool(a["correct"]) for a, _ in pairs) / len(pairs)
            - sum(bool(b["correct"]) for _, b in pairs) / len(pairs)
        )
        mean_nll_delta = abs(
            sum(choice_nll(a) for a, _ in pairs) / len(pairs)
            - sum(choice_nll(b) for _, b in pairs) / len(pairs)
        )
        rows.append(
            {
                "benchmark": benchmark,
                "n": len(pairs),
                "prediction_agreement": agreement,
                "absolute_accuracy_delta": accuracy_delta,
                "absolute_mean_choice_nll_delta": mean_nll_delta,
                "passed": agreement >= args.min_prediction_agreement
                and accuracy_delta <= args.max_per_benchmark_accuracy_delta
                and mean_nll_delta <= args.max_mean_choice_nll_delta,
            }
        )
    receipt = {
        "schema_version": "apertus_full8_native_greek_dtype_parity_v1",
        "status": "passed" if all(row["passed"] for row in rows) else "rejected",
        "candidate": "bfloat16",
        "reference": "float32",
        "thresholds": {
            "min_prediction_agreement": args.min_prediction_agreement,
            "max_per_benchmark_accuracy_delta": args.max_per_benchmark_accuracy_delta,
            "max_mean_choice_nll_delta": args.max_mean_choice_nll_delta,
        },
        "benchmarks": rows,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

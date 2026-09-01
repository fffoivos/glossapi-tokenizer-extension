#!/usr/bin/env python3
"""Promote a larger FP32 candidate batch only when it matches FP32 batch one."""

from __future__ import annotations

import argparse
import json
import math
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
    scores = [float(value["avg_logprob"]) for value in row["choice_scores"]]
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(value - maximum) for value in scores)) - scores[int(row["answer_index"])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch1", type=Path, required=True)
    parser.add_argument("--batch4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = load(args.batch1), load(args.batch4)
    if reference.keys() != candidate.keys():
        raise ValueError("FP32 batch-one/batch-four example identities differ")

    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for key in sorted(reference):
        grouped[key[0]].append((reference[key], candidate[key]))
    rows = []
    for benchmark, pairs in sorted(grouped.items()):
        agreement = sum(a["pred_index"] == b["pred_index"] for a, b in pairs) / len(pairs)
        max_score_delta = max(
            abs(float(sa["sum_logprob"]) - float(sb["sum_logprob"]))
            for a, b in pairs
            for sa, sb in zip(a["choice_scores"], b["choice_scores"], strict=True)
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
                "max_absolute_sum_logprob_delta": max_score_delta,
                "absolute_mean_choice_nll_delta": mean_nll_delta,
                "passed": agreement == 1.0 and max_score_delta <= 1e-4 and mean_nll_delta <= 1e-6,
            }
        )
    receipt = {
        "schema_version": "apertus_full8_native_greek_fp32_batch_parity_v1",
        "status": "passed" if all(row["passed"] for row in rows) else "rejected",
        "reference": "legacy_full_logits_float32_candidate_batch1",
        "candidate": "legacy_full_logits_float32_candidate_batch4",
        "thresholds": {
            "prediction_agreement": 1.0,
            "max_absolute_sum_logprob_delta": 1e-4,
            "max_absolute_mean_choice_nll_delta": 1e-6,
        },
        "benchmarks": rows,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Forecast one-arm and five-arm wall time from measured concurrent throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "experiment_matrix.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-second", type=float, required=True)
    parser.add_argument("--data-parallel", type=int, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()

    if args.tokens_per_second <= 0:
        parser.error("--tokens-per-second must be positive")

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    parallel = matrix["training_control"]["parallelism"]
    if args.data_parallel not in parallel["candidate_data_parallel"]:
        parser.error("--data-parallel is not in the frozen benchmark ladder")

    tokens = int(matrix["planning_estimate"]["total_tokens_rounded"])
    hours = tokens / args.tokens_per_second / 3600
    report = {
        "data_parallel_per_arm": args.data_parallel,
        "nodes_per_arm": args.data_parallel // 4,
        "nodes_for_five_concurrent_arms": 5 * args.data_parallel // 4,
        "measured_tokens_per_second_per_arm": args.tokens_per_second,
        "forecast_training_hours_per_arm_and_campaign_critical_path": hours,
        "forecast_five_arm_aggregate_tokens_per_second": 5
        * args.tokens_per_second,
        "fits_one_12_hour_segment": hours <= 12,
        "fits_preferred_24_hour_training_budget": hours <= 24,
        "fits_36_hour_complete_round_ceiling_before_nontraining_overhead": hours
        <= 36,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


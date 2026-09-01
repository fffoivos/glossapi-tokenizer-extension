#!/usr/bin/env python3
"""Summarize learning, forgetting, and GreekMMLU endpoints without selecting a winner."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import ARMS, TOTAL_ITERATIONS, atomic_write_json, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-trajectory", type=Path, required=True)
    parser.add_argument("--greekmmlu-trajectory", type=Path, required=True)
    parser.add_argument("--full-endpoint-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = read_json(args.validation_trajectory)
    greekmmlu = read_json(args.greekmmlu_trajectory)
    full = read_json(args.full_endpoint_validation)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in validation["rows"]:
        grouped[(row["arm_id"], row["panel"])].append(row)
    loss_summary = []
    for (arm, panel), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["iteration"]))
        initial = next(row for row in rows if int(row["iteration"]) == 0)
        final = next(row for row in rows if int(row["iteration"]) == TOTAL_ITERATIONS)
        best = min(rows, key=lambda row: float(row["bpb"]))
        loss_summary.append(
            {
                "arm_id": arm,
                "panel": panel,
                "initial_bpb": initial["bpb"],
                "best_bpb": best["bpb"],
                "best_iteration": best["iteration"],
                "fast_final_bpb": final["bpb"],
                "forgetting_bpb": float(final["bpb"]) - float(best["bpb"]),
                "relative_improvement_from_initial": (
                    float(initial["bpb"]) - float(final["bpb"])
                ) / float(initial["bpb"]),
            }
        )
    full_rows = {(row["arm_id"], row["panel"]): row for row in full["rows"]}
    if len(full_rows) != 65:
        raise ValueError("full endpoint panel is incomplete")
    for row in loss_summary:
        row["full_final_bpb"] = full_rows[(row["arm_id"], row["panel"])]["bpb"]
    mmlu_endpoints = [row for row in greekmmlu["rows"] if int(row["iteration"]) == TOTAL_ITERATIONS]
    if tuple(sorted(row["arm_id"] for row in mmlu_endpoints)) != tuple(sorted(ARMS)):
        raise ValueError("GreekMMLU endpoint set is incomplete")
    payload = {
        "schema_version": "apertus_mini_core_campaign_summary_v1",
        "status": "completed",
        "winner_selected": False,
        "winner_selection_blocked_until_retention_constraints_and_uncertainty_are_applied": True,
        "loss_learning_and_forgetting": loss_summary,
        "greekmmlu_endpoints": mmlu_endpoints,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "loss_rows": len(loss_summary), "mmlu_endpoints": 5}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

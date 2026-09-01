#!/usr/bin/env python3
"""Collect all 415 exact-checkpoint native-GreekMMLU receipts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import (
    ARMS,
    AUTHORITATIVE_EVALUATION_DTYPE,
    atomic_write_json,
    evaluation_namespace,
    read_json,
    scoped_evaluation_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    campaign = read_json(args.campaign_manifest)
    namespace = evaluation_namespace()
    expected_iterations = tuple(int(value) for value in campaign["evaluation"]["checkpoint_iterations"])
    rows = []
    seen = set()
    for segment_id in (0, 1):
        state_root = (
            scoped_evaluation_root(args.run_root, "evaluation_watch")
            / f"segment_{segment_id}"
        )
        for state_path in sorted(state_root.glob("iteration_*.json")):
            state = read_json(state_path)
            if state.get("status") != "completed":
                continue
            iteration = int(state["iteration"])
            for arm in ARMS:
                key = (arm, iteration)
                if key in seen:
                    raise ValueError(f"duplicate GreekMMLU binding: {key}")
                receipt_path = Path(state["receipts"][arm])
                receipt = read_json(receipt_path)
                metrics = receipt.get("metrics", {})
                clean = metrics.get("decontaminated", {})
                if (
                    receipt.get("schema_version") != "exact_checkpoint_native_greekmmlu_receipt_v1"
                    or receipt.get("status") != "completed"
                    or receipt.get("evaluation_namespace") != namespace
                    or receipt.get("evaluator", {}).get("dtype")
                    != AUTHORITATIVE_EVALUATION_DTYPE
                    or int(receipt.get("checkpoint", {}).get("iteration", -1)) != iteration
                    or int(metrics.get("n", -1)) != 16_632
                    or int(clean.get("n", 0)) <= 0
                ):
                    raise ValueError(f"GreekMMLU receipt drift: {receipt_path}")
                rows.append(
                    {
                        "arm_id": arm,
                        "iteration": iteration,
                        "accuracy": metrics["accuracy"],
                        "choice_nll": metrics["choice_nll"],
                        "correct_answer_bpb": metrics["correct_answer_bpb"],
                        "clean_n": clean["n"],
                        "clean_accuracy": clean["accuracy"],
                        "clean_choice_nll": clean["choice_nll"],
                        "clean_correct_answer_bpb": clean["correct_answer_bpb"],
                        "receipt": str(receipt_path.resolve()),
                        "evaluation_namespace": namespace,
                        "evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
                    }
                )
                seen.add(key)
    expected = {(arm, iteration) for arm in ARMS for iteration in expected_iterations}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"GreekMMLU trajectory coverage drift: missing={missing[:5]} extra={extra[:5]}")
    rows.sort(key=lambda row: (row["arm_id"], row["iteration"]))
    payload = {
        "schema_version": "apertus_mini_greekmmlu_trajectory_v1",
        "status": "completed",
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "evaluation_namespace": namespace,
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "rows": rows,
    }
    atomic_write_json(args.output_json, payload)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_csv) + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, args.output_csv)
    print(json.dumps({"ok": True, "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

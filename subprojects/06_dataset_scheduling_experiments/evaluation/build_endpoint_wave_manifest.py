#!/usr/bin/env python3
"""Resolve the five exact final HF exports for endpoint benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import (
    ARMS,
    AUTHORITATIVE_EVALUATION_DTYPE,
    TOTAL_ITERATIONS,
    atomic_write_json,
    evaluation_namespace,
    read_json,
    scoped_evaluation_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    namespace = evaluation_namespace()
    watch_root = scoped_evaluation_root(args.run_root, "evaluation_watch")
    final_state = read_json(
        watch_root / "segment_1" / f"iteration_{TOTAL_ITERATIONS:07d}.json"
    )
    initial_state = read_json(
        watch_root / "segment_0" / "iteration_0000000.json"
    )
    if final_state.get("status") != "completed" or int(final_state.get("iteration", -1)) != TOTAL_ITERATIONS:
        raise ValueError("final GreekMMLU state is incomplete")
    if initial_state.get("status") != "completed" or int(initial_state.get("iteration", -1)) != 0:
        raise ValueError("initial GreekMMLU state is incomplete")
    tasks = []
    bindings = [("initial_shared", 0, initial_state["receipts"][ARMS[0]])]
    bindings.extend((arm, TOTAL_ITERATIONS, final_state["receipts"][arm]) for arm in ARMS)
    for arm, iteration, receipt_text in bindings:
        greek_receipt_path = Path(receipt_text)
        greek = read_json(greek_receipt_path)
        export_path = Path(greek["checkpoint"]["export_receipt_path"])
        export = read_json(export_path)
        model_path = Path(export["hf_export"]["path"])
        if (
            greek.get("status") != "completed"
            or greek.get("evaluation_namespace") != namespace
            or greek.get("evaluator", {}).get("dtype")
            != AUTHORITATIVE_EVALUATION_DTYPE
            or int(greek["checkpoint"]["iteration"]) != iteration
            or export.get("status") != "completed"
            or not model_path.is_dir()
        ):
            raise ValueError(f"final exact-checkpoint export drift: {arm}")
        tasks.append(
            {
                "arm_id": arm,
                "iteration": iteration,
                "model_path": str(model_path.resolve()),
                "greekmmlu_receipt": str(greek_receipt_path.resolve()),
                "export_receipt": str(export_path.resolve()),
                "output_root": str((args.output_root / arm).resolve()),
            }
        )
    payload = {
        "schema_version": "apertus_mini_endpoint_wave_v1",
        "status": "frozen",
        "evaluation_namespace": namespace,
        "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        "tasks": tasks,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

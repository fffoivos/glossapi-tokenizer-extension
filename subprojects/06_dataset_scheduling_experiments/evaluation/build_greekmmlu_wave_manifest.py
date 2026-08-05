#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = {
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
}
EXPECTED_DATASET = (
    "dascim/GreekMMLU",
    "6a03aa06b68beb932fb75edff3a34e50b3674649",
    "All",
    "test",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-evaluation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        nargs=4,
        action="append",
        metavar=("ARM", "ITERATION", "SOURCE_CHECKPOINT_ROOT", "EVAL_OUTPUT_ROOT"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.checkpoint_evaluation_plan.read_text())
    if plan.get("schema_version") != "apertus_mini_checkpoint_evaluation_plan_v1":
        raise SystemExit("invalid checkpoint evaluation plan schema")
    if plan.get("status") != "frozen":
        raise SystemExit("checkpoint evaluation plan is not frozen")
    dataset = plan.get("greekmmlu_dataset", {})
    observed = (
        dataset.get("repo_id"),
        dataset.get("revision"),
        dataset.get("config"),
        dataset.get("split"),
    )
    if observed != EXPECTED_DATASET:
        raise SystemExit(f"GreekMMLU dataset contract drift: {observed}")
    required = {
        int(row["iteration"])
        for row in plan["checkpoint_rows"]
        if row["native_greekmmlu_required"]
    }
    if not 1 <= len(args.task) <= 5:
        raise SystemExit("a GreekMMLU wave must contain one to five tasks")

    tasks = []
    seen = set()
    for arm, iteration_text, source_root, output_root in args.task:
        iteration = int(iteration_text)
        key = (arm, iteration)
        if arm not in ARMS:
            raise SystemExit(f"unknown schedule arm: {arm}")
        if iteration not in required:
            raise SystemExit(f"iteration {iteration} is not required by the checkpoint plan")
        if key in seen:
            raise SystemExit(f"duplicate wave task: {key}")
        seen.add(key)
        tasks.append(
            {
                "arm_id": arm,
                "source_iteration": iteration,
                "source_checkpoint_root": source_root,
                "eval_output_root": output_root,
            }
        )

    payload = {
        "schema_version": "apertus_mini_greekmmlu_wave_v1",
        "status": "frozen",
        "checkpoint_evaluation_plan": str(args.checkpoint_evaluation_plan),
        "checkpoint_evaluation_plan_sha256": sha256(args.checkpoint_evaluation_plan),
        "greekmmlu_dataset": {
            "repo_id": EXPECTED_DATASET[0],
            "revision": EXPECTED_DATASET[1],
            "config": EXPECTED_DATASET[2],
            "split": EXPECTED_DATASET[3],
            "origin": "natively_authored_greek",
        },
        "tasks": tasks,
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite wave manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

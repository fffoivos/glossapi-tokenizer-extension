#!/usr/bin/env python3
"""Run a bounded range of binary or packing tasks on one debug node."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from contract_utils import file_binding, read_json, require, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("binary", "packing"), required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--data-python", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="exclusive positional task index")
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--workers-per-task", type=int, default=8)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def manifest_path(stage: Path, output_prefix: str) -> Path:
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = stage / "megatron" / prefix
    return Path(str(prefix.resolve()) + ".manifest.json")


def task_contract(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, Path]:
    stage = args.stage_root / "experiment_a"
    if args.kind == "binary":
        contract = stage / "binary_contract/input_receipt.json"
        value = read_json(contract)
        tasks = value["tasks"]
        runner = (
            args.code_root
            / "subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/build_binary_shard.py"
        )
    else:
        contract = stage / "packing_plan.json"
        value = read_json(contract)
        tasks = value["tasks"]
        runner = (
            args.code_root
            / "subprojects/06_dataset_scheduling_experiments/dataset/pack_catalog_bucket.py"
        )
    require(0 <= args.start < args.end <= len(tasks), "task range is outside the frozen contract")
    require(args.parallel >= 1 and args.workers_per_task >= 1, "parallelism must be positive")
    selected = tasks[args.start : args.end]
    for position, task in enumerate(selected, start=args.start):
        require(int(task["task_index"]) == position, f"task index/position drift at {position}")
        if args.kind == "binary":
            require(task.get("task_origin") == "targeted_new_modern", f"binary range includes inherited task {position}")
    return selected, contract, runner


def command_for(
    args: argparse.Namespace,
    contract: Path,
    runner: Path,
    position: int,
) -> list[str]:
    stage = args.stage_root / "experiment_a"
    if args.kind == "binary":
        return [
            str(args.data_python),
            str(runner),
            "--input-receipt",
            str(contract),
            "--heldout-manifest",
            str(stage / "binary_contract/heldout_manifest.json"),
            "--stage-root",
            str(stage),
            "--task-index",
            str(position),
            "--workers",
            str(args.workers_per_task),
            "--chunksize",
            "8",
        ]
    return [
        str(args.data_python),
        str(runner),
        "--packing-plan",
        str(contract),
        "--input-receipt",
        str(stage / "binary_contract/input_receipt.json"),
        "--stage-root",
        str(stage),
        "--bridge-common",
        str(
            args.code_root
            / "subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/bridge_common.py"
        ),
        "--task-index",
        str(position),
    ]


def run_one(position: int, command: list[str], log: Path) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return {
        "position": position,
        "returncode": result.returncode,
        "log": file_binding(log),
        "command": command,
    }


def main() -> int:
    args = parse_args()
    require(not args.receipt.exists(), f"immutable batch receipt exists: {args.receipt}")
    selected, contract, runner = task_contract(args)
    stage = args.stage_root / "experiment_a"
    logs = stage / "logs/task_batches" / args.kind
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        for position in range(args.start, args.end):
            command = command_for(args, contract, runner, position)
            log = logs / f"task_{position:05d}.log"
            futures[executor.submit(run_one, position, command, log)] = position
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["position"]))
    failed = [row for row in results if int(row["returncode"]) != 0]
    require(not failed, f"task batch failed: {[row['position'] for row in failed]}")

    manifests = []
    for position, task in enumerate(selected, start=args.start):
        output = manifest_path(stage, str(task["output_prefix"]))
        require(output.is_file(), f"completed task lacks manifest: {output}")
        manifest = read_json(output)
        require(manifest.get("status") == "completed", f"task manifest did not complete: {output}")
        require(int(manifest.get("task_index", -1)) == position, f"task manifest index drift: {output}")
        manifests.append(file_binding(output))
    payload = {
        "schema_version": "targeted_8b_parallel_task_batch_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": args.kind,
        "task_range": {"start": args.start, "end_exclusive": args.end, "count": args.end - args.start},
        "parallel": args.parallel,
        "workers_per_task": args.workers_per_task if args.kind == "binary" else 0,
        "contract": file_binding(contract),
        "runner": file_binding(runner),
        "results": results,
        "task_manifests": manifests,
    }
    write_json_atomic(args.receipt, payload)
    print(json.dumps({"ok": True, "kind": args.kind, "tasks": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

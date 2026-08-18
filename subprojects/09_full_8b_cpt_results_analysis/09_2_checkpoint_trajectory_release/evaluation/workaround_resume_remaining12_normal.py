#!/usr/bin/env python3
"""Recover the frozen remaining-12 native-suite matrix inside an existing allocation.

This is deliberately an experiment-owned recovery driver, not a scorer.  It
only launches the already-frozen per-shard scorer command, skips completed
receipts, and runs the existing aggregate verifier once every shard is
present.  It exists because a nested ``srun`` attach step exposed one CPU to
the original segment wrapper, making a normal-allocation resume impossible.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import namedtuple
import time
from pathlib import Path


SHARDS = (
    ("demos", "demosqa", 0, 1),
    ("other_mcq", "medical_mcqa,asep_mcqa,gpcr", 0, 1),
    ("oyxoy_other", "oyxoy_nli,oyxoy_metaphor", 0, 1),
    *((f"oyxoy_wsd{i}", "oyxoy_wsd_definition", i, 10) for i in range(10)),
    *((f"oyxoy_wic{i}", "oyxoy_wic", i, 8) for i in range(8)),
)


Task = namedtuple("Task", "label model_path shard benchmarks row_index row_count")


def require_passed_receipt(path):
    data = json.loads(path.read_text())
    if data.get("status") not in {"passed", "frozen"}:
        raise ValueError(f"non-passing receipt: {path}")
    checks = data.get("checks")
    if isinstance(checks, dict) and not all(checks.values()):
        raise ValueError(f"failing check in receipt: {path}")


def completed(path, label):
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed" and json.loads(path.read_text()).get("model", {}).get("label") == label
    except (OSError, json.JSONDecodeError):
        return False


def task_command(task, args):
    log_stem = args.output_root / "logs" / f"{task.label}-{task.shard}-recovery-%j"
    return [
        "srun", f"--jobid={args.job_id}", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=54", "--mem=160G", "--gpus-per-node=1", "--gpus-per-task=1", "--cpu-bind=cores",
        f"--output={log_stem}.out", f"--error={log_stem}.err",
        "uenv", "run", "pytorch/v2.9.1:v2", "--view=default", "--",
        "env", f"PYTHONPATH={args.runtime_wheel}", "python3", str(args.runner),
        "--contract", str(args.contract), "--manifest", str(args.manifest), "--native-runner", str(args.native_runner),
        "--model", f"{task.label}={task.model_path}", "--output-dir", str(args.output_root / task.label / task.shard),
        "--dtype", "float32", "--scorer-mode", "legacy", "--benchmarks", task.benchmarks,
        "--candidate-batch-size", "1", "--example-batch-size", "16", "--max-examples-per-benchmark", "0",
        "--row-shard-index", str(task.row_index), "--row-shard-count", str(task.row_count),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--wait-for-existing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be 1..8 for the 2-node allocation")
    if args.wait_for_existing:
        while True:
            output = subprocess.check_output(
                ["squeue", "-h", "-s", "-j", args.job_id, "-o", "%i"],
                universal_newlines=True,
            )
            active = [line for line in output.splitlines() if line.startswith(args.job_id + ".") and line.split(".", 1)[1].isdigit()]
            if not active:
                break
            time.sleep(10)
        # Let each finished scorer flush and atomically publish its receipt.
        time.sleep(10)
    args.contract = args.assets_root / "remaining12_contract.json"
    args.manifest = args.assets_root / "remaining12_manifest.json"
    args.runtime_wheel = args.wrapper_root / "vendor" / "accelerate-1.14.0-py3-none-any.whl"
    args.runner = args.source_root / "subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
    args.native_runner = args.source_root / "subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
    args.aggregator = args.source_root / "subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
    for path in (args.wrapper_root / "bundle_receipt.json", args.assets_root / "rebind_receipt.json", args.contract, args.manifest, args.runtime_wheel, args.runner, args.native_runner, args.aggregator):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_passed_receipt(args.wrapper_root / "bundle_receipt.json")
    require_passed_receipt(args.assets_root / "rebind_receipt.json")
    contract = json.loads(args.contract.read_text())
    args.output_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    tasks = []
    quarantine = args.output_root / f"incomplete_from_recovery_{args.job_id}"
    for model in contract["checkpoint_scope"]:
        for shard, benchmarks, row_index, row_count in SHARDS:
            root = args.output_root / model["label"] / shard
            if completed(root / "receipt.json", model["label"]):
                continue
            if root.exists():
                destination = quarantine / model["label"] / shard
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                root.rename(destination)
            tasks.append(Task(model["label"], model["model_path"], shard, benchmarks, row_index, row_count))
    print(json.dumps({"action": "resume", "missing_tasks": len(tasks), "max_workers": args.max_workers}, sort_keys=True), flush=True)
    running = {}
    failures = []
    iterator = iter(tasks)
    while True:
        while len(running) < args.max_workers:
            try:
                task = next(iterator)
            except StopIteration:
                break
            process = subprocess.Popen(task_command(task, args), universal_newlines=True)
            running[process] = task
        if not running:
            break
        done = [process for process in running if process.poll() is not None]
        if not done:
            time.sleep(2)
            continue
        for process in done:
            task = running.pop(process)
            code = process.returncode
            print(json.dumps({"label": task.label, "shard": task.shard, "returncode": code}, sort_keys=True), flush=True)
            if code:
                failures.append({"label": task.label, "shard": task.shard, "returncode": code})
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    aggregate = subprocess.run([
        "uenv", "run", "pytorch/v2.9.1:v2", "--view=default", "--", "python3", str(args.aggregator),
        "--root", str(args.output_root), "--contract", str(args.contract), "--manifest", str(args.manifest),
    ], universal_newlines=True)
    return aggregate.returncode


if __name__ == "__main__":
    raise SystemExit(main())

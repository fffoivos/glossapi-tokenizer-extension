#!/usr/bin/env python3
"""Run one exact-checkpoint evaluation inside one debug allocation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def common_export(args: argparse.Namespace) -> str:
    values = [
            "ALL",
            f"FULL8_CODE_ROOT={args.code_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={args.code_bundle_receipt}",
            f"FULL8_OPS_ROOT={args.ops_root}",
            f"FULL8_OPS_BUNDLE_RECEIPT={args.ops_bundle_receipt}",
            f"FULL8_STAGE_ROOT={args.stage_root}",
            f"FULL8_RUN_ROOT={args.run_root}",
            f"FULL8_INITIAL_MEGATRON={args.initial_megatron}",
            f"FULL8_SELECTED_PROFILE={args.selected_profile}",
            "FULL8_EXECUTION_PROFILE=dp32_16node",
            f"FULL8_LAUNCH_GATE={args.launch_gate}",
            f"FULL8_PRELAUNCH_ROOT={args.prelaunch_root}",
            f"FULL8_RECIPE={args.recipe}",
            f"FULL8_PROFILES={args.profiles}",
            f"FULL8_TRAIN_LEAF_SWITCH={args.train_leaf_switch}",
            f"FULL8_EVALUATION_ITERATIONS={args.iterations}",
            f"FULL8_SOURCE_SEGMENT_ID={args.source_segment}",
            f"FULL8_SOURCE_TRAIN_JOB_ID={args.source_train_job}",
            f"FULL8_NEXT_TRAIN_JOB_ID={args.next_train_job}",
            f"FULL8_NEXT_SEGMENT_ID={'' if args.next_segment is None else args.next_segment}",
            f"FULL8_NEXT_SEGMENT_START={'' if args.next_segment_start is None else args.next_segment_start}",
        ]
    if args.prequeued_manifest is not None:
        values.append(f"FULL8_PREQUEUED_MANIFEST={args.prequeued_manifest}")
    if args.prequeue_schedule is not None:
        values.append(f"FULL8_PREQUEUE_SCHEDULE={args.prequeue_schedule}")
    return ",".join(values)


def submit(command: list[str]) -> str:
    failures: list[dict[str, object]] = []
    for attempt in range(1, 6):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        output = result.stdout.strip()
        job_id = output.split(";", 1)[0]
        if result.returncode == 0 and job_id.isdigit():
            return job_id
        failures.append({
            "attempt": attempt, "returncode": result.returncode,
            "stdout": output, "stderr": result.stderr.strip(),
        })
        if attempt < 5:
            time.sleep(2.0 * attempt)
    raise RuntimeError(
        "sbatch failed after five bounded attempts: "
        + json.dumps(failures, sort_keys=True)
    )


def submit_continuation(
    args: argparse.Namespace, *, attempt: int, needs_per_document: bool
) -> str:
    source_job_id = os.environ.get("SLURM_JOB_ID")
    if not source_job_id:
        raise RuntimeError("SLURM_JOB_ID is required")
    resources = [
        "--partition=debug", "--nodes=1",
        "--time=01:25:00" if needs_per_document else "--time=00:10:00",
        "--cpus-per-task=288" if needs_per_document else "--cpus-per-task=16",
        "--mem=450G" if needs_per_document else "--mem=64G",
    ]
    if needs_per_document:
        resources.extend(["--gpus-per-node=4", "--gres=gpu:4", "--exclusive"])
    command = [
        "sbatch", "--uenv-passthrough=ignore", "--parsable",
        *resources,
        f"--dependency=afterany:{source_job_id}",
        f"--job-name=full8b_eval_continue_{args.iterations.split(':')[args.index]}",
        f"--output={args.run_root}/logs/%x-%j.out",
        f"--error={args.run_root}/logs/%x-%j.err",
        f"--export={common_export(args)},FULL8_EVAL_INDEX={args.index},FULL8_EVAL_ATTEMPT={attempt},FULL8_EVAL_JOB_ID={source_job_id},FULL8_CONTINUE_ATTEMPT=0",
        str(args.ops_root / "clariden/continue_checkpoint_evaluation_debug.sbatch"),
    ]
    return submit(command)


def audit_job(args: argparse.Namespace, *, role: str, job_id: str) -> Path:
    output = (
        args.run_root / "orchestration/allocation_receipts"
        / f"{role}_{job_id}.json"
    )
    subprocess.run([
        sys.executable,
        str(args.ops_root / "scripts/audit_submitted_job_resources.py"),
        "--job", f"{role}={job_id}",
        "--scientific-root", str(args.code_root),
        "--scientific-receipt", str(args.code_bundle_receipt),
        "--ops-root", str(args.ops_root),
        "--ops-receipt", str(args.ops_bundle_receipt),
        "--output", str(output),
    ], check=True)
    return output


def audit_or_cancel(args: argparse.Namespace, *, role: str, job_id: str) -> Path:
    """Never leave an unaudited submitted child in the campaign graph."""

    try:
        return audit_job(args, role=role, job_id=job_id)
    except Exception:
        subprocess.run(["scancel", job_id], check=False)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--ops-bundle-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--prelaunch-root", type=Path, required=True)
    parser.add_argument("--initial-megatron", type=Path, required=True)
    parser.add_argument("--train-leaf-switch", required=True)
    parser.add_argument("--prequeued-manifest", type=Path)
    parser.add_argument("--prequeue-schedule", type=Path)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--source-segment", type=int, required=True)
    parser.add_argument("--source-train-job", required=True)
    parser.add_argument("--next-train-job", default="")
    parser.add_argument("--next-segment", type=int)
    parser.add_argument("--next-segment-start", type=int)
    args = parser.parse_args()
    for name in (
        "code_root", "code_bundle_receipt", "ops_root", "ops_bundle_receipt",
        "stage_root", "run_root", "recipe", "profiles", "selected_profile", "launch_gate",
        "prelaunch_root", "initial_megatron",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.prequeued_manifest is not None:
        args.prequeued_manifest = args.prequeued_manifest.resolve()
    if args.prequeue_schedule is not None:
        args.prequeue_schedule = args.prequeue_schedule.resolve()

    sys.path.insert(0, str(args.code_root / "subprojects/07_full_8b_cpt/scripts"))
    from contract import atomic_write_json, read_json

    iterations = [int(value) for value in args.iterations.split(":") if value]
    if not iterations or not 0 <= args.index < len(iterations):
        raise ValueError("invalid evaluation-chain index")
    iteration = iterations[args.index]
    recipe = read_json(args.recipe)
    allowed = {int(value) for value in recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]} - {0}
    if iteration not in allowed:
        raise ValueError(f"iteration {iteration} is not a frozen GreekMMLU milestone")
    per_document = {
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
    } - {0}
    needs_doc = iteration in per_document
    expected_nodes = 1
    if int(os.environ.get("SLURM_NNODES", "0")) != expected_nodes:
        raise RuntimeError(
            f"evaluation allocation drift: expected {expected_nodes} nodes"
        )
    if os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("checkpoint evaluation must run on debug")

    checkpoint = args.run_root / "checkpoints" / f"iter_{iteration:07d}" / ".metadata"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    attempt = int(os.environ.get("FULL8_EVAL_ATTEMPT", "0"))
    if attempt not in {0, 1, 2}:
        raise ValueError("evaluation attempt must be 0, 1, or 2")
    iteration_root = args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}"
    root = iteration_root / f"attempt_{attempt}"
    if root.exists():
        raise FileExistsError(root)
    export_root = root / "export"
    greek_root = root / "greekmmlu"
    doc_root = root / "per_document"
    receipt = root / "exact_checkpoint_native_greekmmlu_receipt.json"
    root.mkdir(parents=True)
    (args.run_root / "checkpoint_evaluation_logs").mkdir(parents=True, exist_ok=True)
    continuation_job = submit_continuation(
        args, attempt=attempt, needs_per_document=needs_doc
    )
    continuation_routing = audit_or_cancel(
        args,
        role=(
            "per_document_continuation"
            if needs_doc else "evaluation_continuation"
        ),
        job_id=continuation_job,
    )

    submission = {
        "schema_version": "apertus_full_8b_greekmmlu_submission_v1",
        "status": "running",
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": iteration,
        "attempt": attempt,
        "jobs": {
            "combined_debug_evaluation": os.environ.get("SLURM_JOB_ID"),
            "afterany_continuation": continuation_job,
        },
        "continuation_allocation_receipt": str(continuation_routing),
        "expected_receipt": str(receipt.resolve()),
        "resource_policy": {
            "partition": "debug", "nodes": expected_nodes,
            "one_running_job": True, "one_successor_max": True,
            "continuation_runs_per_document_inline": needs_doc,
        },
    }
    atomic_write_json(root / "submission.json", submission)

    evaluation_bundle = args.code_root / "subprojects/06_dataset_scheduling_experiments"
    native_root = (
        args.code_root
        / "subprojects/03_apertus_extension_and_embedding_adaptation"
        / "03_4_implementation_experiments/init_bakeoff/eval"
    )
    full8_eval_root = args.code_root / "subprojects/07_full_8b_cpt/evaluation"
    megatron = Path(os.environ.get(
        "FULL8_MEGATRON_ROOT",
        "/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1",
    ))
    tokenizer = Path(os.environ.get(
        "FULL8_TOKENIZER_ROOT",
        "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992",
    ))
    compat = Path(os.environ.get(
        "FULL8_PYTHON_COMPAT_DIR",
        "/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260802T230000Z-mini-b2-v8/compat",
    ))
    clean_subset = Path(os.environ.get(
        "FULL8_GREEKMMLU_CLEAN_SUBSET",
        "/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_subset_manifest.json",
    ))
    token_sha = "bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b"
    namespace = f"full8b_mixed_iter{iteration}"
    label = namespace

    convert_env = os.environ.copy()
    convert_env.update(
        EVALUATION_BUNDLE=str(evaluation_bundle),
        EVALUATION_CODE_BUNDLE_ROOT=str(args.code_root),
        EVALUATION_CODE_BUNDLE_RECEIPT=str(args.code_bundle_receipt),
        MEGATRON_DIR=str(megatron),
        SOURCE_CHECKPOINT_ROOT=str(args.run_root / "checkpoints"),
        SOURCE_ITERATION=str(iteration),
        TOKENIZER_DIR=str(tokenizer),
        EXPORT_ROOT=str(export_root),
        PYTHON_COMPAT_DIR=str(compat),
        TOKENIZER_SHA=token_sha,
        EXPORT_MODEL_SCALE="8B",
        FULL8_EVALUATION_ROOT=str(full8_eval_root),
    )
    subprocess.run(
        ["bash", str(evaluation_bundle / "clariden/convert_checkpoint_for_native_greekmmlu.sbatch")],
        env=convert_env, check=True,
    )

    greek_env = os.environ.copy()
    greek_env.update(
        NATIVE_GREEK_EVAL_ROOT=str(native_root),
        EXPORT_ROOT=str(export_root),
        GREEKMMLU_ROOT=str(greek_root),
        MODEL_LABEL=label,
        EVALUATION_NAMESPACE=namespace,
        EVAL_DTYPE="float32",
    )
    finalize_env = os.environ.copy()
    finalize_env.update(
        EVALUATION_BUNDLE=str(evaluation_bundle),
        EXPORT_RECEIPT=str(export_root / "checkpoint_eval_export_receipt.json"),
        GREEKMMLU_ROOT=str(greek_root),
        MODEL_LABEL=label,
        OUTPUT_RECEIPT=str(receipt),
        GREEKMMLU_CLEAN_SUBSET=str(clean_subset),
        EVALUATION_NAMESPACE=namespace,
    )
    subprocess.run(
        ["bash", str(evaluation_bundle / "clariden/run_checkpoint_native_greekmmlu.sbatch")],
        env=greek_env, check=True,
    )

    subprocess.run(
        ["bash", str(evaluation_bundle / "clariden/finalize_checkpoint_greekmmlu.sbatch")],
        env=finalize_env, check=True,
    )

    if needs_doc:
        doc_root.mkdir()
        submission["resource_policy"].update({
            "split_per_document": False,
            "sequential_groups_inside_continuation": 4,
            "total_node_minutes_per_job_max": 90,
        })
        submission["status"] = "greekmmlu_completed_waiting_for_per_document"
        atomic_write_json(root / "submission.json", submission, exclusive=False)
        print(json.dumps({
            "ok": True,
            "iteration": iteration,
            "continuation_job": continuation_job,
            "per_document_execution": "inline_in_continuation",
        }))
        return 0

    canonical = {
        "schema_version": "apertus_full_8b_authoritative_checkpoint_evaluation_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": iteration,
        "attempt": attempt,
        "attempt_root": str(root.resolve()),
        "greekmmlu_receipt": str(receipt.resolve()),
        "per_document_root": str(doc_root.resolve()) if needs_doc else None,
    }
    atomic_write_json(iteration_root / "authoritative_attempt.json", canonical)
    print(json.dumps({
        "ok": True,
        "iteration": iteration,
        "continuation_job": continuation_job,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

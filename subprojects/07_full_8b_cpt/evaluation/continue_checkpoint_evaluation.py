#!/usr/bin/env python3
"""Inspect one debug evaluation and advance or recover its serial chain."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


RETRYABLE = {"BOOT_FAIL", "FAILED", "NODE_FAIL", "PREEMPTED", "REVOKED", "TIMEOUT"}


def slurm_state(job_id: str, *, attempts: int = 10, delay_seconds: float = 3.0) -> str:
    for attempt in range(attempts):
        result = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
            text=True, capture_output=True, check=False,
        )
        rows = [
            row.split("|", 1)[0].split()[0].split("+", 1)[0]
            for row in result.stdout.splitlines()
            if row.strip()
        ]
        if result.returncode == 0 and rows and rows[0] != "UNKNOWN":
            return rows[0]
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return "UNKNOWN"


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
    full_command = ["sbatch", "--uenv-passthrough=ignore", "--parsable", *command]
    failures: list[dict[str, object]] = []
    for attempt in range(1, 6):
        result = subprocess.run(
            full_command, text=True, capture_output=True, check=False
        )
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


def greekmmlu_receipt_is_complete(
    path: Path, *, iteration: int, read_json
) -> bool:
    if not path.is_file():
        return False
    value = read_json(path)
    return (
        value.get("schema_version") == "exact_checkpoint_native_greekmmlu_receipt_v1"
        and value.get("status") == "completed"
        and int(value.get("checkpoint", {}).get("iteration", -1)) == iteration
    )


def run_per_document_inline(
    args: argparse.Namespace,
    *,
    iteration: int,
    attempt: int,
    attempt_root: Path,
    greekmmlu_receipt: Path,
    atomic_write_json,
    read_json,
) -> None:
    """Score the four frozen panel groups without a nested Slurm submission."""

    if os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("per-document continuation must run on debug")
    if int(os.environ.get("SLURM_NNODES", "0")) != 1:
        raise RuntimeError("per-document continuation must use one node")
    visible = [
        value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value
    ]
    if len(visible) < 4:
        raise RuntimeError(
            "per-document continuation requires four Slurm-assigned GPUs"
        )

    manifest_path = args.stage_root / "validation/validation_manifest.json"
    manifest = read_json(manifest_path)
    panels = manifest.get("panels", [])
    if manifest.get("status") != "frozen" or len(panels) != 13:
        raise ValueError("validation manifest is not the frozen 13-panel contract")

    doc_root = attempt_root / "per_document"
    doc_root.mkdir(exist_ok=True)
    export_root = attempt_root / "export"
    tokenizer = Path(os.environ.get(
        "FULL8_TOKENIZER_ROOT",
        "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992",
    ))
    group_script = args.ops_root / "clariden/run_per_document_group_resource_aware.sh"
    environment = os.environ.copy()
    environment.update(
        FULL8_CODE_ROOT=str(args.code_root),
        FULL8_CODE_BUNDLE_RECEIPT=str(args.code_bundle_receipt),
        FULL8_OPS_ROOT=str(args.ops_root),
        FULL8_OPS_BUNDLE_RECEIPT=str(args.ops_bundle_receipt),
        FULL8_VALIDATION_MANIFEST=str(manifest_path),
        FULL8_HF_MODEL=str(export_root / "hf"),
        FULL8_HF_TOKENIZER=str(tokenizer),
        FULL8_DOCVAL_OUTPUT=str(doc_root),
    )
    for group in range(4):
        group_panels = panels[group * 4 : min(group * 4 + 4, len(panels))]
        if group_panels and all(
            (doc_root / f"{panel['name']}.receipt.json").is_file()
            and (doc_root / f"{panel['name']}.documents.jsonl").is_file()
            for panel in group_panels
        ):
            continue
        environment["SLURM_ARRAY_TASK_ID"] = str(group)
        subprocess.run(["bash", str(group_script)], env=environment, check=True)

    iteration_root = attempt_root.parent
    subprocess.run([
        sys.executable,
        str(args.ops_root / "evaluation/finalize_split_checkpoint_evaluation.py"),
        "--iteration", str(iteration),
        "--attempt", str(attempt),
        "--iteration-root", str(iteration_root),
        "--attempt-root", str(attempt_root),
        "--greekmmlu-receipt", str(greekmmlu_receipt),
        "--validation-manifest", str(manifest_path),
        "--per-document-root", str(doc_root),
    ], check=True)
    submission_path = attempt_root / "submission.json"
    if submission_path.is_file():
        submission = read_json(submission_path)
        submission["status"] = "completed"
        submission.setdefault("jobs", {})["inline_per_document_continuation"] = (
            os.environ.get("SLURM_JOB_ID")
        )
        submission.setdefault("resource_policy", {}).update({
            "qos_submit_slots_used_max": 2,
            "nested_debug_submissions": 0,
            "sequential_groups_inside_continuation": 4,
        })
        atomic_write_json(submission_path, submission, exclusive=False)


def evaluation_command(
    args: argparse.Namespace, *, index: int, attempt: int, per_document: set[int]
) -> list[str]:
    iterations = [int(value) for value in args.iterations.split(":") if value]
    iteration = iterations[index]
    # Even per-document milestones start on one node.  The evaluator converts
    # and scores GreekMMLU there, then fans the 13 frozen panels into four
    # independent one-node debug jobs.  This keeps every job below a0140's
    # 90 total node-minute QOS cap.
    nodes = 1
    time_limit = "01:25:00"
    return [
        "--partition=debug", f"--time={time_limit}", f"--nodes={nodes}",
        "--ntasks-per-node=1",
        f"--job-name=full8b_eval_{iteration}",
        f"--output={args.run_root}/logs/%x-%j.out",
        f"--error={args.run_root}/logs/%x-%j.err",
        f"--export={common_export(args)},FULL8_EVAL_INDEX={index},FULL8_EVAL_ATTEMPT={attempt}",
        str(args.ops_root / "clariden/run_checkpoint_evaluation_debug.sbatch"),
    ]


def write_event(atomic_write_json, args: argparse.Namespace, name: str, payload: dict) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    root = args.run_root / "orchestration/events"
    value = {
        "schema_version": "apertus_full_8b_campaign_event_v1",
        "status": "recorded",
        "recorded_at": now.isoformat(),
        "event": name,
        "source_segment": args.source_segment,
        **payload,
    }
    atomic_write_json(
        root / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{name}_eval.json", value
    )
    atomic_write_json(
        args.run_root / "orchestration/latest.json", value, exclusive=False
    )


def write_or_verify_queue_receipt(
    atomic_write_json, read_json, path: Path, payload: dict
) -> str:
    """Make a final continuation safe to rerun after its queue was completed."""

    if not path.exists():
        atomic_write_json(path, payload)
        return "written"
    existing = read_json(path)
    comparable_existing = dict(existing)
    comparable_payload = dict(payload)
    comparable_existing.pop("completed_at", None)
    comparable_payload.pop("completed_at", None)
    if comparable_existing != comparable_payload:
        raise ValueError(f"evaluation queue receipt drift: {path}")
    return "verified_existing"


def supervisor_submission_receipt(args: argparse.Namespace) -> Path:
    return (
        args.run_root / "orchestration/supervisor_submission_receipts"
        / f"segment_{args.next_segment}.json"
    )


def adopt_existing_supervisor_submission(
    args: argparse.Namespace, read_json, atomic_write_json
) -> dict | None:
    """Adopt or finish auditing a previously submitted exact supervisor."""

    path = supervisor_submission_receipt(args)
    if not path.exists():
        return None
    value = read_json(path)
    expected = {
        "schema_version": "apertus_full_8b_supervisor_submission_v1",
        "next_segment": args.next_segment,
        "next_segment_start": args.next_segment_start,
        "source_train_job": args.next_train_job,
        "operational_root": str(args.ops_root),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"supervisor submission receipt drift for {key}")
    job_id = str(value.get("supervisor_job", ""))
    if not job_id.isdigit():
        raise ValueError("supervisor submission receipt lacks a job id")
    state = slurm_state(job_id)
    if state not in {"PENDING", "RUNNING", "COMPLETED"}:
        raise RuntimeError(
            f"recorded supervisor {job_id} is {state}; refusing a duplicate"
        )
    status = value.get("status")
    if status == "submitted_unverified":
        routing = audit_job(args, role="supervisor", job_id=job_id)
        value = {
            **value,
            "status": "passed",
            "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "allocation_routing_receipt": str(routing),
        }
        atomic_write_json(path, value, exclusive=False)
    elif status != "passed":
        raise RuntimeError(f"supervisor submission receipt has status {status}")
    routing = Path(value["allocation_routing_receipt"])
    if not routing.is_file():
        raise FileNotFoundError(routing)
    return value


def submit_supervisor_with_receipt(
    args: argparse.Namespace, atomic_write_json, command: list[str]
) -> dict:
    """Submit exactly one supervisor and make its audit restart-safe."""

    path = supervisor_submission_receipt(args)
    supervisor = submit(command)
    value = {
        "schema_version": "apertus_full_8b_supervisor_submission_v1",
        "status": "submitted_unverified",
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "next_segment": args.next_segment,
        "next_segment_start": args.next_segment_start,
        "source_train_job": args.next_train_job,
        "supervisor_job": supervisor,
        "operational_root": str(args.ops_root),
    }
    try:
        atomic_write_json(path, value)
    except Exception:
        subprocess.run(["scancel", supervisor], check=False)
        raise
    try:
        routing = audit_job(args, role="supervisor", job_id=supervisor)
    except Exception as error:
        subprocess.run(["scancel", supervisor], check=False)
        atomic_write_json(path, {
            **value,
            "status": "rejected",
            "rejected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": repr(error),
        }, exclusive=False)
        raise
    value = {
        **value,
        "status": "passed",
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "allocation_routing_receipt": str(routing),
    }
    atomic_write_json(path, value, exclusive=False)
    return value


def prequeue_successor(args: argparse.Namespace, read_json) -> dict | None:
    """Install one delayed normal holder without consuming a debug-QOS slot."""

    if args.prequeued_manifest is None or args.prequeue_schedule is None:
        return None
    if args.next_segment is None or not args.next_train_job:
        return None
    selected = read_json(args.selected_profile)
    boundaries = [int(value) for value in selected["selection"]["segment_boundaries"]]
    target_segment = args.next_segment + 1
    if target_segment >= len(boundaries) - 1:
        return None
    schedule = read_json(args.prequeue_schedule)
    policies = [
        row for row in schedule.get("targets", [])
        if int(row["target_segment_id"]) == target_segment
    ]
    if len(policies) != 1:
        raise ValueError("prequeue schedule lacks one exact successor policy")
    policy = policies[0]
    receipt = (
        args.run_root / "orchestration/prequeue_submission_receipts"
        / f"segment_{target_segment}.json"
    )
    command = [
        sys.executable, str(args.ops_root / "scripts/prequeue_next_segment.py"),
        "--scientific-root", str(args.code_root),
        "--scientific-receipt", str(args.code_bundle_receipt),
        "--ops-root", str(args.ops_root),
        "--ops-receipt", str(args.ops_bundle_receipt),
        "--stage-root", str(args.stage_root),
        "--run-root", str(args.run_root),
        "--initial-megatron", str(args.initial_megatron),
        "--selected-profile", str(args.selected_profile),
        "--launch-gate", str(args.launch_gate),
        "--prelaunch-root", str(args.prelaunch_root),
        "--recipe", str(args.recipe),
        "--profiles", str(args.profiles),
        "--train-leaf-switch", args.train_leaf_switch,
        "--manifest", str(args.prequeued_manifest),
        "--schedule", str(args.prequeue_schedule),
        "--source-segment", str(args.next_segment),
        "--source-train-job", args.next_train_job,
        "--target-segment", str(target_segment),
        "--minimum-train-seconds", str(int(policy["minimum_train_seconds"])),
        "--maximum-hold-seconds", str(int(policy["maximum_hold_seconds"])),
        "--eligible-after-minutes", str(int(policy["source_trigger_minutes"])),
        "--output", str(receipt),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {
            "status": "not_installed",
            "target_segment_id": target_segment,
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
        }
    return read_json(receipt)


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
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--continue-attempt", type=int, default=0)
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
    if args.attempt not in {0, 1, 2}:
        raise ValueError("invalid evaluation attempt")
    iteration = iterations[args.index]
    recipe = read_json(args.recipe)
    per_document = {
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
    } - {0}
    authoritative = (
        args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}"
        / "authoritative_attempt.json"
    )
    state = slurm_state(args.evaluation_job)

    if state == "UNKNOWN":
        if args.continue_attempt >= 4:
            write_event(atomic_write_json, args, "campaign_blocked", {
                "reason": "evaluation_slurm_state_remained_unknown",
                "evaluation_job": args.evaluation_job,
                "iteration": iteration,
            })
            raise RuntimeError("evaluation Slurm state remained UNKNOWN")
        retry = submit([
            "--partition=debug", "--time=00:10:00", "--nodes=1",
            "--begin=now+2minutes",
            f"--job-name=full8b_eval_continue_{iteration}",
            f"--output={args.run_root}/logs/%x-%j.out",
            f"--error={args.run_root}/logs/%x-%j.err",
            f"--export={common_export(args)},FULL8_EVAL_INDEX={args.index},FULL8_EVAL_ATTEMPT={args.attempt},FULL8_EVAL_JOB_ID={args.evaluation_job},FULL8_CONTINUE_ATTEMPT={args.continue_attempt + 1}",
            str(args.ops_root / "clariden/continue_checkpoint_evaluation_debug.sbatch"),
        ])
        routing = audit_or_cancel(
            args, role="evaluation_continuation", job_id=retry
        )
        write_event(atomic_write_json, args, "evaluation_state_recheck_submitted", {
            "iteration": iteration, "evaluation_job": args.evaluation_job,
            "continuation_job": retry, "continue_attempt": args.continue_attempt + 1,
            "allocation_routing_receipt": str(routing),
        })
        return 0

    completed = False
    if authoritative.is_file():
        value = read_json(authoritative)
        completed = (
            value.get("schema_version") == "apertus_full_8b_authoritative_checkpoint_evaluation_v1"
            and value.get("status") == "completed"
            and int(value.get("iteration", -1)) == iteration
            and int(value.get("attempt", -1)) == args.attempt
            and Path(value["greekmmlu_receipt"]).is_file()
            and (
                iteration not in per_document
                or Path(value["per_document_root"]).is_dir()
            )
        )

    attempt_root = (
        args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}"
        / f"attempt_{args.attempt}"
    )
    greekmmlu_receipt = (
        attempt_root / "exact_checkpoint_native_greekmmlu_receipt.json"
    )
    greekmmlu_ready = greekmmlu_receipt_is_complete(
        greekmmlu_receipt, iteration=iteration, read_json=read_json
    )
    if (
        iteration in per_document
        and not completed
        and greekmmlu_ready
        and state in ({"COMPLETED"} | RETRYABLE)
    ):
        run_per_document_inline(
            args,
            iteration=iteration,
            attempt=args.attempt,
            attempt_root=attempt_root,
            greekmmlu_receipt=greekmmlu_receipt,
            atomic_write_json=atomic_write_json,
            read_json=read_json,
        )
        authoritative_value = read_json(authoritative)
        completed = (
            authoritative_value.get("status") == "completed"
            and int(authoritative_value.get("iteration", -1)) == iteration
            and int(authoritative_value.get("attempt", -1)) == args.attempt
        )
        state = "COMPLETED"
        write_event(atomic_write_json, args, "per_document_inline_completed", {
            "iteration": iteration,
            "evaluation_job": args.evaluation_job,
            "evaluation_terminal_state": slurm_state(args.evaluation_job),
            "attempt": args.attempt,
            "per_document_root": str(attempt_root / "per_document"),
        })

    if state != "COMPLETED" or not completed:
        if state not in RETRYABLE or args.attempt >= 2:
            write_event(atomic_write_json, args, "campaign_blocked", {
                "reason": "evaluation_failed_or_evidence_missing",
                "iteration": iteration, "evaluation_job": args.evaluation_job,
                "slurm_state": state, "authoritative_evidence": completed,
            })
            raise RuntimeError(
                f"evaluation cannot be recovered automatically: state={state}, evidence={completed}"
            )
        replacement = submit(
            evaluation_command(
                args, index=args.index, attempt=args.attempt + 1,
                per_document=per_document,
            )
        )
        routing = audit_or_cancel(args, role="evaluation", job_id=replacement)
        write_event(atomic_write_json, args, "evaluation_retry_submitted", {
            "iteration": iteration, "failed_job": args.evaluation_job,
            "slurm_state": state, "replacement_job": replacement,
            "attempt": args.attempt + 1,
            "allocation_routing_receipt": str(routing),
        })
        return 0

    if args.index + 1 < len(iterations):
        next_job = submit(
            evaluation_command(
                args, index=args.index + 1, attempt=0, per_document=per_document
            )
        )
        routing = audit_or_cancel(args, role="evaluation", job_id=next_job)
        write_event(atomic_write_json, args, "next_evaluation_submitted", {
            "completed_iteration": iteration,
            "next_iteration": iterations[args.index + 1],
            "next_evaluation_job": next_job,
            "allocation_routing_receipt": str(routing),
        })
        return 0

    queue_receipt = args.run_root / "evaluation_queues" / f"segment_{args.source_segment}.json"
    queue_receipt_action = write_or_verify_queue_receipt(
        atomic_write_json, read_json, queue_receipt, {
        "schema_version": "apertus_full_8b_evaluation_queue_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iterations": iterations,
        "max_active": 1,
        "resource_policy": "serial_debug_afterany_continuation_v1",
    })
    write_event(atomic_write_json, args, "evaluation_queue_receipt_processed", {
        "queue_receipt": str(queue_receipt),
        "action": queue_receipt_action,
    })

    if args.next_train_job:
        if args.next_segment is None or args.next_segment_start is None:
            raise ValueError("next train job lacks supervisor coordinates")
        existing_supervisor = adopt_existing_supervisor_submission(
            args, read_json, atomic_write_json
        )
        if existing_supervisor is not None:
            write_event(atomic_write_json, args, "next_segment_supervisor_adopted", {
                "completed_iteration": iteration,
                "next_segment": args.next_segment,
                "next_train_job": args.next_train_job,
                "next_supervisor_job": existing_supervisor["supervisor_job"],
                "allocation_routing_receipt": existing_supervisor[
                    "allocation_routing_receipt"
                ],
            })
            return 0
        successor_prequeue = prequeue_successor(args, read_json)
        if successor_prequeue is not None:
            write_event(atomic_write_json, args, "successor_prequeue_processed", {
                "source_segment": args.next_segment,
                "source_train_job": args.next_train_job,
                "prequeue": successor_prequeue,
            })
        supervisor_submission = submit_supervisor_with_receipt(
            args, atomic_write_json, [
                "--partition=debug", "--time=00:20:00", "--nodes=1",
                f"--dependency=afterany:{args.next_train_job}",
                f"--job-name=full8b_supervise_s{args.next_segment}a0",
                f"--output={args.run_root}/logs/%x-%j.out",
                f"--error={args.run_root}/logs/%x-%j.err",
                f"--export={common_export(args)}",
                str(
                    args.ops_root
                    / "clariden/supervise_campaign_resource_aware.sbatch"
                ),
                str(args.next_segment), "0", str(args.next_segment_start),
                args.next_train_job,
            ]
        )
        supervisor = supervisor_submission["supervisor_job"]
        write_event(atomic_write_json, args, "next_segment_supervisor_submitted", {
            "completed_iteration": iteration,
            "next_segment": args.next_segment,
            "next_train_job": args.next_train_job,
            "next_supervisor_job": supervisor,
            "allocation_routing_receipt": supervisor_submission[
                "allocation_routing_receipt"
            ],
        })
        return 0

    training_receipt = args.run_root / "training_completion_receipt.json"
    subprocess.run([
        sys.executable,
        str(args.code_root / "subprojects/07_full_8b_cpt/scripts/finalize_training.py"),
        "--run-root", str(args.run_root),
        "--selected-profile", str(args.selected_profile),
        "--launch-gate", str(args.launch_gate),
        "--recipe", str(args.recipe),
        "--output", str(training_receipt),
    ], check=True)
    subprocess.run([
        sys.executable,
        str(args.code_root / "subprojects/07_full_8b_cpt/scripts/finalize_campaign.py"),
        "--run-root", str(args.run_root),
        "--launch-gate", str(args.launch_gate),
        "--recipe", str(args.recipe),
        "--selected-profile", str(args.selected_profile),
        "--training-receipt", str(training_receipt),
        "--output", str(args.run_root / "campaign_evidence_completion_receipt.json"),
        "--max-seconds", "60", "--poll-seconds", "5",
    ], check=True)
    write_event(atomic_write_json, args, "campaign_evidence_completed", {
        "final_iteration": iteration,
        "training_receipt": str(training_receipt),
    })
    print(json.dumps({"ok": True, "iteration": iteration, "campaign_complete": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

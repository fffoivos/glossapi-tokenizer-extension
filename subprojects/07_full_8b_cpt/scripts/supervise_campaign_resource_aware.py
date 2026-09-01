#!/usr/bin/env python3
"""Gate one full-8B segment, recover infrastructure failures, and continue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import re
import tempfile
import time
from pathlib import Path

_code_root = os.environ.get("FULL8_CODE_ROOT")
if not _code_root:
    raise RuntimeError("FULL8_CODE_ROOT is required before importing the supervisor")
sys.path.insert(
    0, str(Path(_code_root) / "subprojects/07_full_8b_cpt/scripts")
)
from contract import atomic_write_json, read_json


RETRYABLE = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED", "TIMEOUT"}
ITERATION = re.compile(r"iteration\s+(\d+)\s*/")


def slurm_state(job_id: str, *, attempts: int = 5, delay_seconds: float = 2.0) -> str:
    """Read a terminal Slurm state without treating one empty sacct read as fact."""
    for attempt in range(attempts):
        result = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
            text=True, capture_output=True, check=False,
        )
        rows = [
            row.split("|")[0].split()[0].split("+")[0]
            for row in result.stdout.splitlines()
            if row.strip()
        ]
        if result.returncode == 0 and rows and rows[0] != "UNKNOWN":
            return rows[0]
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return "UNKNOWN"


def active_job_state(
    job_id: str, *, attempts: int = 5, delay_seconds: float = 2.0
) -> str:
    """Return a live Slurm state without confusing a controller error with exit.

    A prequeued holder may have exited while its source segment was still
    writing its checkpoint gate.  A missing ``squeue`` row is therefore an
    expected terminal condition, but a failed controller query is not enough
    evidence to submit a second training leaf.
    """

    for attempt in range(attempts):
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            text=True,
            capture_output=True,
            check=False,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        if result.returncode == 0:
            return rows[0] if rows else "ABSENT"
        # Once Slurm has purged a completed holder, this controller reports an
        # invalid job id rather than an otherwise successful empty query.  It
        # is terminal evidence, provided that sacct confirms the final state
        # before any replacement leaf is submitted.
        if "invalid job id" in result.stderr.lower():
            return "ABSENT"
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return "UNKNOWN"


def allow_nested_sbatch(command: list[str]) -> list[str]:
    """Make sbatch legal when this supervisor itself runs inside uenv."""
    if not command or Path(command[0]).name != "sbatch":
        raise ValueError("supervisor submit() only accepts sbatch commands")
    if "--uenv-passthrough=ignore" in command:
        return list(command)
    return [command[0], "--uenv-passthrough=ignore", *command[1:]]


def submit(command: list[str]) -> str:
    nested = allow_nested_sbatch(command)
    failures: list[dict[str, object]] = []
    for attempt in range(1, 6):
        result = subprocess.run(
            nested, text=True, capture_output=True, check=False
        )
        output = result.stdout.strip()
        job_id = output.split(";", 1)[0]
        if result.returncode == 0 and job_id.isdigit():
            return job_id
        failures.append({
            "attempt": attempt,
            "returncode": result.returncode,
            "stdout": output,
            "stderr": result.stderr.strip(),
        })
        if attempt < 5:
            time.sleep(2.0 * attempt)
    raise RuntimeError(
        "sbatch failed after five bounded attempts: "
        + json.dumps(failures, sort_keys=True)
    )


def audit_submitted_jobs(
    args: argparse.Namespace, *, label: str, jobs: list[tuple[str, str]]
) -> Path:
    output = (
        args.run_root / "orchestration/allocation_receipts"
        / f"{label}_{'_'.join(job for _, job in jobs)}.json"
    )
    command = [
        sys.executable,
        str(args.ops_root / "scripts/audit_submitted_job_resources.py"),
        "--scientific-root", str(args.code_root),
        "--scientific-receipt", str(args.code_bundle_receipt),
        "--ops-root", str(args.ops_root),
        "--ops-receipt", str(args.ops_bundle_receipt),
        "--output", str(output),
    ]
    for role, job_id in jobs:
        command.extend(("--job", f"{role}={job_id}"))
    subprocess.run(command, check=True)
    return output


def run_or_verify_immutable_receipt(command: list[str], output: Path) -> None:
    """Reproduce an existing receipt without overwriting immutable evidence."""

    if not output.exists():
        subprocess.run(command, check=True)
        return
    try:
        output_index = command.index("--output") + 1
    except (ValueError, IndexError) as error:
        raise ValueError("immutable receipt command lacks --output") from error
    with tempfile.TemporaryDirectory(prefix="full8b-receipt-recheck-") as temporary:
        candidate_path = Path(temporary) / output.name
        candidate_command = list(command)
        candidate_command[output_index] = str(candidate_path)
        subprocess.run(candidate_command, check=True)
        existing = read_json(output)
        candidate = read_json(candidate_path)
    for payload in (existing, candidate):
        payload.pop("completed_at", None)
    if existing != candidate:
        raise ValueError(f"immutable receipt reproduction drift: {output}")


def encode_evaluation_iterations(iterations: list[int]) -> str:
    """Encode a list without Slurm's comma-delimited --export separator."""
    return ":".join(map(str, iterations))


def event(args: argparse.Namespace, name: str, payload: dict) -> None:
    root = args.run_root / "orchestration" / "events"
    now = dt.datetime.now(dt.timezone.utc)
    value = {
        "schema_version": "apertus_full_8b_campaign_event_v1",
        "status": "recorded",
        "recorded_at": now.isoformat(),
        "event": name,
        "segment_id": args.segment_id,
        "attempt": args.attempt,
        **payload,
    }
    atomic_write_json(root / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{name}_s{args.segment_id}_a{args.attempt}.json", value)
    atomic_write_json(args.run_root / "orchestration" / "latest.json", value, exclusive=False)


def common_export(args: argparse.Namespace) -> str:
    values = [
            f"FULL8_CODE_ROOT={args.code_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={args.code_bundle_receipt}",
            f"FULL8_STAGE_ROOT={args.stage_root}",
            f"FULL8_RUN_ROOT={args.run_root}",
            f"FULL8_INITIAL_MEGATRON={args.initial_megatron}",
            f"FULL8_SELECTED_PROFILE={args.selected_profile}",
            f"FULL8_EXECUTION_PROFILE={args.profile_id}",
            f"FULL8_LAUNCH_GATE={args.launch_gate}",
            f"FULL8_PRELAUNCH_ROOT={args.prelaunch_root}",
            f"FULL8_RECIPE={args.recipe}",
            f"FULL8_PROFILES={args.profiles}",
            f"FULL8_TRAIN_LEAF_SWITCH={args.train_leaf_switch}",
            f"FULL8_OPS_ROOT={args.ops_root}",
            f"FULL8_OPS_BUNDLE_RECEIPT={args.ops_bundle_receipt}",
    ]
    if args.prequeued_manifest is not None:
        values.append(f"FULL8_PREQUEUED_MANIFEST={args.prequeued_manifest}")
    if args.prequeue_schedule is not None:
        values.append(f"FULL8_PREQUEUE_SCHEDULE={args.prequeue_schedule}")
    return "ALL," + ",".join(values)


def cancel_prequeued_suffix(args: argparse.Namespace) -> list[str]:
    """Cancel only the rejected happy-path suffix before dynamic recovery."""

    if args.prequeued_manifest is None:
        return []
    graph = read_json(args.prequeued_manifest)
    if (
        graph.get("schema_version") != "apertus_full_8b_prequeued_launch_graph_v1"
        or graph.get("status") != "submitted"
    ):
        raise ValueError("prequeued launch graph drift")
    job_ids: list[str] = []
    for row in graph.get("segments", []):
        if int(row["segment_id"]) >= args.segment_id:
            for key in ("evaluation_queue_job",):
                if row.get(key):
                    job_ids.append(str(row[key]))
        if int(row["segment_id"]) > args.segment_id:
            for key in ("train_job", "supervisor_job"):
                if row.get(key):
                    job_ids.append(str(row[key]))
    if graph.get("evidence_finalizer_job"):
        job_ids.append(str(graph["evidence_finalizer_job"]))
    job_ids = sorted(set(job_ids))
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=True)
    args.prequeued_manifest = None
    return job_ids


def adopt_prequeued_train(
    args: argparse.Namespace, *, source_segment: int, target_segment: int,
    source_train_job: str, start: int, end: int,
) -> tuple[str, Path] | None:
    if args.prequeued_manifest is None:
        return None
    graph = read_json(args.prequeued_manifest)
    if (
        graph.get("schema_version") != "apertus_full_8b_prequeued_launch_graph_v1"
        or graph.get("status") != "submitted"
    ):
        raise ValueError("prequeued launch graph drift")
    matches = [
        row for row in graph.get("segments", [])
        if int(row["segment_id"]) == target_segment
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("duplicate prequeued target segment")
    row = matches[0]
    expected = {
        "source_segment_id": source_segment,
        "source_train_job": source_train_job,
        "start_iteration": start,
        "end_iteration": end,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"prequeued segment binding drift for {key}")
    job_id = str(row["train_job"])
    permit = Path(row["permit_path"]).resolve()
    if permit.exists():
        raise FileExistsError(f"prequeued permit already exists: {permit}")
    live_state = active_job_state(job_id)
    if live_state in {"PENDING", "RUNNING"}:
        return job_id, permit
    if live_state == "UNKNOWN":
        raise RuntimeError(
            f"prequeued holder {job_id} state is unavailable; refusing a duplicate leaf"
        )

    terminal_state = slurm_state(job_id)
    if terminal_state == "UNKNOWN":
        raise RuntimeError(
            f"prequeued holder {job_id} is absent from squeue but terminal state is unavailable"
        )
    if terminal_state in {"PENDING", "RUNNING"}:
        # A short squeue/sacct visibility race must not create a duplicate leaf.
        return job_id, permit
    if terminal_state == "COMPLETED":
        raise RuntimeError(
            f"prequeued holder {job_id} completed before source segment {source_segment} "
            "could issue its permit"
        )

    # The holder never received a source checkpoint permit, so no segment work
    # could have started.  Fall back to a fresh, directly queued leaf from the
    # just-frozen source checkpoint instead of issuing a permit to this terminal
    # allocation.  This changes queue mechanics only, not samples or training.
    event(args, "prequeued_holder_terminal_fallback", {
        "prequeued_train_job": job_id,
        "prequeued_live_state": live_state,
        "prequeued_terminal_state": terminal_state,
        "source_segment": source_segment,
        "target_segment": target_segment,
        "updates": [start, end],
    })
    return None


def last_logged_iteration(path: Path) -> int | None:
    if not path.is_file():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITERATION.search(line)
        if match:
            latest = int(match.group(1))
    return latest


def retryable_terminal(
    state: str, graceful_marker: Path, latest: int | None, end: int,
    *, verified_recovery_checkpoint: bool = False,
) -> bool:
    if state in RETRYABLE:
        return True
    if state == "FAILED" and verified_recovery_checkpoint:
        return True
    return graceful_marker.is_file() and latest is not None and latest < end


def exact_load_view(args: argparse.Namespace, *, segment: int, attempt: int, iteration: int) -> Path:
    root = args.run_root / "recovery_load_views" / f"segment_{segment}_attempt_{attempt}_iter_{iteration:07d}"
    if root.exists():
        raise FileExistsError(root)
    source = args.run_root / "checkpoints" / f"iter_{iteration:07d}"
    if not (source / ".metadata").is_file():
        raise ValueError(f"exact fallback checkpoint is incomplete: {source}")
    root.mkdir(parents=True)
    os.symlink(source.resolve(), root / source.name)
    (root / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n", encoding="utf-8")
    return root


def submit_attempt(args: argparse.Namespace, *, segment: int, attempt: int, start: int, recovery: bool, load_override: Path | None = None) -> tuple[str, str]:
    end = args.boundaries[segment + 1]
    load = load_override or (args.initial_megatron if start == 0 else args.run_root / "checkpoints")
    logs = args.run_root / "logs"
    exports = common_export(args) + "," + ",".join(
        (
            f"FULL8_SEGMENT_ID={segment}", f"FULL8_ATTEMPT={attempt}",
            f"FULL8_START_ITERATION={start}", f"FULL8_END_ITERATION={end}",
            f"FULL8_LOAD_CHECKPOINT={load}", f"FULL8_RECOVERY_MODE={1 if recovery else 0}",
        )
    )
    train = submit([
        "sbatch", "--parsable", "--partition=normal", "--time=12:00:00",
        "--switches=1", f"--exclude={args.train_exclude}", f"--nodes={args.nodes}", f"--job-name=full8b_s{segment}a{attempt}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err", f"--export={exports}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"),
    ])
    supervisor = submit([
        "sbatch", "--parsable", "--partition=debug", "--time=00:20:00",
        "--nodes=1", f"--dependency=afterany:{train}", f"--job-name=full8b_supervise_s{segment}a{attempt}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err", f"--export={common_export(args)}",
        str(args.ops_root / "clariden/supervise_campaign_resource_aware.sbatch"),
        str(segment), str(attempt), str(start), train,
    ])
    return train, supervisor


def submit_train_only(
    args: argparse.Namespace, *, segment: int, attempt: int, start: int
) -> str:
    end = args.boundaries[segment + 1]
    load = args.initial_megatron if start == 0 else args.run_root / "checkpoints"
    exports = common_export(args) + "," + ",".join(
        (
            f"FULL8_SEGMENT_ID={segment}", f"FULL8_ATTEMPT={attempt}",
            f"FULL8_START_ITERATION={start}", f"FULL8_END_ITERATION={end}",
            f"FULL8_LOAD_CHECKPOINT={load}", "FULL8_RECOVERY_MODE=0",
        )
    )
    return submit([
        "sbatch", "--parsable", "--partition=normal", "--time=12:00:00",
        "--switches=1", f"--exclude={args.train_exclude}", f"--nodes={args.nodes}",
        f"--job-name=full8b_s{segment}a{attempt}",
        f"--output={args.run_root}/logs/%x-%j.out",
        f"--error={args.run_root}/logs/%x-%j.err", f"--export={exports}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"),
    ])


def submit_evaluation_chain(
    args: argparse.Namespace,
    *,
    evaluations: list[int],
    source_train_job: str,
    next_train_job: str,
    next_segment: int | None,
    next_segment_start: int | None,
) -> str:
    if not evaluations:
        raise ValueError(f"segment {args.segment_id} has no GreekMMLU milestones")
    recipe = read_json(args.recipe)
    per_document = {
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
    } - {0}
    # Per-document milestones are split by the evaluator into independent
    # one-node jobs so each request respects the 90 total node-minute QOS cap.
    nodes = 1
    time_limit = "01:25:00"
    exports = common_export(args) + "," + ",".join(
        (
            f"FULL8_EVALUATION_ITERATIONS={encode_evaluation_iterations(evaluations)}",
            "FULL8_EVAL_INDEX=0",
            f"FULL8_SOURCE_SEGMENT_ID={args.segment_id}",
            f"FULL8_SOURCE_TRAIN_JOB_ID={source_train_job}",
            f"FULL8_NEXT_TRAIN_JOB_ID={next_train_job}",
            f"FULL8_NEXT_SEGMENT_ID={'' if next_segment is None else next_segment}",
            f"FULL8_NEXT_SEGMENT_START={'' if next_segment_start is None else next_segment_start}",
        )
    )
    return submit([
        "sbatch", "--parsable", "--partition=debug", f"--time={time_limit}",
        f"--nodes={nodes}", "--ntasks-per-node=1",
        f"--job-name=full8b_eval_{evaluations[0]}",
        f"--output={args.run_root}/logs/%x-%j.out",
        f"--error={args.run_root}/logs/%x-%j.err", f"--export={exports}",
        str(args.ops_root / "clariden/run_checkpoint_evaluation_debug.sbatch"),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-bundle-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-megatron", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--prelaunch-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--train-leaf-switch", required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--ops-bundle-receipt", type=Path, required=True)
    parser.add_argument("--prequeued-manifest", type=Path)
    parser.add_argument("--prequeue-schedule", type=Path)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--attempt-start", type=int, required=True)
    parser.add_argument("--train-job-id", required=True)
    args = parser.parse_args()
    for name in ("code_root", "code_bundle_receipt", "stage_root", "run_root", "initial_megatron", "selected_profile", "launch_gate", "prelaunch_root", "recipe", "profiles", "ops_root", "ops_bundle_receipt"):
        setattr(args, name, getattr(args, name).resolve())
    if args.prequeued_manifest is not None:
        args.prequeued_manifest = args.prequeued_manifest.resolve()
    if args.prequeue_schedule is not None:
        args.prequeue_schedule = args.prequeue_schedule.resolve()
    selected = read_json(args.selected_profile)
    args.profile_id = selected["selection"]["profile_id"]
    args.boundaries = [int(value) for value in selected["selection"]["segment_boundaries"]]
    args.nodes = int(selected["selection"]["nodes"])
    args.train_exclude = subprocess.run(
        [
            str(args.code_root / "subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh"),
            args.train_leaf_switch,
            str(args.nodes),
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not args.train_exclude:
        raise ValueError("empty leaf-switch exclusion")
    if (
        Path(selected.get("recipe", {}).get("path", "")).resolve() != args.recipe
        or Path(selected.get("profiles", {}).get("path", "")).resolve() != args.profiles
    ):
        raise ValueError("selected-profile contract binding drift")
    if not 0 <= args.segment_id < len(args.boundaries) - 1:
        raise ValueError("segment id outside selected profile")
    original_start = args.boundaries[args.segment_id]
    end = args.boundaries[args.segment_id + 1]
    state = slurm_state(args.train_job_id)
    event(args, "train_terminal", {"job_id": args.train_job_id, "slurm_state": state, "updates": [args.attempt_start, end]})
    attempt_root = args.run_root / "segments" / f"segment_{args.segment_id}" / f"attempt_{args.attempt}_updates_{args.attempt_start}_{end}"
    log = attempt_root / "training.log"
    graceful_marker = attempt_root / "graceful_stop_requested"
    latest = last_logged_iteration(log)
    incomplete_graceful = graceful_marker.is_file() and latest is not None and latest < end
    if state != "COMPLETED" or incomplete_graceful:
        recovery_start = original_start
        recovery = False
        recovery_receipt = args.run_root / "recovery_receipts" / f"segment_{args.segment_id}_attempt_{args.attempt}.json"
        if log.is_file():
            command = [
                sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/train/freeze_recovery_checkpoint.py"),
                "--checkpoint-root", str(args.run_root / "checkpoints"), "--log", str(log),
                "--segment-start", str(original_start), "--segment-end", str(end), "--output", str(recovery_receipt),
            ]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode == 0:
                recovery_start = int(read_json(recovery_receipt)["iteration"])
                recovery = True
        if (
            not retryable_terminal(
                state, graceful_marker, latest, end,
                verified_recovery_checkpoint=recovery,
            )
            or args.attempt >= 2
        ):
            event(args, "campaign_blocked", {
                "reason": "nonretryable_or_exhausted_training_failure",
                "slurm_state": state,
                "verified_recovery_checkpoint": recovery,
                "recovery_receipt": str(recovery_receipt) if recovery else None,
            })
            raise RuntimeError(f"training failure is not automatically recoverable: {state}")
        load_override = None
        if not recovery and recovery_start > 0:
            load_override = exact_load_view(args, segment=args.segment_id, attempt=args.attempt + 1, iteration=recovery_start)
        cancelled = cancel_prequeued_suffix(args)
        train, supervisor = submit_attempt(args, segment=args.segment_id, attempt=args.attempt + 1, start=recovery_start, recovery=recovery, load_override=load_override)
        routing_receipt = audit_submitted_jobs(
            args, label=f"recovery_s{args.segment_id}_a{args.attempt + 1}",
            jobs=[("train", train), ("supervisor", supervisor)],
        )
        event(args, "training_retry_submitted", {"replacement_train_job": train, "replacement_supervisor_job": supervisor, "recovery_start": recovery_start, "recovery_receipt": str(recovery_receipt) if recovery else None, "graceful_stop": incomplete_graceful, "last_logged_iteration": latest, "cancelled_prequeued_suffix_jobs": cancelled, "allocation_routing_receipt": str(routing_receipt)})
        return 0

    checkpoint_receipt = args.run_root / "checkpoint_receipts" / f"iter_{end:07d}.json"
    try:
        audit_receipt = args.run_root / "training_audits" / f"segment_{args.segment_id}_attempt_{args.attempt}.json"
        audit_command = [
            sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/train/audit_training_attempt.py"),
            "--log", str(log), "--validation-manifest", str(args.stage_root / "validation/validation_manifest.json"),
            "--recipe", str(args.recipe), "--start", str(args.attempt_start),
            "--end", str(end), "--output", str(audit_receipt),
        ]
        run_or_verify_immutable_receipt(audit_command, audit_receipt)
        subprocess.run([
            sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/train/freeze_checkpoint.py"),
            "--checkpoint-root", str(args.run_root / "checkpoints"), "--iteration", str(end),
            "--recipe", str(args.recipe),
            "--selected-profile", str(args.selected_profile),
            "--schedule-manifest", str(args.stage_root / "schedules/schedule_manifest.json"),
            "--output", str(checkpoint_receipt),
        ], check=True)
    except Exception:
        event(args, "campaign_blocked", {"reason": "checkpoint_or_scientific_gate_failed", "checkpoint_iteration": end})
        raise
    recipe = read_json(args.recipe)
    evaluations = [int(value) for value in recipe["evaluation"]["greekmmlu"]["checkpoint_updates"] if original_start < int(value) <= end]
    next_train = ""
    next_permit = None
    next_segment = None
    next_start = None
    if args.segment_id + 1 < len(args.boundaries) - 1:
        next_segment = args.segment_id + 1
        next_start = end
        adopted = adopt_prequeued_train(
            args,
            source_segment=args.segment_id,
            target_segment=next_segment,
            source_train_job=args.train_job_id,
            start=next_start,
            end=args.boundaries[next_segment + 1],
        )
        if adopted is None:
            next_train = submit_train_only(
                args, segment=next_segment, attempt=0, start=next_start
            )
        else:
            next_train, next_permit = adopted
    evaluation_job = submit_evaluation_chain(
        args,
        evaluations=evaluations,
        source_train_job=args.train_job_id,
        next_train_job=next_train,
        next_segment=next_segment,
        next_segment_start=next_start,
    )
    jobs = [("evaluation", evaluation_job)]
    if next_train:
        jobs.insert(0, ("train", next_train))
    routing_receipt = audit_submitted_jobs(
        args, label=f"segment_{args.segment_id}_success", jobs=jobs
    )
    if next_permit is not None:
        atomic_write_json(next_permit, {
            "schema_version": "apertus_full_8b_prequeued_train_permit_v1",
            "status": "passed",
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_segment_id": args.segment_id,
            "source_train_job": args.train_job_id,
            "source_checkpoint_receipt": str(checkpoint_receipt.resolve()),
            "prequeued_train_job": next_train,
            "target_segment_id": next_segment,
            "start_iteration": next_start,
            "end_iteration": args.boundaries[next_segment + 1],
            "allocation_routing_receipt": str(routing_receipt.resolve()),
        })
    event(args, "segment_checkpoint_gated", {
        "checkpoint_iteration": end,
        "checkpoint_receipt": str(checkpoint_receipt),
        "evaluation_chain_job": evaluation_job,
        "evaluation_iterations": evaluations,
        "next_train_job": next_train or None,
        "next_supervisor_submission": "deferred_to_final_evaluation_job",
        "allocation_routing_receipt": str(routing_receipt),
    })
    if next_train:
        event(args, "next_segment_submitted", {
            "next_segment": next_segment,
            "train_job": next_train,
            "supervisor_job": None,
            "updates": [next_start, args.boundaries[next_segment + 1]],
            "supervisor_policy": "last_segment_evaluation_submits_afterany_supervisor",
            "prequeued": next_permit is not None,
            "launch_permit": str(next_permit) if next_permit is not None else None,
        })
    else:
        event(args, "final_evaluation_chain_submitted", {
            "evaluation_chain_job": evaluation_job,
            "final_iteration": end,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

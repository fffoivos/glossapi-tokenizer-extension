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
import time
from pathlib import Path

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


def allow_nested_sbatch(command: list[str]) -> list[str]:
    """Make sbatch legal when this supervisor itself runs inside uenv."""
    if not command or Path(command[0]).name != "sbatch":
        raise ValueError("supervisor submit() only accepts sbatch commands")
    if "--uenv-passthrough=ignore" in command:
        return list(command)
    return [command[0], "--uenv-passthrough=ignore", *command[1:]]


def submit(command: list[str]) -> str:
    output = subprocess.run(
        allow_nested_sbatch(command), text=True, capture_output=True, check=True
    ).stdout.strip()
    return output.split(";", 1)[0]


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
    return "ALL," + ",".join(
        (
            f"FULL8_CODE_ROOT={args.code_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={args.code_bundle_receipt}",
            f"FULL8_STAGE_ROOT={args.stage_root}",
            f"FULL8_RUN_ROOT={args.run_root}",
            f"FULL8_INITIAL_MEGATRON={args.initial_megatron}",
            f"FULL8_SELECTED_PROFILE={args.selected_profile}",
            f"FULL8_EXECUTION_PROFILE={args.profile_id}",
            f"FULL8_LAUNCH_GATE={args.launch_gate}",
            f"FULL8_PRELAUNCH_ROOT={args.prelaunch_root}",
        )
    )


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
        "sbatch", "--parsable", f"--nodes={args.nodes}", f"--job-name=full8b_s{segment}a{attempt}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err", f"--export={exports}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"),
    ])
    supervisor = submit([
        "sbatch", "--parsable", f"--dependency=afterany:{train}", f"--job-name=full8b_supervise_s{segment}a{attempt}",
        f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err", f"--export={common_export(args)}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/supervise_campaign.sbatch"),
        str(segment), str(attempt), str(start), train,
    ])
    return train, supervisor


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
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--attempt-start", type=int, required=True)
    parser.add_argument("--train-job-id", required=True)
    args = parser.parse_args()
    for name in ("code_root", "code_bundle_receipt", "stage_root", "run_root", "initial_megatron", "selected_profile", "launch_gate", "prelaunch_root"):
        setattr(args, name, getattr(args, name).resolve())
    selected = read_json(args.selected_profile)
    args.profile_id = selected["selection"]["profile_id"]
    args.boundaries = [int(value) for value in selected["selection"]["segment_boundaries"]]
    args.nodes = int(selected["selection"]["nodes"])
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
        train, supervisor = submit_attempt(args, segment=args.segment_id, attempt=args.attempt + 1, start=recovery_start, recovery=recovery, load_override=load_override)
        event(args, "training_retry_submitted", {"replacement_train_job": train, "replacement_supervisor_job": supervisor, "recovery_start": recovery_start, "recovery_receipt": str(recovery_receipt) if recovery else None, "graceful_stop": incomplete_graceful, "last_logged_iteration": latest})
        return 0

    checkpoint_receipt = args.run_root / "checkpoint_receipts" / f"iter_{end:07d}.json"
    try:
        audit_receipt = args.run_root / "training_audits" / f"segment_{args.segment_id}_attempt_{args.attempt}.json"
        subprocess.run([
            sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/train/audit_training_attempt.py"),
            "--log", str(log), "--validation-manifest", str(args.stage_root / "validation/validation_manifest.json"),
            "--start", str(args.attempt_start), "--end", str(end), "--output", str(audit_receipt),
        ], check=True)
        subprocess.run([
            sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/train/freeze_checkpoint.py"),
            "--checkpoint-root", str(args.run_root / "checkpoints"), "--iteration", str(end),
            "--recipe", str(args.code_root / "subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json"),
            "--selected-profile", str(args.selected_profile),
            "--schedule-manifest", str(args.stage_root / "schedules/schedule_manifest.json"),
            "--output", str(checkpoint_receipt),
        ], check=True)
    except Exception:
        event(args, "campaign_blocked", {"reason": "checkpoint_or_scientific_gate_failed", "checkpoint_iteration": end})
        raise
    recipe = read_json(args.code_root / "subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json")
    evaluations = [int(value) for value in recipe["evaluation"]["greekmmlu"]["checkpoint_updates"] if original_start < int(value) <= end]
    queue_receipt = args.run_root / "evaluation_queues" / f"segment_{args.segment_id}.json"
    queue = submit([
        "sbatch", "--parsable", f"--job-name=full8b_evalq_s{args.segment_id}",
        f"--output={args.run_root}/logs/%x-%j.out", f"--error={args.run_root}/logs/%x-%j.err",
        f"--export={common_export(args)},FULL8_TRAIN_JOB_ID={args.train_job_id},FULL8_EVALUATION_ITERATIONS={encode_evaluation_iterations(evaluations)},FULL8_EVALUATION_QUEUE_RECEIPT={queue_receipt}",
        str(args.code_root / "subprojects/07_full_8b_cpt/clariden/run_evaluation_queue.sbatch"),
    ])
    event(args, "segment_checkpoint_gated", {"checkpoint_iteration": end, "checkpoint_receipt": str(checkpoint_receipt), "evaluation_queue_job": queue, "evaluation_iterations": evaluations})
    if args.segment_id + 1 < len(args.boundaries) - 1:
        train, supervisor = submit_attempt(args, segment=args.segment_id + 1, attempt=0, start=end, recovery=False)
        event(args, "next_segment_submitted", {"next_segment": args.segment_id + 1, "train_job": train, "supervisor_job": supervisor, "updates": [end, args.boundaries[args.segment_id + 2]]})
    else:
        training_receipt = args.run_root / "training_completion_receipt.json"
        subprocess.run([
            sys.executable, str(args.code_root / "subprojects/07_full_8b_cpt/scripts/finalize_training.py"),
            "--run-root", str(args.run_root), "--selected-profile", str(args.selected_profile),
            "--launch-gate", str(args.launch_gate), "--output", str(training_receipt),
        ], check=True)
        finalizer = submit([
            "sbatch", "--parsable", "--job-name=full8b_evidence_finalizer",
            f"--output={args.run_root}/logs/%x-%j.out", f"--error={args.run_root}/logs/%x-%j.err",
            f"--export={common_export(args)},FULL8_TRAINING_RECEIPT={training_receipt}",
            str(args.code_root / "subprojects/07_full_8b_cpt/clariden/finalize_campaign.sbatch"),
        ])
        event(args, "training_completed", {"training_completion_receipt": str(training_receipt), "evidence_finalizer_job": finalizer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

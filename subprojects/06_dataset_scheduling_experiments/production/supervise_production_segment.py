#!/usr/bin/env python3
"""Monitor one five-arm segment, retry infra failures, and advance the campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from campaign_contract import (
    AUTHORITATIVE_EVALUATION_DTYPE,
    atomic_write_json,
    evaluation_namespace,
    read_json,
    resolve_evaluation_runtime,
    verify_code_bundle_receipt,
)


TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
RETRYABLE_INFRA = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED", "TIMEOUT"}
MAX_ATTEMPT = 2


def slurm_status(job_id: str) -> dict[str, str]:
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State,ExitCode,Reason"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {"state": "UNKNOWN", "exit_code": "", "reason": result.stderr.strip()}
    state, exit_code, reason = (rows[0] + ["", "", ""])[:3]
    normalized_state = state.split()[0].split("+")[0] if state.strip() else "UNKNOWN"
    return {"state": normalized_state, "exit_code": exit_code, "reason": reason}


def wait_terminal(job_id: str, *, poll_seconds: int = 60) -> dict[str, str]:
    unknown_since: float | None = None
    while True:
        status = slurm_status(job_id)
        if status["state"] in TERMINAL:
            return status
        if status["state"] == "UNKNOWN":
            unknown_since = unknown_since or time.monotonic()
            if time.monotonic() - unknown_since > 600:
                raise RuntimeError(f"Slurm lost job {job_id}: {status['reason']}")
        else:
            unknown_since = None
        time.sleep(poll_seconds)


def submit(command: list[str]) -> str:
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    return output.split(";", 1)[0]


def common_exports(args: argparse.Namespace, *, evaluation: bool = False) -> str:
    bundle = args.evaluation_bundle if evaluation else args.scientific_bundle
    fields = [
        f"CAMPAIGN_MANIFEST={args.campaign_manifest}",
        f"RUN_ROOT={args.run_root}",
        f"SCIENTIFIC_BUNDLE={bundle}",
        f"RUN_TAG={args.run_tag}",
    ]
    if evaluation:
        fields.extend(
            (
                f"EVALUATION_NAMESPACE={args.evaluation_namespace}",
                f"EVAL_DTYPE={AUTHORITATIVE_EVALUATION_DTYPE}",
            )
        )
        if args.evaluation_bundle != args.scientific_bundle:
            fields.extend(
                (
                    f"EVALUATION_MEGATRON_DIR={args.evaluation_megatron}",
                    "OPERATIONAL_RECOVERY_RECEIPT="
                    f"{args.operational_recovery_receipt}",
                )
            )
    elif args.training_scientific_bundle_receipt:
        fields.append(
            "RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT="
            f"{args.training_scientific_bundle_receipt}"
        )
    return "ALL," + ",".join(fields)


def record_event(args: argparse.Namespace, event: str, payload: dict) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    row = {
        "schema_version": "apertus_mini_campaign_event_v1",
        "recorded_at": now.isoformat(),
        "event": event,
        "segment_id": args.segment_id,
        "segment_attempt": args.segment_attempt,
        **payload,
    }
    root = args.run_root / "orchestration" / "events"
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{event}_s{args.segment_id}_a{args.segment_attempt}.json"
    atomic_write_json(root / name, row)
    atomic_write_json(args.run_root / "orchestration" / "latest.json", row, exclusive=False)


def submit_attempt(
    args: argparse.Namespace,
    *,
    segment_id: int,
    attempt: int,
    resume_receipt: Path | None,
    recovery_start: int | None,
) -> dict[str, str]:
    training_common = common_exports(args)
    evaluation_common = common_exports(args, evaluation=True)
    extras = f",SEGMENT_ID={segment_id},SEGMENT_ATTEMPT={attempt}"
    if resume_receipt is not None:
        extras += f",RESUME_RECEIPT={resume_receipt}"
    if recovery_start is not None:
        extras += f",RECOVERY_START={recovery_start},RECOVERY_RECEIPT={resume_receipt}"
    logs = args.run_root / "logs"
    train = submit(
        [
            "sbatch", "--parsable", f"--job-name={args.run_tag}_s{segment_id}a{attempt}",
            f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
            f"--export={training_common}{extras}",
            str(args.scientific_bundle / "clariden" / "train_five_arm_segment.sbatch"),
        ]
    )
    watch = submit(
        [
            "sbatch", "--parsable", f"--dependency=after:{train}",
            f"--job-name={args.run_tag}_watch{segment_id}a{attempt}",
            f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
            f"--export={evaluation_common}{extras},TRAIN_JOB_ID={train}",
            str(args.evaluation_bundle / "clariden" / "watch_checkpoint_evaluations.sbatch"),
        ]
    )
    supervisor = submit(
        [
            "sbatch", "--parsable", f"--dependency=after:{train}",
            f"--job-name={args.run_tag}_supervise{segment_id}a{attempt}",
            f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
            f"--export={evaluation_common}{extras},TRAIN_JOB_ID={train},WATCH_JOB_ID={watch}",
            str(args.evaluation_bundle / "clariden" / "supervise_production_segment.sbatch"),
        ]
    )
    return {"train": train, "watch": watch, "supervisor": supervisor}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--segment-attempt", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--train-job-id", required=True)
    parser.add_argument("--watch-job-id", required=True)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--recovery-start", type=int)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    args.campaign_manifest = args.campaign_manifest.resolve()
    args.run_root = args.run_root.resolve()
    campaign = read_json(args.campaign_manifest)
    canonical_scientific_bundle = Path(campaign["assets"]["scientific_bundle"])
    training_bundle_override = os.environ.get("TRAINING_SCIENTIFIC_BUNDLE", "")
    training_bundle_receipt = os.environ.get(
        "TRAINING_SCIENTIFIC_BUNDLE_RECEIPT", ""
    )
    if bool(training_bundle_override) != bool(training_bundle_receipt):
        raise ValueError(
            "training scientific bundle override and receipt must be supplied together"
        )
    if training_bundle_override:
        args.scientific_bundle = Path(training_bundle_override).resolve()
        args.training_scientific_bundle_receipt = Path(
            training_bundle_receipt
        ).resolve()
        verify_code_bundle_receipt(
            args.training_scientific_bundle_receipt,
            args.scientific_bundle,
            "scientific",
        )
    else:
        args.scientific_bundle = canonical_scientific_bundle
        args.training_scientific_bundle_receipt = None
    args.evaluation_bundle, args.evaluation_megatron = resolve_evaluation_runtime(
        args.campaign_manifest, campaign
    )
    args.operational_recovery_receipt = os.environ.get(
        "OPERATIONAL_RECOVERY_RECEIPT", ""
    )
    args.evaluation_namespace = evaluation_namespace()
    controller_bundle = os.environ.get("RECOVERY_CONTROLLER_BUNDLE", "")
    controller_receipt = os.environ.get("RECOVERY_CONTROLLER_BUNDLE_RECEIPT", "")
    if bool(controller_bundle) != bool(controller_receipt):
        raise ValueError("recovery controller bundle and receipt must be supplied together")
    if controller_bundle:
        args.controller_bundle = Path(controller_bundle).resolve()
        args.controller_bundle_receipt = Path(controller_receipt).resolve()
        verify_code_bundle_receipt(
            args.controller_bundle_receipt,
            args.controller_bundle,
            "scientific",
        )
    else:
        args.controller_bundle = args.evaluation_bundle
        args.controller_bundle_receipt = None
    boundary = os.environ.get("SOURCE_VALIDATION_ATTEMPT_BOUNDARY", "")
    args.source_validation_attempt_boundary = int(boundary) if boundary else None
    authority = os.environ.get("SOURCE_VALIDATION_ATTEMPT_AUTHORITY", "").strip()
    args.source_validation_attempt_authority = (
        [item for item in re.split(r"[;,]", authority) if item]
        if authority
        else []
    )
    if args.source_validation_attempt_boundary is not None and args.source_validation_attempt_authority:
        raise ValueError("legacy and multi-stage source authority are exclusive")
    has_source_authority = (
        args.source_validation_attempt_boundary is not None
        or bool(args.source_validation_attempt_authority)
    )
    if has_source_authority != (args.segment_attempt > 0):
        raise ValueError(
            "recovered segment attempts require explicit source-validation authority"
        )
    return args


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    train = wait_terminal(args.train_job_id, poll_seconds=args.poll_seconds)
    record_event(args, "train_terminal", {"job_id": args.train_job_id, **train})

    if train["state"] != "COMPLETED":
        if train["state"] not in RETRYABLE_INFRA or args.segment_attempt >= MAX_ATTEMPT:
            record_event(args, "campaign_blocked", {"reason": "nonretryable_or_exhausted_training_failure"})
            raise RuntimeError(f"training stopped as {train}; automatic retry is not authorized")
        watch = slurm_status(args.watch_job_id)
        if watch["state"] not in TERMINAL:
            subprocess.run(["scancel", args.watch_job_id], check=False)
            wait_terminal(args.watch_job_id, poll_seconds=args.poll_seconds)

        current_start = (
            int(args.recovery_start)
            if args.recovery_start is not None
            else int(read_json(args.campaign_manifest)["segments"][args.segment_id]["start_iteration"])
        )
        recovery = args.run_root / "receipts" / (
            f"segment_{args.segment_id}_attempt_{args.segment_attempt}_recovery.json"
        )
        command = [
            sys.executable,
            str(args.evaluation_bundle / "production" / "freeze_common_recovery_checkpoint.py"),
            "--campaign-manifest", str(args.campaign_manifest), "--segment-id", str(args.segment_id),
            "--run-root", str(args.run_root), "--segment-attempt", str(args.segment_attempt),
            "--start-iteration", str(current_start), "--output", str(recovery),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            recovery_start = int(read_json(recovery)["iteration"])
            resume = recovery
        else:
            recovery_start = args.recovery_start
            resume = args.resume_receipt
            if current_start > 0 and resume is None:
                record_event(args, "campaign_blocked", {"reason": "no_common_checkpoint_or_prior_receipt"})
                raise RuntimeError("failed attempt has no common checkpoint and no prior receipt")
        jobs = submit_attempt(
            args,
            segment_id=args.segment_id,
            attempt=args.segment_attempt + 1,
            resume_receipt=resume,
            recovery_start=recovery_start,
        )
        record_event(
            args,
            "infra_retry_submitted",
            {"recovery_start": recovery_start, "resume_receipt": str(resume) if resume else None, "jobs": jobs},
        )
        return 0

    watch = wait_terminal(args.watch_job_id, poll_seconds=args.poll_seconds)
    record_event(args, "watch_terminal", {"job_id": args.watch_job_id, **watch})
    if watch["state"] != "COMPLETED":
        record_event(args, "campaign_blocked", {"reason": "evaluation_watcher_failure"})
        raise RuntimeError(f"checkpoint evaluation watcher stopped as {watch}")

    receipt = args.run_root / "receipts" / f"segment_{args.segment_id}_checkpoint.json"
    source_validation = (
        args.run_root
        / "receipts"
        / f"segment_{args.segment_id}_source_validation.json"
    )
    gate = args.run_root / "receipts" / f"segment_{args.segment_id}_gate.json"
    run_checked(
        [
            sys.executable,
            str(args.evaluation_bundle / "production" / "freeze_segment_checkpoint.py"),
            "--campaign-manifest", str(args.campaign_manifest), "--segment-id", str(args.segment_id),
            "--run-root", str(args.run_root), "--segment-attempt", str(args.segment_attempt),
            "--output", str(receipt),
        ]
    )
    source_audit_command = [
            sys.executable,
            str(
                args.controller_bundle
                / "evaluation"
                / "audit_segment_source_validation.py"
            ),
            "--campaign-manifest", str(args.campaign_manifest),
            "--segment-id", str(args.segment_id),
            "--run-root", str(args.run_root),
            "--initial-validation-receipt",
            str(args.run_root / "initial_validation" / "initial_validation_receipt.json"),
            "--output", str(source_validation),
    ]
    if args.source_validation_attempt_boundary is not None:
        source_audit_command.extend(
            [
                "--authoritative-attempt-boundary",
                str(args.source_validation_attempt_boundary),
                "--post-boundary-attempt",
                str(args.segment_attempt),
            ]
        )
    elif args.source_validation_attempt_authority:
        for item in args.source_validation_attempt_authority:
            source_audit_command.extend(["--attempt-authority-through", item])
    run_checked(source_audit_command)
    run_checked(
        [
            sys.executable,
            str(args.evaluation_bundle / "production" / "gate_segment.py"),
            "--campaign-manifest", str(args.campaign_manifest), "--segment-id", str(args.segment_id),
            "--run-root", str(args.run_root), "--checkpoint-receipt", str(receipt),
            "--source-validation-receipt", str(source_validation),
            "--output", str(gate),
        ]
    )
    record_event(
        args,
        "segment_passed",
        {
            "checkpoint_receipt": str(receipt),
            "source_validation_receipt": str(source_validation),
            "gate_receipt": str(gate),
            "evaluation_namespace": args.evaluation_namespace,
            "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
        },
    )

    if args.segment_id == 0:
        jobs = submit_attempt(
            args, segment_id=1, attempt=0, resume_receipt=receipt, recovery_start=None
        )
        record_event(args, "next_segment_submitted", {"jobs": jobs})
    else:
        training_complete = {
            "schema_version": "apertus_mini_training_completion_v1",
            "status": "completed",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "campaign_manifest": str(args.campaign_manifest),
            "segment_gates": [
                str(args.run_root / "receipts" / "segment_0_gate.json"),
                str(args.run_root / "receipts" / "segment_1_gate.json"),
            ],
            "greekmmlu_bindings": 415,
            "checkpoint_averaging": False,
        }
        atomic_write_json(args.run_root / "training_completion_receipt.json", training_complete)
        common = common_exports(args, evaluation=True)
        logs = args.run_root / "logs"
        full_validation = submit(
            [
                "sbatch", "--parsable", f"--job-name={args.run_tag}_fullval",
                f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
                f"--export={common},CHECKPOINT_RECEIPT={receipt}",
                str(args.evaluation_bundle / "clariden" / "run_full_endpoint_validation.sbatch"),
            ]
        )
        greek_endpoints = submit(
            [
                "sbatch", "--parsable", f"--job-name={args.run_tag}_greekep",
                f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
                f"--export={common}",
                str(args.evaluation_bundle / "clariden" / "run_greek_endpoint_wave.sbatch"),
            ]
        )
        retention = submit(
            [
                "sbatch", "--parsable", f"--job-name={args.run_tag}_retention",
                f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err",
                f"--export={common}",
                str(args.evaluation_bundle / "clariden" / "run_retention_endpoint_wave.sbatch"),
            ]
        )
        core = submit(
            [
                "sbatch", "--parsable", f"--dependency=afterok:{full_validation}",
                f"--job-name={args.run_tag}_core", f"--output={logs}/%x-%j.out",
                f"--error={logs}/%x-%j.err", f"--export={common}",
                str(args.evaluation_bundle / "clariden" / "finalize_core_campaign_evidence.sbatch"),
            ]
        )
        completion = submit(
            [
                "sbatch", "--parsable",
                f"--dependency=afterok:{core}:{greek_endpoints}:{retention}",
                f"--job-name={args.run_tag}_complete", f"--output={logs}/%x-%j.out",
                f"--error={logs}/%x-%j.err", f"--export={common}",
                str(args.evaluation_bundle / "clariden" / "finalize_campaign_evidence.sbatch"),
            ]
        )
        record_event(
            args,
            "training_complete_post_evaluation_submitted",
            {
                "completion_receipt": str(args.run_root / "training_completion_receipt.json"),
                "jobs": {
                    "full_endpoint_validation": full_validation,
                    "greek_endpoints": greek_endpoints,
                    "retention": retention,
                    "core_evidence": core,
                    "completion": completion,
                },
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

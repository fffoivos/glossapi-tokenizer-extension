#!/usr/bin/env python3
"""Idempotently submit and monitor exact-checkpoint GreekMMLU waves."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign_contract import (
    ARMS,
    AUTHORITATIVE_EVALUATION_DTYPE,
    atomic_write_json,
    evaluation_namespace,
    padded_iteration,
    read_json,
    resolve_evaluation_runtime,
    scoped_evaluation_root,
    sha256_file,
)


TERMINAL_FAILURES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}
DEFAULT_MAX_ATTEMPTS = 3


def evaluation_attempt_limits() -> dict[int, int]:
    text = os.environ.get("EVALUATION_ATTEMPT_LIMIT_OVERRIDES", "").strip()
    if not text:
        return {}
    limits: dict[int, int] = {}
    for item in text.split(","):
        fields = item.split(":", 1)
        if len(fields) != 2:
            raise ValueError(f"invalid evaluation attempt-limit override: {item!r}")
        iteration, limit = map(int, fields)
        if iteration < 0 or not DEFAULT_MAX_ATTEMPTS < limit <= 5:
            raise ValueError(f"unsafe evaluation attempt-limit override: {item!r}")
        if iteration in limits:
            raise ValueError(f"duplicate evaluation attempt-limit override: {iteration}")
        limits[iteration] = limit
    receipt_text = os.environ.get("OPERATIONAL_RECOVERY_RECEIPT", "")
    if not receipt_text:
        raise ValueError("evaluation attempt override lacks a recovery receipt")
    receipt = read_json(Path(receipt_text))
    declared = {
        int(iteration): int(limit)
        for iteration, limit in receipt.get(
            "evaluation_attempt_limit_overrides", {}
        ).items()
    }
    if declared != limits:
        raise ValueError("evaluation attempt-limit override receipt drift")
    return limits


def slurm_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-o", "State"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = [
        line.strip().split()[0].split("+")[0]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return rows[0] if rows else "UNKNOWN"


def receipt_complete(
    path: Path, arm: str, iteration: int, namespace: str
) -> bool:
    state_path = path.parent / "pipeline_state.json"
    if not path.is_file() or not state_path.is_file():
        return False
    try:
        value = read_json(path)
        pipeline = read_json(state_path)
        checkpoint = value.get("checkpoint", {})
        metrics = value.get("metrics", {})
        clean = metrics.get("decontaminated", {})
        export_path = Path(checkpoint["export_receipt_path"])
        headline_values = (
            metrics.get("accuracy"),
            metrics.get("choice_nll"),
            metrics.get("correct_answer_bpb"),
            clean.get("accuracy"),
            clean.get("choice_nll"),
            clean.get("correct_answer_bpb"),
        )
    except Exception:
        return False
    return (
        value.get("schema_version") == "exact_checkpoint_native_greekmmlu_receipt_v1"
        and value.get("status") == "completed"
        and value.get("evaluation_namespace") == namespace
        and value.get("evaluator", {}).get("dtype")
        == AUTHORITATIVE_EVALUATION_DTYPE
        and int(checkpoint.get("iteration", -1)) == iteration
        and int(metrics.get("n", -1)) == 16_632
        and int(clean.get("n", 0)) > 0
        and all(
            isinstance(metric, (int, float)) and math.isfinite(metric)
            for metric in headline_values
        )
        and export_path.is_file()
        and sha256_file(export_path) == checkpoint.get("export_receipt_sha256")
        and pipeline.get("schema_version")
        == "apertus_mini_checkpoint_native_greekmmlu_pipeline_state_v1"
        and pipeline.get("status") == "complete"
        and pipeline.get("arm_id") == arm
        and int(pipeline.get("source_iteration", -1)) == iteration
        and Path(pipeline.get("receipt", "")).resolve() == path.resolve()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-job-id", required=True)
    parser.add_argument("--segment-attempt", type=int, default=0)
    parser.add_argument("--recovery-start", type=int)
    parser.add_argument("--recovery-receipt", type=Path)
    parser.add_argument("--sleep-seconds", type=int, default=120)
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=int(os.environ.get("WATCH_MAX_SECONDS", "86_000")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign = read_json(args.campaign_manifest)
    if campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1" or campaign.get("status") != "frozen":
        raise ValueError("campaign manifest is not frozen")
    segment = campaign["segments"][args.segment_id]
    nominal_start, end = int(segment["start_iteration"]), int(segment["end_iteration"])
    start = nominal_start if args.recovery_start is None else int(args.recovery_start)
    if not nominal_start <= start < end:
        raise ValueError(f"invalid recovery start {start} for segment {args.segment_id}")
    iterations = [
        int(value)
        for value in campaign["evaluation"]["checkpoint_iterations"]
        if (nominal_start < int(value) <= end)
        or (args.segment_id == 0 and int(value) == 0)
    ]
    assets = campaign["assets"]
    scientific, evaluation_megatron = resolve_evaluation_runtime(
        args.campaign_manifest, campaign
    )
    namespace = evaluation_namespace()
    attempt_limits = evaluation_attempt_limits()
    checkpoint_plan = Path(assets["checkpoint_plan"]["path"])
    state_root = scoped_evaluation_root(
        args.run_root, "evaluation_watch"
    ) / f"segment_{args.segment_id}"
    manifests_root = scoped_evaluation_root(
        args.run_root, "evaluation_manifests"
    ) / f"segment_{args.segment_id}"
    evaluations_root = scoped_evaluation_root(args.run_root, "evaluations")
    state_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    evaluations_root.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    recovery_roots: dict[str, Path] | None = None
    if start > nominal_start:
        if args.recovery_receipt is None:
            raise ValueError("a recovery watcher requires --recovery-receipt")
        recovery = read_json(args.recovery_receipt)
        rows = {row["arm_id"]: row for row in recovery.get("arms", [])}
        if (
            recovery.get("schema_version") != "apertus_mini_segment_checkpoint_receipt_v1"
            or recovery.get("status") != "passed"
            or int(recovery.get("iteration", -1)) != start
            or tuple(sorted(rows)) != tuple(sorted(ARMS))
        ):
            raise ValueError("recovery watcher receipt drift")
        recovery_roots = {arm: Path(rows[arm]["checkpoint_root"]) for arm in ARMS}

    while True:
        pending = 0
        for iteration in iterations:
            state_path = state_root / f"iteration_{iteration:07d}.json"
            if state_path.is_file():
                state = read_json(state_path)
                if (
                    state.get("evaluation_namespace") != namespace
                    or state.get("authoritative_evaluation_dtype")
                    != AUTHORITATIVE_EVALUATION_DTYPE
                ):
                    raise ValueError(f"evaluation state namespace/dtype drift: {state_path}")
            else:
                state = {
                    "schema_version": "apertus_mini_evaluation_watch_state_v1",
                    "iteration": iteration,
                    "evaluation_namespace": namespace,
                    "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
                    "status": "waiting_for_checkpoint",
                    "attempts": [],
                }
            if state.get("status") == "completed":
                continue
            pending += 1
            if iteration == 0:
                roots = {arm: Path(assets["initial_checkpoint_root"]) for arm in ARMS}
            elif iteration <= start and recovery_roots is not None:
                roots = recovery_roots
            elif iteration <= start:
                # A fresh watcher can only encounter this branch for already completed
                # state. Failing closed here prevents a retry from silently dropping a
                # checkpoint whose source attempt is no longer available.
                raise RuntimeError(
                    f"iteration {iteration} is incomplete but precedes watcher start {start}"
                )
            else:
                roots = {
                    arm: args.run_root / "segments" / f"segment_{args.segment_id}" / f"attempt_{args.segment_attempt}" / arm / "checkpoints"
                    for arm in ARMS
                }
            checkpoint_directory = "release" if iteration == 0 else padded_iteration(iteration)
            ready = all(
                (root / checkpoint_directory / ".metadata").is_file()
                for root in roots.values()
            )
            if not ready:
                continue
            attempts = list(state.get("attempts", []))
            completed_receipts: dict[str, Path] = {}
            for recorded_attempt in attempts:
                recorded_root = Path(recorded_attempt["wave_output_root"])
                for arm in ARMS:
                    candidate = (
                        recorded_root
                        / "tasks"
                        / arm
                        / "exact_checkpoint_native_greekmmlu_receipt.json"
                    )
                    if receipt_complete(candidate, arm, iteration, namespace):
                        completed_receipts[arm] = candidate
            if attempts:
                if set(completed_receipts) == set(ARMS):
                    state["status"] = "completed"
                    state["receipts"] = {
                        arm: str(completed_receipts[arm]) for arm in ARMS
                    }
                    atomic_write_json(state_path, state, exclusive=False)
                    pending -= 1
                    continue
                latest = attempts[-1]
                job_state = slurm_state(str(latest["job_id"]))
                latest["last_observed_state"] = job_state
                if job_state not in TERMINAL_FAILURES | {"COMPLETED"}:
                    atomic_write_json(state_path, state, exclusive=False)
                    continue
                latest["status"] = "failed_or_incomplete"
                attempt_limit = attempt_limits.get(iteration, DEFAULT_MAX_ATTEMPTS)
                if len(attempts) >= attempt_limit:
                    state["status"] = "failed"
                    atomic_write_json(state_path, state, exclusive=False)
                    raise RuntimeError(
                        f"evaluation iteration {iteration} exhausted "
                        f"{attempt_limit} attempts"
                    )

            attempt = len(attempts) + 1
            missing_arms = [arm for arm in ARMS if arm not in completed_receipts]
            wave_manifest = manifests_root / f"iteration_{iteration:07d}_attempt_{attempt}.json"
            wave_root = evaluations_root / f"iteration_{iteration:07d}" / f"attempt_{attempt}"
            command = [
                sys.executable,
                str(scientific / "evaluation" / "build_greekmmlu_wave_manifest.py"),
                "--checkpoint-evaluation-plan",
                str(checkpoint_plan),
                "--output",
                str(wave_manifest),
            ]
            for arm in missing_arms:
                command.extend(
                    [
                        "--task",
                        arm,
                        str(iteration),
                        str(roots[arm]),
                        str(wave_root / "tasks" / arm),
                    ]
                )
            subprocess.run(command, check=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "EVALUATION_BUNDLE": str(scientific),
                    "NATIVE_GREEK_EVAL_ROOT": assets["native_greek_eval_root"],
                    "MEGATRON_DIR": str(evaluation_megatron),
                    "TOKENIZER_DIR": assets["tokenizer_dir"],
                    "PYTHON_COMPAT_DIR": assets["python_compat_dir"],
                    "CHECKPOINT_EVALUATION_PLAN": str(checkpoint_plan),
                    "GREEKMMLU_CLEAN_SUBSET": assets["greekmmlu_clean_subset"],
                    "WAVE_MANIFEST": str(wave_manifest),
                    "WAVE_OUTPUT_ROOT": str(wave_root),
                    "EVALUATION_NAMESPACE": namespace,
                    "EVAL_DTYPE": AUTHORITATIVE_EVALUATION_DTYPE,
                }
            )
            submitted = subprocess.run(
                ["bash", str(scientific / "clariden" / "submit_checkpoint_native_greekmmlu_wave.sh")],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                env=environment,
            ).stdout.strip()
            job_id = submitted.rsplit("=", 1)[-1]
            attempts.append(
                {
                    "attempt": attempt,
                    "job_id": job_id,
                    "wave_manifest": str(wave_manifest),
                    "wave_output_root": str(wave_root),
                    "requested_arms": missing_arms,
                    "status": "submitted",
                }
            )
            state["attempts"] = attempts
            state["status"] = "submitted"
            atomic_write_json(state_path, state, exclusive=False)

        if pending == 0:
            summary = {
                "schema_version": "apertus_mini_evaluation_watch_summary_v1",
                "status": "completed",
                "segment_id": args.segment_id,
                "segment_attempt": args.segment_attempt,
                "evaluation_namespace": namespace,
                "authoritative_evaluation_dtype": AUTHORITATIVE_EVALUATION_DTYPE,
                "recovery_start": start,
                "iterations": iterations,
                "bindings": len(iterations) * len(ARMS),
            }
            atomic_write_json(state_root / "summary.json", summary, exclusive=False)
            print(json.dumps(summary, sort_keys=True))
            return 0
        train_state = slurm_state(args.train_job_id)
        if train_state in TERMINAL_FAILURES:
            raise RuntimeError(
                f"training job {args.train_job_id} ended as {train_state} with {pending} evaluations pending"
            )
        if time.monotonic() - begin >= args.max_seconds:
            raise TimeoutError(f"evaluation watcher timed out with {pending} points pending")
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

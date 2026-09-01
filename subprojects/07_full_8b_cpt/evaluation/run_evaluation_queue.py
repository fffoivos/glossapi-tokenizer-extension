#!/usr/bin/env python3
"""Maintain at most four checkpoint-evaluation pipelines with bounded retries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from contract import atomic_write_json, read_json


ACTIVE = {"CONFIGURING", "COMPLETING", "PENDING", "RUNNING", "RESIZING", "SUSPENDED"}
SUCCESS = {"COMPLETED"}


def decode_evaluation_iterations(value: str) -> list[int]:
    """Accept the colon-safe Slurm transport and legacy comma-separated input."""
    return [int(item) for item in re.split(r"[:,]", value) if item]


def state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job_id, "--format=State"],
        text=True, capture_output=True, check=False,
    )
    rows = [row.split("|")[0].split()[0].split("+")[0] for row in result.stdout.splitlines() if row.strip()]
    return rows[0] if rows else "UNKNOWN"


def attempt_dirs(iteration_root: Path) -> list[Path]:
    return sorted(iteration_root.glob("attempt_[0-2]"), key=lambda path: int(path.name.split("_")[-1]))


def attempt_complete(path: Path, needs_doc: bool) -> bool:
    receipt = path / "exact_checkpoint_native_greekmmlu_receipt.json"
    if not receipt.is_file():
        return False
    value = read_json(receipt)
    if str(value.get("status", "")).lower() not in {"completed", "passed"}:
        return False
    if needs_doc:
        receipts = list((path / "per_document").glob("*.receipt.json"))
        if len(receipts) != 13:
            return False
        for item in receipts:
            row = read_json(item)
            if row.get("status") != "completed" or int(row.get("aggregate", {}).get("documents", 0)) <= 0:
                return False
    return True


def submit_pipeline(args: argparse.Namespace, iteration: int, attempt: int) -> None:
    env = os.environ.copy()
    env.update(
        FULL8_CODE_ROOT=str(args.code_root),
        FULL8_RUN_ROOT=str(args.run_root),
        FULL8_STAGE_ROOT=str(args.stage_root),
        FULL8_ITERATION=str(iteration),
        FULL8_DEPENDENCY=f"afterok:{args.train_job_id}",
        FULL8_EVAL_ATTEMPT=str(attempt),
        DRY_RUN="0",
        FULL8_RECIPE=str(args.recipe),
    )
    subprocess.run(
        [str(args.code_root / "subprojects/07_full_8b_cpt/evaluation/submit_greekmmlu_checkpoint.sh")],
        env=env, check=True,
    )


def canonicalize(iteration_root: Path, attempt: Path) -> None:
    value = {
        "schema_version": "apertus_full_8b_authoritative_checkpoint_evaluation_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": int(iteration_root.name.split("_")[-1]),
        "attempt": int(attempt.name.split("_")[-1]),
        "attempt_root": str(attempt.resolve()),
        "greekmmlu_receipt": str((attempt / "exact_checkpoint_native_greekmmlu_receipt.json").resolve()),
        "per_document_root": str((attempt / "per_document").resolve()) if (attempt / "per_document").exists() else None,
    }
    canonical = iteration_root / "authoritative_attempt.json"
    if not canonical.exists():
        atomic_write_json(canonical, value)


def inspect(args: argparse.Namespace, iteration: int) -> tuple[str, int | None]:
    root = args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}"
    recipe = read_json(args.recipe)
    needs_doc = iteration in {
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
    } - {0}
    canonical = root / "authoritative_attempt.json"
    if canonical.is_file():
        chosen = Path(read_json(canonical)["attempt_root"])
        return ("complete", int(chosen.name.split("_")[-1])) if attempt_complete(chosen, needs_doc) else ("failed", None)
    attempts = attempt_dirs(root)
    if not attempts:
        return "missing", None
    latest = attempts[-1]
    number = int(latest.name.split("_")[-1])
    if attempt_complete(latest, needs_doc):
        canonicalize(root, latest)
        return "complete", number
    submission = latest / "submission.json"
    if not submission.is_file():
        return "failed", number
    jobs = [str(value) for value in read_json(submission).get("jobs", {}).values() if value]
    states = {job: state(job) for job in jobs}
    if any(value in ACTIVE or value == "UNKNOWN" for value in states.values()):
        return "active", number
    if jobs and all(value in SUCCESS for value in states.values()) and attempt_complete(latest, needs_doc):
        canonicalize(root, latest)
        return "complete", number
    for job, value in states.items():
        if value not in SUCCESS:
            subprocess.run(["scancel", job], check=False)
    return "failed", number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--train-job-id", required=True)
    parser.add_argument("--iterations", required=True)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=41400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.code_root = args.code_root.resolve(); args.stage_root = args.stage_root.resolve(); args.run_root = args.run_root.resolve(); args.recipe = args.recipe.resolve()
    iterations = decode_evaluation_iterations(args.iterations)
    started = time.monotonic()
    while True:
        completed = []
        active = []
        waiting = []
        blocked = []
        for iteration in iterations:
            status, attempt = inspect(args, iteration)
            if status == "complete":
                completed.append(iteration)
            elif status == "active":
                active.append(iteration)
            elif status in {"missing", "failed"}:
                next_attempt = 0 if attempt is None else attempt + 1
                if next_attempt > 2:
                    blocked.append(iteration)
                else:
                    waiting.append((iteration, next_attempt))
        if blocked:
            raise RuntimeError(f"checkpoint evaluation attempts exhausted: {blocked}")
        slots = max(0, args.max_active - len(active))
        for iteration, attempt in waiting[:slots]:
            submit_pipeline(args, iteration, attempt)
            active.append(iteration)
        if len(completed) == len(iterations):
            payload = {
                "schema_version": "apertus_full_8b_evaluation_queue_v1",
                "status": "completed",
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "iterations": iterations,
                "max_active": args.max_active,
            }
            atomic_write_json(args.output, payload)
            print(json.dumps({"ok": True, "completed": len(completed)}))
            return 0
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError(f"evaluation queue exceeded controller window; completed={completed}, active={active}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

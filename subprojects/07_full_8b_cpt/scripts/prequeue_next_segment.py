#!/usr/bin/env python3
"""Submit one fail-closed successor allocation before its source segment ends."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def test_submission(command: list[str]) -> dict[str, object]:
    """Ask Slurm to validate the exact request without creating a job."""

    test_command = [command[0], "--test-only", *command[1:]]
    result = subprocess.run(
        test_command, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "sbatch --test-only rejected successor allocation: "
            + json.dumps({
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }, sort_keys=True)
        )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def source_state(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        text=True, capture_output=True, check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return rows[0] if result.returncode == 0 and rows else "UNKNOWN"


def active_job_state(job_id: str) -> str:
    state = source_state(job_id)
    if state not in {"PENDING", "RUNNING"}:
        raise RuntimeError(
            f"job {job_id} is {state}, expected PENDING or RUNNING"
        )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-root", type=Path, required=True)
    parser.add_argument("--scientific-receipt", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--ops-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-megatron", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--prelaunch-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--train-leaf-switch", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--source-segment", type=int, required=True)
    parser.add_argument("--source-train-job", required=True)
    parser.add_argument("--target-segment", type=int, required=True)
    parser.add_argument("--minimum-train-seconds", type=int, required=True)
    parser.add_argument("--maximum-hold-seconds", type=int, required=True)
    parser.add_argument("--eligible-after-minutes", type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "scientific_root", "scientific_receipt", "ops_root", "ops_receipt",
        "stage_root", "run_root", "initial_megatron", "selected_profile",
        "launch_gate", "prelaunch_root", "recipe", "profiles", "manifest",
        "output", "schedule",
    ):
        setattr(args, name, getattr(args, name).resolve())

    if args.target_segment != args.source_segment + 1:
        raise ValueError("target segment must be the immediate successor")
    if args.minimum_train_seconds <= 0 or args.maximum_hold_seconds < 0:
        raise ValueError("invalid allocation-time budget")
    source = source_state(args.source_train_job)
    allowed_source_states = (
        {"PENDING", "RUNNING"}
        if args.eligible_after_minutes is not None
        else {"RUNNING"}
    )
    if source not in allowed_source_states:
        raise RuntimeError(
            f"source training job is {source}; expected {sorted(allowed_source_states)}"
        )
    if args.eligible_after_minutes is not None and args.eligible_after_minutes <= 0:
        raise ValueError("eligible-after minutes must be positive")

    selected = read_json(args.selected_profile)
    boundaries = [int(value) for value in selected["selection"]["segment_boundaries"]]
    if not 0 <= args.target_segment < len(boundaries) - 1:
        raise ValueError("target segment is outside the selected profile")
    start = boundaries[args.target_segment]
    end = boundaries[args.target_segment + 1]
    if boundaries[args.source_segment + 1] != start:
        raise ValueError("source and target segment boundaries are not contiguous")
    nodes = int(selected["selection"]["nodes"])
    if nodes != 16:
        raise ValueError("prequeue policy is frozen to the proven 16-node profile")
    schedule = read_json(args.schedule)
    if (
        schedule.get("schema_version") != "apertus_full_8b_prequeue_schedule_v1"
        or schedule.get("status") != "approved"
        or schedule.get("segment_boundaries") != boundaries
    ):
        raise ValueError("prequeue schedule drift")
    allocation_seconds = int(schedule.get("allocation_seconds", -1))
    reserve_seconds = int(schedule.get("allocation_reserve_seconds", -1))
    if allocation_seconds != 43_200 or reserve_seconds != 1_200:
        raise ValueError("prequeue allocation or reserve drift")
    if (
        args.minimum_train_seconds
        + args.maximum_hold_seconds
        + reserve_seconds
        != allocation_seconds
    ):
        raise ValueError("train plus hold budget does not close exactly")
    policies = [
        row for row in schedule.get("targets", [])
        if int(row["target_segment_id"]) == args.target_segment
    ]
    if len(policies) != 1:
        raise ValueError("prequeue schedule must contain exactly one target policy")
    policy = policies[0]
    if (
        int(policy["minimum_train_seconds"]) != args.minimum_train_seconds
        or int(policy["maximum_hold_seconds"]) != args.maximum_hold_seconds
    ):
        raise ValueError("prequeue allocation-time budget drift")
    if (
        args.eligible_after_minutes is not None
        and int(policy["source_trigger_minutes"]) != args.eligible_after_minutes
    ):
        raise ValueError("prequeue source-trigger drift")
    source_minimum_train_seconds = int(
        policy.get("source_minimum_train_seconds", -1)
    )
    source_trigger_seconds = int(policy["source_trigger_minutes"]) * 60
    if source_minimum_train_seconds <= 0:
        raise ValueError("prequeue source conservative-runtime budget missing")
    if not 0 < source_trigger_seconds < source_minimum_train_seconds:
        raise ValueError("prequeue source-trigger is outside conservative source runtime")
    if source_minimum_train_seconds - source_trigger_seconds != args.maximum_hold_seconds:
        raise ValueError(
            "prequeue source-trigger does not preserve the target hold budget"
        )

    exclusion = subprocess.run(
        [
            str(args.scientific_root / "subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh"),
            args.train_leaf_switch,
            str(nodes),
        ],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if not exclusion:
        raise ValueError("empty leaf-switch exclusion")

    permit = (
        args.run_root / "orchestration/prequeue_permits"
        / f"segment_{args.target_segment}.json"
    ).resolve()
    if permit.exists():
        raise FileExistsError(f"prequeue permit already exists: {permit}")

    lock_path = args.manifest.with_suffix(args.manifest.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    submitted_job: str | None = None
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        graph = read_json(args.manifest)
        if (
            graph.get("schema_version") != "apertus_full_8b_prequeued_launch_graph_v1"
            or graph.get("status") != "submitted"
        ):
            raise ValueError("prequeued manifest drift")
        matches = [
            row for row in graph.get("segments", [])
            if int(row["segment_id"]) == args.target_segment
        ]
        if matches:
            row = matches[0]
            if (
                len(matches) == 1
                and str(row.get("source_train_job")) == args.source_train_job
                and active_job_state(str(row.get("train_job"))) in {"PENDING", "RUNNING"}
            ):
                atomic_write_json(args.output, {
                    "schema_version": "apertus_full_8b_prequeue_submission_v1",
                    "status": "already_submitted",
                    "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "target_segment_id": args.target_segment,
                    "train_job": str(row.get("train_job")),
                })
                return 0
            raise ValueError("duplicate or conflicting prequeued target segment")

        common = [
            f"FULL8_CODE_ROOT={args.scientific_root}",
            f"FULL8_CODE_BUNDLE_RECEIPT={args.scientific_receipt}",
            f"FULL8_OPS_ROOT={args.ops_root}",
            f"FULL8_OPS_BUNDLE_RECEIPT={args.ops_receipt}",
            f"FULL8_STAGE_ROOT={args.stage_root}",
            f"FULL8_RUN_ROOT={args.run_root}",
            f"FULL8_INITIAL_MEGATRON={args.initial_megatron}",
            f"FULL8_SELECTED_PROFILE={args.selected_profile}",
            f"FULL8_EXECUTION_PROFILE={selected['selection']['profile_id']}",
            f"FULL8_LAUNCH_GATE={args.launch_gate}",
            f"FULL8_PRELAUNCH_ROOT={args.prelaunch_root}",
            f"FULL8_RECIPE={args.recipe}",
            f"FULL8_PROFILES={args.profiles}",
            f"FULL8_TRAIN_LEAF_SWITCH={args.train_leaf_switch}",
            f"FULL8_PREQUEUED_MANIFEST={args.manifest}",
            f"FULL8_PREQUEUE_SCHEDULE={args.schedule}",
            f"FULL8_PREQUEUE_PERMIT={permit}",
            f"FULL8_PREQUEUE_SOURCE_SEGMENT={args.source_segment}",
            f"FULL8_PREQUEUE_SOURCE_TRAIN_JOB={args.source_train_job}",
            f"FULL8_SEGMENT_ID={args.target_segment}",
            "FULL8_ATTEMPT=0",
            f"FULL8_START_ITERATION={start}",
            f"FULL8_END_ITERATION={end}",
            f"FULL8_LOAD_CHECKPOINT={args.run_root / 'checkpoints'}",
            "FULL8_RECOVERY_MODE=0",
            f"FULL8_MIN_TRAIN_SECONDS={args.minimum_train_seconds}",
            f"FULL8_MAX_HOLD_SECONDS={args.maximum_hold_seconds}",
            f"FULL8_ALLOCATION_RESERVE_SECONDS={reserve_seconds}",
        ]
        command = [
            "sbatch", "--uenv-passthrough=ignore", "--parsable",
            "--partition=normal", "--time=12:00:00",
            "--switches=1", f"--exclude={exclusion}", f"--nodes={nodes}",
            f"--job-name=full8b_s{args.target_segment}a0_hold",
            f"--output={args.run_root}/logs/%x-%j.out",
            f"--error={args.run_root}/logs/%x-%j.err",
            f"--export=ALL,{','.join(common)}",
            str(args.ops_root / "clariden/run_prequeued_train_holder.sbatch"),
        ]
        if args.eligible_after_minutes is not None:
            command.insert(
                3,
                f"--dependency=after:{args.source_train_job}+{args.eligible_after_minutes}",
            )
        if args.test_only:
            validation = test_submission(command)
            atomic_write_json(args.output, {
                "schema_version": "apertus_full_8b_prequeue_submission_v1",
                "status": "test_only_passed",
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_segment_id": args.source_segment,
                "source_train_job": args.source_train_job,
                "target_segment_id": args.target_segment,
                "updates": [start, end],
                "minimum_train_seconds": args.minimum_train_seconds,
                "maximum_hold_seconds": args.maximum_hold_seconds,
                "eligible_after_minutes": args.eligible_after_minutes,
                "slurm_validation": validation,
            })
            print(json.dumps(validation, sort_keys=True))
            return 0
        allocation_receipt = (
            args.run_root / "orchestration/allocation_receipts"
            / f"prequeued_s{args.target_segment}_pending.json"
        )
        try:
            submitted_job = submit(command)
            allocation_receipt = allocation_receipt.with_name(
                f"prequeued_s{args.target_segment}_{submitted_job}.json"
            )
            subprocess.run([
                sys.executable,
                str(args.ops_root / "scripts/audit_submitted_job_resources.py"),
                "--scientific-root", str(args.scientific_root),
                "--scientific-receipt", str(args.scientific_receipt),
                "--ops-root", str(args.ops_root),
                "--ops-receipt", str(args.ops_receipt),
                "--job", f"train={submitted_job}",
                "--output", str(allocation_receipt),
            ], check=True)
            graph.setdefault("segments", []).append({
                "segment_id": args.target_segment,
                "source_segment_id": args.source_segment,
                "source_train_job": args.source_train_job,
                "train_job": submitted_job,
                "start_iteration": start,
                "end_iteration": end,
                "minimum_train_seconds": args.minimum_train_seconds,
                "maximum_hold_seconds": args.maximum_hold_seconds,
                "allocation_reserve_seconds": reserve_seconds,
                "permit_path": str(permit),
                "operational_root": str(args.ops_root),
                "operational_receipt": str(args.ops_receipt),
                "prequeue_schedule": str(args.schedule),
                "eligible_after_minutes": args.eligible_after_minutes,
                "allocation_receipt": str(allocation_receipt.resolve()),
            })
            graph["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            atomic_write_json(args.manifest, graph)
        except Exception:
            if submitted_job:
                subprocess.run(["scancel", submitted_job], check=False)
            raise

    atomic_write_json(args.output, {
        "schema_version": "apertus_full_8b_prequeue_submission_v1",
        "status": "submitted",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_segment_id": args.source_segment,
        "source_train_job": args.source_train_job,
        "target_segment_id": args.target_segment,
        "train_job": submitted_job,
        "updates": [start, end],
        "minimum_train_seconds": args.minimum_train_seconds,
        "maximum_hold_seconds": args.maximum_hold_seconds,
        "eligible_after_minutes": args.eligible_after_minutes,
        "allocation_receipt": str(allocation_receipt.resolve()),
    })
    print(submitted_job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

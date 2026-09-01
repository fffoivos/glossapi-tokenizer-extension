#!/usr/bin/env python3
"""Atomically replace a pending legacy campaign supervisor with a new bundle."""

from __future__ import annotations

import argparse
import datetime as dt
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
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def state(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        text=True, capture_output=True, check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return rows[0] if result.returncode == 0 and rows else "ABSENT"


def submit(command: list[str]) -> str:
    failures: list[dict[str, object]] = []
    for attempt in range(1, 6):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        job_id = result.stdout.strip().split(";", 1)[0]
        if result.returncode == 0 and job_id.isdigit():
            return job_id
        failures.append({
            "attempt": attempt,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        })
        if attempt < 5:
            time.sleep(2.0 * attempt)
    raise RuntimeError("sbatch failed after five bounded attempts: " + json.dumps(failures))


def verify_bundle(scientific_root: Path, scientific_receipt: Path,
                  ops_root: Path, ops_receipt: Path) -> None:
    verifier = (
        scientific_root
        / "subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
    )
    for root, receipt, kind in (
        (scientific_root, scientific_receipt, "scientific"),
        (ops_root, ops_receipt, "efficiency"),
    ):
        subprocess.run(
            [sys.executable, str(verifier), "--root", str(root),
             "--receipt", str(receipt), "--kind", kind],
            check=True,
        )


def validate_old_supervisor_binding(args: argparse.Namespace, old: dict) -> Path | None:
    """Bind either an original receipt or an already-audited prior transition.

    The latter permits a second, receipt-bound operational repair while the
    current supervisor is still pending. It never infers a job from a name.
    """

    if old.get("schema_version") == "apertus_full_8b_supervisor_submission_v1":
        expected = {
            "status": "passed",
            "next_segment": args.segment,
            "next_segment_start": args.attempt_start,
            "source_train_job": args.source_train_job,
            "supervisor_job": args.old_supervisor_job,
        }
        for key, value in expected.items():
            if old.get(key) != value:
                raise ValueError(f"legacy supervisor receipt drift for {key}")
        return None

    if old.get("schema_version") != "apertus_full_8b_supervisor_transition_v1":
        raise ValueError("unsupported pending-supervisor receipt schema")
    expected = {
        "status": "completed",
        "segment": args.segment,
        "attempt": args.attempt,
        "attempt_start": args.attempt_start,
        "source_train_job": args.source_train_job,
        "replacement_supervisor_job": args.old_supervisor_job,
    }
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f"prior supervisor-transition drift for {key}")
    root_text = old.get("replacement_operational_root")
    routing_text = old.get("allocation_routing_receipt")
    if not isinstance(root_text, str) or not root_text:
        raise ValueError("prior supervisor-transition lacks replacement bundle")
    if not isinstance(routing_text, str) or not routing_text:
        raise ValueError("prior supervisor-transition lacks allocation audit")
    routing = read_json(Path(routing_text))
    if (
        routing.get("schema_version")
        != "apertus_full_8b_allocation_routing_receipt_v1"
        or routing.get("status") != "passed"
    ):
        raise ValueError("prior supervisor allocation audit drift")
    jobs = routing.get("jobs")
    if not isinstance(jobs, list) or not any(
        row.get("role") == "supervisor"
        and str(row.get("job_id")) == args.old_supervisor_job
        and row.get("partition") == "debug"
        and int(row.get("nodes", -1)) == 1
        and int(row.get("time_limit_seconds", -1)) == 1200
        for row in jobs
    ):
        raise ValueError("prior supervisor allocation audit does not bind job")
    return Path(root_text).resolve()


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
    parser.add_argument("--prequeued-manifest", type=Path, required=True)
    parser.add_argument("--prequeue-schedule", type=Path, required=True)
    parser.add_argument("--old-supervisor-job", required=True)
    parser.add_argument("--old-supervisor-receipt", type=Path, required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--attempt-start", type=int, required=True)
    parser.add_argument("--source-train-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "scientific_root", "scientific_receipt", "ops_root", "ops_receipt",
        "stage_root", "run_root", "initial_megatron", "selected_profile",
        "launch_gate", "prelaunch_root", "recipe", "profiles",
        "prequeued_manifest", "prequeue_schedule", "old_supervisor_receipt",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not args.old_supervisor_job.isdigit() or not args.source_train_job.isdigit():
        raise ValueError("supervisor and source training job ids must be numeric")
    if args.output.exists():
        raise FileExistsError(args.output)

    verify_bundle(
        args.scientific_root, args.scientific_receipt, args.ops_root,
        args.ops_receipt,
    )
    old = read_json(args.old_supervisor_receipt)
    prior_ops_root = validate_old_supervisor_binding(args, old)
    if prior_ops_root is not None:
        verify_bundle(
            args.scientific_root,
            args.scientific_receipt,
            prior_ops_root,
            Path(f"{prior_ops_root}.receipt.json"),
        )
    old_state = state(args.old_supervisor_job)
    if old_state != "PENDING":
        raise RuntimeError(
            f"legacy supervisor {args.old_supervisor_job} is {old_state}, expected PENDING"
        )

    exports = [
        f"FULL8_CODE_ROOT={args.scientific_root}",
        f"FULL8_CODE_BUNDLE_RECEIPT={args.scientific_receipt}",
        f"FULL8_OPS_ROOT={args.ops_root}",
        f"FULL8_OPS_BUNDLE_RECEIPT={args.ops_receipt}",
        f"FULL8_STAGE_ROOT={args.stage_root}",
        f"FULL8_RUN_ROOT={args.run_root}",
        f"FULL8_INITIAL_MEGATRON={args.initial_megatron}",
        f"FULL8_SELECTED_PROFILE={args.selected_profile}",
        f"FULL8_LAUNCH_GATE={args.launch_gate}",
        f"FULL8_PRELAUNCH_ROOT={args.prelaunch_root}",
        f"FULL8_RECIPE={args.recipe}",
        f"FULL8_PROFILES={args.profiles}",
        f"FULL8_TRAIN_LEAF_SWITCH={args.train_leaf_switch}",
        f"FULL8_PREQUEUED_MANIFEST={args.prequeued_manifest}",
        f"FULL8_PREQUEUE_SCHEDULE={args.prequeue_schedule}",
    ]
    command = [
        "sbatch", "--uenv-passthrough=ignore", "--parsable",
        "--partition=debug", "--time=00:20:00", "--nodes=1",
        f"--dependency=afterany:{args.source_train_job}",
        f"--job-name=full8b_supervise_s{args.segment}a{args.attempt}_v27",
        f"--output={args.run_root}/logs/%x-%j.out",
        f"--error={args.run_root}/logs/%x-%j.err",
        f"--export=ALL,{','.join(exports)}",
        str(args.ops_root / "clariden/supervise_campaign_resource_aware.sbatch"),
        str(args.segment), str(args.attempt), str(args.attempt_start),
        args.source_train_job,
    ]
    if args.test_only:
        result = subprocess.run(
            ["sbatch", "--test-only", *command[1:]],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        atomic_write_json(args.output, {
            "schema_version": "apertus_full_8b_supervisor_transition_v1",
            "status": "test_only_passed",
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "old_supervisor_job": args.old_supervisor_job,
            "old_state": old_state,
            "slurm_validation": {
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(),
            },
        })
        return 0

    replacement = submit(command)
    receipt = (
        args.run_root / "orchestration/allocation_receipts"
        / f"supervisor_transition_s{args.segment}_{args.old_supervisor_job}_{replacement}.json"
    )
    try:
        subprocess.run([
            sys.executable, str(args.ops_root / "scripts/audit_submitted_job_resources.py"),
            "--scientific-root", str(args.scientific_root),
            "--scientific-receipt", str(args.scientific_receipt),
            "--ops-root", str(args.ops_root), "--ops-receipt", str(args.ops_receipt),
            "--job", f"supervisor={replacement}", "--output", str(receipt),
        ], check=True)
        if state(replacement) != "PENDING":
            raise RuntimeError(f"replacement supervisor {replacement} is not pending")
        if state(args.old_supervisor_job) != "PENDING":
            raise RuntimeError("legacy supervisor left PENDING before cancellation")
        subprocess.run(["scancel", args.old_supervisor_job], check=True)
    except Exception:
        subprocess.run(["scancel", replacement], check=False)
        raise
    atomic_write_json(args.output, {
        "schema_version": "apertus_full_8b_supervisor_transition_v1",
        "status": "completed",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "old_supervisor_job": args.old_supervisor_job,
        "old_operational_root": old.get("operational_root"),
        "replacement_supervisor_job": replacement,
        "replacement_operational_root": str(args.ops_root),
        "source_train_job": args.source_train_job,
        "segment": args.segment,
        "attempt": args.attempt,
        "attempt_start": args.attempt_start,
        "allocation_routing_receipt": str(receipt.resolve()),
    })
    print(replacement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

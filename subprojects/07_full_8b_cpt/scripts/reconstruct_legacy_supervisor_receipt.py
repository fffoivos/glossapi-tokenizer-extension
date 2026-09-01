#!/usr/bin/env python3
"""Reconstruct a missing v19 supervisor-submission receipt from immutable evidence.

This bridge exists only for the one historical v19 continuation which submitted
and resource-audited the next supervisor but did not write the later
``supervisor_submission_receipts`` record.  It never submits, cancels, holds,
or otherwise changes a Slurm job.  It writes the missing receipt only when the
completed evaluation, prequeued manifest, routing audit, and live pending job
all bind the same successor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def slurm_job_fields(job_id: str) -> dict[str, str]:
    result = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"cannot inspect pending supervisor {job_id}: {result.stderr.strip()}"
        )
    fields = dict(re.findall(r"([A-Za-z][A-Za-z0-9]*)=([^\s]+)", result.stdout))
    require(fields.get("JobId") == job_id, "scontrol job id drift")
    return fields


def has_afterany_dependency(value: str, job_id: str) -> bool:
    return bool(re.search(rf"(?:^|,)afterany:{re.escape(job_id)}(?:\(|,|$)", value))


def expected_payload(args: argparse.Namespace, evidence: dict[str, Path]) -> dict:
    return {
        "schema_version": "apertus_full_8b_supervisor_submission_v1",
        "status": "passed",
        "reconstructed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reconstruction": "v19_missing_supervisor_receipt_bridge_v1",
        "next_segment": args.next_segment,
        "next_segment_start": args.next_segment_start,
        "source_train_job": args.next_train_job,
        "supervisor_job": args.supervisor_job,
        "operational_root": str(args.operational_root),
        "allocation_routing_receipt": str(args.allocation_routing_receipt),
        "evidence_sha256": {
            name: sha256_file(path) for name, path in sorted(evidence.items())
        },
    }


def existing_matches(path: Path, expected: dict) -> bool:
    value = read_json(path)
    for key in (
        "schema_version",
        "status",
        "reconstruction",
        "next_segment",
        "next_segment_start",
        "source_train_job",
        "supervisor_job",
        "operational_root",
        "allocation_routing_receipt",
        "evidence_sha256",
    ):
        if value.get(key) != expected.get(key):
            return False
    return True


def exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate(args: argparse.Namespace) -> dict[str, Path]:
    evaluation_root = (
        args.run_root
        / "checkpoint_evaluations"
        / f"iter_{args.completed_iteration:07d}"
    )
    evidence = {
        "authoritative_attempt": evaluation_root / "authoritative_attempt.json",
        "exact_native_greekmmlu": (
            evaluation_root
            / "attempt_0"
            / "exact_checkpoint_native_greekmmlu_receipt.json"
        ),
        "evaluation_queue": (
            args.run_root / "evaluation_queues" / f"segment_{args.source_segment}.json"
        ),
        "campaign_latest": args.run_root / "orchestration" / "latest.json",
        "allocation_routing": args.allocation_routing_receipt,
        "prequeued_manifest": args.prequeued_manifest,
    }
    for name, path in evidence.items():
        require(path.is_file(), f"missing {name} evidence: {path}")

    authoritative = read_json(evidence["authoritative_attempt"])
    require(
        authoritative.get("status") == "completed"
        and authoritative.get("iteration") == args.completed_iteration,
        "authoritative evaluation receipt drift",
    )
    exact = read_json(evidence["exact_native_greekmmlu"])
    metrics = exact.get("metrics", {})
    require(
        exact.get("status") == "completed" and int(metrics.get("n", 0)) > 0,
        "exact native GreekMMLU receipt is incomplete",
    )
    queue = read_json(evidence["evaluation_queue"])
    require(
        queue.get("schema_version") == "apertus_full_8b_evaluation_queue_v1"
        and queue.get("status") == "completed"
        and args.completed_iteration in queue.get("iterations", []),
        "completed evaluation queue receipt drift",
    )
    latest = read_json(evidence["campaign_latest"])
    expected_latest = {
        "event": "next_segment_supervisor_submitted",
        "source_segment": args.source_segment,
        "completed_iteration": args.completed_iteration,
        "next_segment": args.next_segment,
        "next_train_job": args.next_train_job,
        "next_supervisor_job": args.supervisor_job,
    }
    for key, value in expected_latest.items():
        require(latest.get(key) == value, f"campaign event drift for {key}")

    routing = read_json(evidence["allocation_routing"])
    require(
        routing.get("schema_version") == "apertus_full_8b_allocation_routing_receipt_v1"
        and routing.get("status") == "passed",
        "allocation routing receipt is not passed",
    )
    operational = routing.get("operational_bundle", {})
    require(
        operational.get("root") == str(args.operational_root)
        and all(operational.get("checks", {}).values()),
        "routing receipt operational bundle drift",
    )
    jobs = routing.get("jobs", [])
    matching = [
        row for row in jobs
        if row.get("role") == "supervisor" and str(row.get("job_id")) == args.supervisor_job
    ]
    require(len(matching) == 1, "routing receipt lacks the exact supervisor")
    row = matching[0]
    require(
        row.get("partition") == "debug"
        and row.get("nodes") == 1
        and row.get("time_limit_seconds") == 1200
        and row.get("state") == "PENDING",
        "routing receipt supervisor geometry drift",
    )

    manifest = read_json(evidence["prequeued_manifest"])
    rows = [
        row for row in manifest.get("segments", [])
        if row.get("segment_id") == args.next_segment
    ]
    require(len(rows) == 1, "prequeued manifest lacks one successor segment")
    row = rows[0]
    require(
        str(row.get("train_job")) == args.next_train_job
        and row.get("start_iteration") == args.next_segment_start,
        "prequeued manifest successor drift",
    )

    fields = slurm_job_fields(args.supervisor_job)
    expected_command = str(
        args.operational_root / "clariden" / "supervise_campaign_resource_aware.sbatch"
    )
    require(fields.get("JobState") == "PENDING", "legacy supervisor is no longer pending")
    require(fields.get("Partition") == "debug", "legacy supervisor partition drift")
    require(fields.get("TimeLimit") == "00:20:00", "legacy supervisor time limit drift")
    require(fields.get("NumNodes") in {"1", "1-1"}, "legacy supervisor node count drift")
    require(
        has_afterany_dependency(fields.get("Dependency", ""), args.next_train_job),
        "legacy supervisor dependency drift",
    )
    require(fields.get("Command") == expected_command, "legacy supervisor command drift")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prequeued-manifest", type=Path, required=True)
    parser.add_argument("--allocation-routing-receipt", type=Path, required=True)
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--supervisor-job", required=True)
    parser.add_argument("--next-train-job", required=True)
    parser.add_argument("--source-segment", type=int, required=True)
    parser.add_argument("--next-segment", type=int, required=True)
    parser.add_argument("--next-segment-start", type=int, required=True)
    parser.add_argument("--completed-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "run_root", "prequeued_manifest", "allocation_routing_receipt",
        "operational_root", "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    require(args.supervisor_job.isdigit(), "supervisor job id must be numeric")
    require(args.next_train_job.isdigit(), "next training job id must be numeric")
    require(args.source_segment >= 0, "source segment must be non-negative")
    require(args.next_segment == args.source_segment + 1, "segment continuity drift")
    require(args.next_segment_start > 0, "successor start must be positive")

    evidence = validate(args)
    payload = expected_payload(args, evidence)
    if args.test_only:
        print(json.dumps({
            "status": "test_only_passed",
            "supervisor_job": args.supervisor_job,
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    if args.output.exists():
        if not existing_matches(args.output, payload):
            raise ValueError(f"existing reconstructed receipt drift: {args.output}")
        print(str(args.output))
        return 0
    exclusive_write(args.output, payload)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

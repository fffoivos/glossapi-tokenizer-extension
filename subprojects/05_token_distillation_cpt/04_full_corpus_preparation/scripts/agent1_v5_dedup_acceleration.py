#!/usr/bin/env python3
"""Immutable planning and closure checks for Agent-1 v5 dedup acceleration.

This module intentionally does not submit or cancel Slurm jobs.  It produces
and validates receipts consumed by the separate held-job submission wrapper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import agent1_v5_datatrove as dedup  # noqa: E402
from agent1_v5_pipeline import canonical_json, sha256_file  # noqa: E402


BOUNDARY_SCHEMA = "agent1_v5_dedup_acceleration_boundary_v1"
PREFLIGHT_SCHEMA = "agent1_v5_dedup_acceleration_preflight_v1"
CUTOVER_SCHEMA = "agent1_v5_dedup_acceleration_cutover_v1"
BENCHMARK_PLAN_SCHEMA = "agent1_v5_dedup_acceleration_benchmark_plan_v1"
BENCHMARK_SCHEMA = "agent1_v5_dedup_acceleration_benchmark_v1"
CHUNK_PLAN_SCHEMA = "agent1_v5_dedup_acceleration_chunk_plan_v1"
SUBMISSION_SCHEMA = "agent1_v5_dedup_acceleration_submission_v1"
RELEASE_AUTHORIZATION_SCHEMA = "agent1_v5_dedup_acceleration_release_authorization_v1"
ARRAY_EXECUTION_EVIDENCE_SCHEMA = "agent1_v5_dedup_acceleration_array_execution_evidence_v1"
RECOVERY_SCHEMA = "agent1_v5_dedup_acceleration_recovery_v1"
EXECUTION_SCHEMA = "agent1_v5_dedup_acceleration_execution_v1"
SENTINEL_REQUEST_SCHEMA = "agent1_v5_dedup_acceleration_takeover_request_v1"
SENTINEL_ARM_SCHEMA = "agent1_v5_dedup_acceleration_takeover_arm_v1"
SENTINEL_STOP_SCHEMA = "agent1_v5_dedup_acceleration_sentinel_stop_v1"
SENTINEL_QUEUE_SCHEMA = "agent1_v5_dedup_acceleration_sentinel_queue_evidence_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    def compatible(existing: Mapping[str, Any]) -> bool:
        comparable_existing = dict(existing)
        comparable_value = dict(value)
        comparable_existing.pop("created_at", None)
        comparable_value.pop("created_at", None)
        return canonical_json(comparable_existing) == canonical_json(comparable_value)

    if path.exists():
        existing = _read(path)
        if not compatible(existing):
            raise FileExistsError(f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with os.fdopen(os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read(path)
            if not compatible(existing):
                raise FileExistsError(f"immutable output differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _passed(path: Path, *, label: str) -> dict[str, Any]:
    value = _read(path)
    if value.get("status") != "passed":
        raise ValueError(f"{label} is not passed: {path}")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _parse_named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"evidence must be LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes run root: {path}") from error


def _validate_receipts(run_root: Path, through_rank: int) -> list[dict[str, Any]]:
    if through_rank < 0:
        raise ValueError("through rank must be non-negative")
    receipts: list[dict[str, Any]] = []
    for rank in range(through_rank + 1):
        path = run_root / "60-dedup" / "minhash-signatures" / "receipts" / f"{rank:06d}.json"
        receipt = _read(path)
        if receipt.get("schema_version") != dedup.SIGNATURE_RECEIPT_SCHEMA:
            raise ValueError(f"signature receipt schema mismatch: {path}")
        if receipt.get("status") != "passed" or int(receipt.get("task_index", -1)) != rank:
            raise ValueError(f"signature receipt is not passed for rank {rank}")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 32:
            raise ValueError(f"signature receipt output closure failed for rank {rank}")
        for output in outputs:
            dedup.validate_file_receipt(output, root=run_root)
        receipts.append(receipt)
    return receipts


def validate_boundary(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    receipts = _validate_receipts(run_root, int(args.through_rank))
    manifest = _read(args.combined_manifest)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("combined manifest files missing")
    next_rank = int(args.through_rank) + 1
    if next_rank >= len(files):
        raise ValueError("no unstarted rank remains")
    value: dict[str, Any] = {
        "schema_version": BOUNDARY_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "run_root": str(run_root),
        "through_rank": int(args.through_rank),
        "first_missing_rank": next_rank,
        "legacy_receipt_count": len(receipts),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "legacy_receipts": [
            {
                "rank": receipt["task_index"],
                "receipt_sha256": sha256_file(
                    run_root
                    / "60-dedup"
                    / "minhash-signatures"
                    / "receipts"
                    / f"{int(receipt['task_index']):06d}.json"
                ),
                "outputs": receipt["outputs"],
            }
            for receipt in receipts
        ],
        "fence_job_id": str(args.fence_job_id),
        "final_legacy_job_id": str(args.final_legacy_job_id),
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "first_missing_rank": next_rank, "receipts": len(receipts)}))
    return 0


def _validate_receipt(path: Path, schema: str) -> dict[str, Any]:
    value = _read(path)
    if value.get("schema_version") != schema or value.get("status") != "passed":
        raise ValueError(f"required receipt is not passed: {path}")
    return value


def _array_indices(array_spec: str) -> list[int]:
    match = re.fullmatch(r"([0-9]+)-([0-9]+)%([1-9][0-9]*)", array_spec)
    if match is None:
        raise ValueError(f"invalid array specification: {array_spec}")
    first, last = int(match.group(1)), int(match.group(2))
    if first < 0 or last < first:
        raise ValueError(f"invalid array bounds: {array_spec}")
    return list(range(first, last + 1))


def _validate_identifier(value: str, *, label: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{5,63}", value) is None:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _passed_signature_receipts(run_root: Path, last_rank: int) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    receipt_root = run_root / "60-dedup" / "minhash-signatures" / "receipts"
    for rank in range(last_rank + 1):
        path = receipt_root / f"{rank:06d}.json"
        if not path.exists():
            continue
        receipt = _read(path)
        if receipt.get("schema_version") != dedup.SIGNATURE_RECEIPT_SCHEMA:
            raise ValueError(f"signature receipt schema mismatch: {path}")
        if receipt.get("status") != "passed" or int(receipt.get("task_index", -1)) != rank:
            raise ValueError(f"signature receipt is not passed for rank {rank}")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 32:
            raise ValueError(f"signature receipt output closure failed for rank {rank}")
        for output in outputs:
            dedup.validate_file_receipt(output, root=run_root)
        receipts.append({"rank": rank, "receipt_sha256": sha256_file(path)})
    return receipts


def record_failed_array_recovery(args: argparse.Namespace) -> int:
    """Prove that a released array failed before producing canonical work."""

    run_root = args.run_root.resolve()
    recovery_id = _validate_identifier(args.recovery_id, label="recovery ID")
    submission = _validate_receipt(args.failed_submission, SUBMISSION_SCHEMA)
    authorization = _validate_receipt(args.failed_release_authorization, RELEASE_AUTHORIZATION_SCHEMA)
    release = _validate_receipt(args.failed_release_observation, "agent1_v5_dedup_acceleration_release_observation_v1")
    plan = _validate_receipt(args.failed_chunk_plan, CHUNK_PLAN_SCHEMA)
    evidence = _validate_receipt(args.scheduler_evidence, ARRAY_EXECUTION_EVIDENCE_SCHEMA)
    job_id = str(submission.get("array_job_id", ""))
    if not job_id.isdigit() or evidence.get("array_job_id") != job_id or release.get("array_job_id") != job_id:
        raise ValueError("failed-array job identity drift")
    if submission.get("run_root") != str(run_root):
        raise ValueError("failed submission run-root drift")
    if authorization.get("submission_receipt_sha256") != sha256_file(args.failed_submission):
        raise ValueError("failed release authorization does not bind submission receipt")
    if (
        authorization.get("array_job_id") != job_id
        or authorization.get("submission_nonce") != submission.get("submission_nonce")
    ):
        raise ValueError("failed release authorization identity drift")
    if release.get("submission_nonce") != submission.get("submission_nonce") or release.get("release_requested") is not True:
        raise ValueError("failed release observation nonce drift")
    if submission.get("chunk_plan_sha256") != sha256_file(args.failed_chunk_plan):
        raise ValueError("failed submission chunk-plan binding drift")
    expected_tasks = _array_indices(str(submission.get("array_spec", "")))
    if (
        evidence.get("attempt_id") != recovery_id
        or
        evidence.get("array_spec") != submission.get("array_spec")
        or evidence.get("expected_state") != "FAILED"
        or evidence.get("expected_exit_code") != "127:0"
    ):
        raise ValueError("failed-array scheduler evidence binding drift")
    tasks = evidence.get("tasks")
    if not isinstance(tasks, list) or sorted(int(task.get("task_index", -1)) for task in tasks) != expected_tasks:
        raise ValueError("failed-array scheduler evidence does not close every task")
    for task in tasks:
        if task.get("state") != "FAILED" or task.get("exit_code") != "127:0":
            raise ValueError("failed-array task did not fail with the expected pre-work exit 127")
        if task.get("account") != "a0140" or task.get("partition") != "normal":
            raise ValueError("failed-array scheduler identity drift")
        if task.get("job_id") != f"{job_id}_{int(task['task_index'])}":
            raise ValueError("failed-array scheduler task ID drift")
    raw_planned_ranks = [
        int(rank)
        for chunk in plan.get("chunks", [])
        for rank in chunk.get("ranks", [])
    ]
    planned_ranks = set(raw_planned_ranks)
    if not planned_ranks or len(planned_ranks) != len(raw_planned_ranks):
        raise ValueError("failed chunk plan has empty or duplicate ranks")
    if min(planned_ranks) < 0 or max(planned_ranks) > int(plan["last_rank"]):
        raise ValueError("failed chunk plan rank bounds drift")
    metrics_root = args.metrics_root.resolve()
    expected_metrics_root = run_root / "60-dedup" / "minhash-signatures" / "accelerated-metrics"
    if metrics_root != expected_metrics_root:
        raise ValueError("failed-array metrics root is not the canonical production path")
    production_metrics = []
    for path in metrics_root.glob("*.json"):
        metric = _read(path)
        if metric.get("benchmark_plan_sha256") is None:
            production_metrics.append(path)
    if production_metrics:
        raise ValueError("failed array produced production metrics; automatic pre-work recovery is unsafe")
    receipts = _passed_signature_receipts(run_root, int(plan["last_rank"]))
    if len(receipts) != int(args.expected_receipt_count):
        raise ValueError("canonical receipt count changed after failed array")
    receipt_ranks = {int(receipt["rank"]) for receipt in receipts}
    if receipt_ranks & planned_ranks:
        raise ValueError("failed array completed a rank from its pending plan")
    claim_root = run_root / "60-dedup" / "minhash-signatures" / "claims"
    active_claims = [rank for rank in planned_ranks if (claim_root / f"{rank:06d}.json").exists()]
    if active_claims:
        raise ValueError(f"failed array left active claims: {active_claims[:10]}")
    old_runner_sha = str(submission.get("runner_sha256", ""))
    new_runner_sha = sha256_file(args.new_runner)
    if not old_runner_sha or old_runner_sha == new_runner_sha:
        raise ValueError("recovery requires a distinct corrected runner")
    new_pipeline_root = args.new_pipeline_root.resolve()
    try:
        args.new_runner.resolve().relative_to(new_pipeline_root)
    except ValueError as error:
        raise ValueError("corrected runner is outside the corrected pipeline root") from error
    value: dict[str, Any] = {
        "schema_version": RECOVERY_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "recovery_id": recovery_id,
        "run_root": str(run_root),
        "failed_array_job_id": job_id,
        "failed_submission_sha256": sha256_file(args.failed_submission),
        "failed_release_authorization_sha256": sha256_file(args.failed_release_authorization),
        "failed_release_observation_sha256": sha256_file(args.failed_release_observation),
        "failed_chunk_plan_sha256": sha256_file(args.failed_chunk_plan),
        "scheduler_evidence_sha256": sha256_file(args.scheduler_evidence),
        "old_runner_sha256": old_runner_sha,
        "new_runner_sha256": new_runner_sha,
        "new_pipeline_root": str(new_pipeline_root),
        "failed_pending_ranks": sorted(planned_ranks),
        "canonical_receipt_count": len(receipts),
        "canonical_receipts": receipts,
        "production_metric_count": 0,
        "active_claim_count": 0,
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "recovery_id": recovery_id, "receipts": len(receipts)}))
    return 0


def build_preflight(args: argparse.Namespace) -> int:
    """Freeze the non-scheduler evidence required before a debug fence.

    Scheduler state deliberately arrives as a separately captured JSON object:
    it is too volatile to reuse an old preflight at the moment the fence is
    placed, so the Bash cutover controller repeats those checks immediately
    before ``sbatch --hold``.
    """

    run_root = args.run_root.resolve()
    contract = args.contract.resolve()
    manifest_path = args.combined_manifest.resolve()
    runtime = args.runtime_receipt.resolve()
    audit_path = args.full_input_audit.resolve()
    config = args.config.resolve()
    for label, path in {
        "contract": contract,
        "combined manifest": manifest_path,
        "runtime receipt": runtime,
        "full input audit": audit_path,
    }.items():
        _require_within(path, run_root, label=label)

    combined, _ = dedup._load_release_structure(manifest_path)
    dedup._validate_full_input_audit(
        audit_path,
        contract_path=contract,
        manifest_path=manifest_path,
        runtime_path=runtime,
        combined=combined,
    )
    exact = _passed(args.exact_manifest, label="exact-index manifest")
    if exact.get("combined_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("exact-index manifest is not bound to combined manifest")

    evidence: dict[str, dict[str, str]] = {}
    required_evidence = {
        "acquisition_integrity": args.acquisition_audit,
        "publication_pre": args.publication_pre,
        "publication_metadata": args.publication_metadata,
    }
    for label, path in required_evidence.items():
        _require_within(path, run_root, label=label)
        _passed(path, label=label)
        evidence[label] = _binding(path)
    for raw in args.evidence:
        label, path = _parse_named_path(raw)
        _require_within(path, run_root, label=f"evidence {label}")
        _passed(path, label=f"evidence {label}")
        if label in evidence:
            raise ValueError(f"duplicate evidence label: {label}")
        evidence[label] = _binding(path)

    snapshot = _read(args.scheduler_snapshot)
    expected_snapshot = {
        "partition": "debug",
        "qos": "debug-qos",
        "account": "a0140",
        "user": "fffoivos",
        "max_jobs_per_user": 1,
        "max_submit_jobs_per_user": 2,
        "effective_legacy_qos": "debug-qos",
    }
    for key, expected in expected_snapshot.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"scheduler snapshot drift for {key}: {snapshot.get(key)!r}")

    pipeline = args.pipeline_root.resolve()
    code_files = [
        pipeline / "scripts" / "agent1_v5_datatrove.py",
        pipeline / "scripts" / "agent1_v5_dedup_acceleration.py",
        pipeline / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh",
        pipeline / "slurm" / "agent1_v5_eiger" / "normal_signature_benchmark.sh",
        pipeline / "slurm" / "agent1_v5_eiger" / "submit_accelerated_signature_array.sh",
        pipeline / "slurm" / "agent1_v5_eiger" / "cutover_to_accelerated_signatures.sh",
        config,
    ]
    if not args.legacy_helper.is_file():
        raise FileNotFoundError(args.legacy_helper)
    for path in code_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    value: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "scope": "current_run_dedup_acceleration_pre_fence",
        "run_root": str(run_root),
        "construction_integrity_status": "passed",
        "blocking_acceleration_findings": [],
        "explicit_fence_approval": bool(args.approve_fence),
        "run_contract": _binding(contract),
        "combined_manifest": _binding(manifest_path),
        "runtime_receipt": _binding(runtime),
        "full_input_audit": _binding(audit_path),
        "exact_manifest": _binding(args.exact_manifest),
        "construction_evidence": evidence,
        "scheduler_snapshot": _binding(args.scheduler_snapshot),
        "pipeline_root": str(pipeline),
        "code_bindings": [_binding(path) for path in code_files],
        "legacy_helper": _binding(args.legacy_helper),
        "candidate_rank_inventory": [
            {"rank": row["rank"], "bytes": row["bytes"], "rows": row["rows"], "sha256": row["sha256"]}
            for row in combined["files"]
        ],
    }
    if value["explicit_fence_approval"] is not True:
        raise ValueError("preflight requires explicit --approve-fence")
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "tasks": len(combined["files"]), "status": "passed"}))
    return 0


def finalize_cutover(args: argparse.Namespace) -> int:
    """Turn a staged receipt boundary into the final cutover receipt."""

    preflight = _validate_receipt(args.preflight, PREFLIGHT_SCHEMA)
    if preflight.get("explicit_fence_approval") is not True:
        raise ValueError("preflight has no fence approval")
    boundary = _validate_receipt(args.boundary, BOUNDARY_SCHEMA)
    fence = _passed(args.fence_evidence, label="fence evidence")
    successor = _passed(args.successor_evidence, label="successor evidence")
    if str(fence.get("fence_job_id")) != str(boundary.get("fence_job_id")):
        raise ValueError("fence evidence does not bind boundary fence")
    if str(fence.get("final_legacy_job_id")) != str(boundary.get("final_legacy_job_id")):
        raise ValueError("fence evidence does not bind final legacy job")
    if fence.get("fence_cleanup_result") != "cancelled" or fence.get("debug_signature_queue_empty") is not True:
        raise ValueError("fence cleanup has not closed the debug signature queue")
    if successor.get("expected_successor_rejection") is not True or successor.get("no_successor") is not True:
        raise ValueError("successor rejection evidence is incomplete")
    value: dict[str, Any] = {
        "schema_version": CUTOVER_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "run_root": preflight["run_root"],
        "preflight_sha256": sha256_file(args.preflight),
        "boundary_sha256": sha256_file(args.boundary),
        "fence_evidence_sha256": sha256_file(args.fence_evidence),
        "successor_evidence_sha256": sha256_file(args.successor_evidence),
        "fence_job_id": str(boundary["fence_job_id"]),
        "final_legacy_job_id": str(boundary["final_legacy_job_id"]),
        "final_legacy_rank": int(boundary["through_rank"]),
        "first_missing_rank": int(boundary["first_missing_rank"]),
        "legacy_receipt_count": int(boundary["legacy_receipt_count"]),
        "combined_manifest_sha256": boundary["combined_manifest_sha256"],
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "first_missing_rank": value["first_missing_rank"]}))
    return 0


def finalize_sentinel_cutover(args: argparse.Namespace) -> int:
    """Finalize a serial handoff that stopped itself instead of using a QoS fence.

    The scheduler check is performed by the shell controller immediately before
    this command.  This function validates the immutable evidence and every
    signature receipt through the stop rank before exposing the standard
    cutover schema consumed by benchmark and production planning.
    """

    request = _validate_receipt(args.request, SENTINEL_REQUEST_SCHEMA)
    arm = _validate_receipt(args.arm_receipt, SENTINEL_ARM_SCHEMA)
    stop = _validate_receipt(args.stop_receipt, SENTINEL_STOP_SCHEMA)
    queue = _validate_receipt(args.queue_evidence, SENTINEL_QUEUE_SCHEMA)
    run_root = Path(request.get("run_root", "")).resolve()
    if not run_root.is_dir():
        raise ValueError("sentinel request run root is unavailable")
    stop_rank = int(request.get("stop_after_rank", -1))
    if stop_rank < 0:
        raise ValueError("sentinel request stop rank is invalid")
    manifest = args.combined_manifest.resolve()
    if request.get("combined_manifest_sha256") != sha256_file(manifest):
        raise ValueError("sentinel request manifest binding drift")
    if arm.get("request_sha256") != sha256_file(args.request):
        raise ValueError("sentinel arm does not bind request")
    if int(arm.get("stop_after_rank", -1)) != stop_rank:
        raise ValueError("sentinel arm stop rank drift")
    if stop.get("request_sha256") != sha256_file(args.request):
        raise ValueError("sentinel stop does not bind request")
    if int(stop.get("stopped_after_rank", -1)) != stop_rank:
        raise ValueError("sentinel stop rank drift")
    if int(stop.get("first_missing_rank", -1)) != stop_rank + 1:
        raise ValueError("sentinel first missing rank drift")
    if stop.get("successor_submitted") is not False:
        raise ValueError("sentinel stop claims a successor submission")
    if arm.get("active_helper_sha256") != request.get("guarded_helper_sha256"):
        raise ValueError("armed helper does not bind requested guard")
    if stop.get("active_helper_sha256") != request.get("guarded_helper_sha256"):
        raise ValueError("stop helper does not bind requested guard")
    if arm.get("takeover_tool_sha256") != request.get("takeover_tool_sha256"):
        raise ValueError("armed takeover tool drift")
    if stop.get("takeover_tool_sha256") != request.get("takeover_tool_sha256"):
        raise ValueError("stop takeover tool drift")
    receipt_binding = stop.get("signature_receipt")
    expected_receipt = run_root / "60-dedup" / "minhash-signatures" / "receipts" / f"{stop_rank:06d}.json"
    if not isinstance(receipt_binding, Mapping):
        raise ValueError("sentinel stop receipt binding missing")
    if receipt_binding.get("path") != str(expected_receipt.resolve()) or receipt_binding.get("sha256") != sha256_file(expected_receipt):
        raise ValueError("sentinel stop receipt binding drift")
    if queue.get("debug_signature_queue_empty") is not True or queue.get("legacy_successor_present") is not False:
        raise ValueError("sentinel queue evidence is incomplete")
    receipts = _validate_receipts(run_root, stop_rank)
    value: dict[str, Any] = {
        "schema_version": CUTOVER_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "method": "sentinel",
        "run_root": str(run_root),
        "request_sha256": sha256_file(args.request),
        "arm_receipt_sha256": sha256_file(args.arm_receipt),
        "stop_receipt_sha256": sha256_file(args.stop_receipt),
        "queue_evidence_sha256": sha256_file(args.queue_evidence),
        "final_legacy_rank": stop_rank,
        "first_missing_rank": stop_rank + 1,
        "legacy_receipt_count": len(receipts),
        "combined_manifest_sha256": sha256_file(manifest),
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "first_missing_rank": value["first_missing_rank"], "method": "sentinel"}))
    return 0


def make_benchmark_plan(args: argparse.Namespace) -> int:
    """Freeze 1→2→4→5 worker canary inputs before any benchmark starts."""

    audit = _validate_receipt(args.full_input_audit, dedup.FULL_INPUT_AUDIT_SCHEMA)
    cutover = _validate_receipt(args.cutover_receipt, CUTOVER_SCHEMA)
    manifest = _read(args.combined_manifest)
    if sha256_file(args.combined_manifest) != audit.get("combined_manifest_sha256"):
        raise ValueError("full audit is not bound to benchmark manifest")
    if cutover.get("combined_manifest_sha256") != audit.get("combined_manifest_sha256"):
        raise ValueError("cutover and full audit manifest bindings differ")
    first = int(cutover["first_missing_rank"])
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("combined manifest files missing")
    rank_count = int(args.rank_count)
    phase_specs = list(args.phase_ranks or [])
    expected_phases = ((1, 2, "baseline"), (2, 4, "two_worker"), (4, 8, "four_worker"), (5, 10, "five_worker"))
    row_by_rank = {int(row.get("rank", -1)): row for row in files}
    excluded: list[dict[str, Any]] = []
    if phase_specs:
        if len(phase_specs) != len(expected_phases):
            raise ValueError("explicit benchmark ranks require exactly four phase lists")
        parsed = []
        for raw, (_, count, _) in zip(phase_specs, expected_phases):
            ranks = [int(item) for item in raw.split(",") if item]
            if len(ranks) != count or len(set(ranks)) != len(ranks):
                raise ValueError("explicit benchmark phase has an invalid rank count or duplicate")
            parsed.append(ranks)
        flat = [rank for ranks in parsed for rank in ranks]
        if len(set(flat)) != 24 or min(flat) != first:
            raise ValueError("explicit benchmark must start at first missing rank and contain 24 unique ranks")
        for rank in flat:
            row = row_by_rank.get(rank)
            if row is None:
                raise ValueError(f"benchmark rank is absent from manifest: {rank}")
            if row.get("origin") != "nanochat_base" or int(row.get("rows", -1)) != 196608:
                raise ValueError("benchmark ranks must be homogeneous full NanoChat shards")
        selected = [row_by_rank[rank] for rank in flat]
        for rank in range(first, max(flat) + 1):
            if rank in flat:
                continue
            row = row_by_rank.get(rank)
            if row is None or row.get("origin") == "nanochat_base" and int(row.get("rows", -1)) == 196608:
                raise ValueError("explicit benchmark may skip only non-full NanoChat shards")
            excluded.append({"rank": rank, "bytes": row["bytes"], "rows": row["rows"], "sha256": row["sha256"], "reason": "non_full_nanochat_shard"})
    else:
        if rank_count != 24:
            raise ValueError("benchmark must freeze exactly 24 ranks")
        selected = files[first : first + rank_count]
        if len(selected) != rank_count:
            raise ValueError("not enough remaining ranks for benchmark")
        for expected_rank, row in enumerate(selected, start=first):
            if int(row.get("rank", -1)) != expected_rank:
                raise ValueError("benchmark ranks are not contiguous")
            if row.get("origin") != "nanochat_base" or int(row.get("rows", -1)) != 196608:
                raise ValueError("benchmark ranks must be homogeneous full NanoChat shards")
        parsed = []
        cursor = 0
        for _, count, _ in expected_phases:
            parsed.append([int(row["rank"]) for row in selected[cursor : cursor + count]])
            cursor += count
    phases = [
        {"index": index, "name": name, "workers": workers, "ranks": parsed[index]}
        for index, (workers, _, name) in enumerate(expected_phases)
    ]
    value: dict[str, Any] = {
        "schema_version": BENCHMARK_PLAN_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "run_root": str(args.run_root.resolve()),
        "full_input_audit_sha256": sha256_file(args.full_input_audit),
        "cutover_receipt_sha256": sha256_file(args.cutover_receipt),
        "combined_manifest_sha256": sha256_file(args.combined_manifest),
        "first_missing_rank": first,
        "phases": phases,
        "explicit_nonbenchmark_exclusions": excluded,
        "rank_inventory": [
            {"rank": row["rank"], "bytes": row["bytes"], "rows": row["rows"], "sha256": row["sha256"]}
            for row in selected
        ],
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "phases": len(phases), "ranks": len(selected), "excluded": len(excluded)}))
    return 0


def _phase_metrics(plan: Mapping[str, Any], metrics_root: Path) -> tuple[dict[int, list[tuple[Path, dict[str, Any]]]], list[dict[str, Any]]]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("benchmark plan phases missing")
    expected = {
        (int(phase["index"]), int(rank)): int(phase["workers"])
        for phase in phases
        for rank in phase["ranks"]
    }
    found: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(metrics_root.glob("*.json")):
        metric = _read(path)
        if metric.get("schema_version") != "agent1_v5_accelerated_signature_metric_v1":
            continue
        if metric.get("benchmark_plan_sha256") != plan.get("_sha256"):
            continue
        key = (int(metric.get("phase_index", -1)), int(metric.get("rank", -1)))
        if key in found:
            raise ValueError(f"duplicate benchmark metric for phase/rank {key}")
        found[key] = (path, metric)
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing or unexpected:
        raise ValueError(f"benchmark metric closure failed: missing={missing} unexpected={unexpected}")
    grouped: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for key, workers in expected.items():
        path, metric = found[key]
        if metric.get("status") != "passed" or int(metric.get("workers", -1)) != workers:
            raise ValueError(f"failed or worker-drift benchmark metric: {path}")
        for field in ("input_bytes", "input_rows", "elapsed_seconds", "started_epoch", "finished_epoch"):
            if int(metric.get(field, 0)) <= 0:
                raise ValueError(f"benchmark metric has invalid {field}: {path}")
        grouped.setdefault(key[0], []).append((path, metric))
    return grouped, phases


def approve_benchmark(args: argparse.Namespace) -> int:
    """Apply the documented deterministic 1→2→4→5 selection gates."""

    plan = _validate_receipt(args.benchmark_plan, BENCHMARK_PLAN_SCHEMA)
    plan = dict(plan)
    plan["_sha256"] = sha256_file(args.benchmark_plan)
    grouped, phases = _phase_metrics(plan, args.metrics_root)
    observations = _passed(args.observations, label="benchmark observations")
    if observations.get("schema_version") != "agent1_v5_dedup_benchmark_observations_v1":
        raise ValueError("unsupported benchmark observations schema")
    if observations.get("benchmark_plan_sha256") != plan["_sha256"]:
        raise ValueError("benchmark observations do not bind benchmark plan")
    observed_by_phase = {
        int(item["phase_index"]): item for item in observations.get("phases", []) if isinstance(item, Mapping)
    }
    if set(observed_by_phase) != {int(phase["index"]) for phase in phases}:
        raise ValueError("benchmark observations lack a phase")

    summaries: dict[int, dict[str, Any]] = {}
    universal: dict[int, bool] = {}
    for phase in phases:
        index = int(phase["index"])
        metrics = [value for _, value in grouped[index]]
        started = min(int(metric["started_epoch"]) for metric in metrics)
        finished = max(int(metric["finished_epoch"]) for metric in metrics)
        wall = finished - started
        if wall <= 0:
            raise ValueError(f"benchmark phase {index} has non-positive wall time")
        bytes_total = sum(int(metric["input_bytes"]) for metric in metrics)
        normalized = [float(metric["elapsed_seconds"]) / int(metric["input_bytes"]) for metric in metrics]
        observation = observed_by_phase[index]
        limits_ok = (
            int(observation.get("sample_count", 0)) >= 2
            and int(observation.get("aggregate_rss_bytes", -1)) < 32 * 1024**3
            and float(observation.get("read_peak_5m_bps", float("inf"))) < 1024**3
            and float(observation.get("write_peak_5m_bps", float("inf"))) < 500 * 1024**2
            and float(observation.get("max_cpu_cores", float("inf"))) <= 32.0
            and observation.get("warnings", []) == []
            and observation.get("errors", []) == []
        )
        universal[index] = limits_ok
        summaries[index] = {
            "name": phase["name"],
            "workers": int(phase["workers"]),
            "ranks": list(phase["ranks"]),
            "metric_bindings": [_binding(path) for path, _ in grouped[index]],
            "wall_seconds": wall,
            "input_bytes": bytes_total,
            "aggregate_bytes_per_second": bytes_total / wall,
            "median_seconds_per_input_byte": statistics.median(normalized),
            "universal_gates_passed": limits_ok,
            "observation": observation,
        }

    baseline = summaries[0]
    if not universal[0]:
        raise ValueError("baseline universal gates failed; refusing acceleration decision")
    speedup_2 = summaries[1]["aggregate_bytes_per_second"] / baseline["aggregate_bytes_per_second"]
    speedup_4 = summaries[2]["aggregate_bytes_per_second"] / baseline["aggregate_bytes_per_second"]
    speedup_5 = summaries[3]["aggregate_bytes_per_second"] / baseline["aggregate_bytes_per_second"]
    normalized_limit = baseline["median_seconds_per_input_byte"] * 1.15
    two_diagnostic = universal[1] and speedup_2 >= 1.70
    four_eligible = universal[2] and speedup_4 >= 3.40 and summaries[2]["median_seconds_per_input_byte"] <= normalized_limit
    five_eligible = universal[3] and speedup_5 >= 4.25 and summaries[3]["median_seconds_per_input_byte"] <= normalized_limit
    if not two_diagnostic:
        approved, selected_workers, reason = False, None, "two-worker diagnostic missed 1.70x or a universal gate"
    elif five_eligible:
        approved, selected_workers, reason = True, 5, "five-worker phase met all gates and 4.25x target"
    elif four_eligible:
        approved, selected_workers, reason = True, 4, "four-worker phase met all gates; five-worker phase was not eligible"
    else:
        approved, selected_workers, reason = False, None, "four-worker phase was not eligible; resume legacy rollback path"
    value: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "benchmark_plan_sha256": plan["_sha256"],
        "observations_sha256": sha256_file(args.observations),
        "approved": approved,
        "selected_workers": selected_workers,
        "selection_reason": reason,
        "speedups": {"two": speedup_2, "four": speedup_4, "five": speedup_5},
        "normalized_elapsed_limit_seconds_per_byte": normalized_limit,
        "two_worker_diagnostic_passed": two_diagnostic,
        "four_worker_eligible": four_eligible,
        "five_worker_eligible": five_eligible,
        "phases": [summaries[index] for index in sorted(summaries)],
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "approved": approved, "selected_workers": selected_workers}))
    return 0


def validate_worker_authorization(args: argparse.Namespace) -> int:
    """Reject an array task unless a held submission was fully authorized."""

    submission = _validate_receipt(args.submission_receipt, SUBMISSION_SCHEMA)
    authorization = _validate_receipt(args.release_authorization, RELEASE_AUTHORIZATION_SCHEMA)
    job_id = str(args.array_job_id)
    if not job_id.isdigit() or submission.get("array_job_id") != job_id:
        raise ValueError("array job ID does not match immutable submission")
    if authorization.get("submission_receipt_sha256") != sha256_file(args.submission_receipt):
        raise ValueError("release authorization does not bind immutable submission")
    expected = {
        "array_job_id": job_id,
        "submission_nonce": args.submission_nonce,
        "chunk_plan_sha256": args.chunk_plan_sha256,
        "runner_sha256": sha256_file(args.runner),
        "attempt_id": args.attempt_id,
        "recovery_receipt_sha256": args.recovery_receipt_sha256,
        "selected_workers": int(args.workers),
    }
    for key, value in expected.items():
        if submission.get(key) != value or authorization.get(key) != value:
            raise ValueError(f"submission/authorization binding drift: {key}")
    print(canonical_json({"ok": True, "array_job_id": job_id}))
    return 0


def record_submission(args: argparse.Namespace) -> int:
    """Record an already identity-checked held normal array submission."""

    benchmark = _validate_receipt(args.benchmark_receipt, BENCHMARK_SCHEMA)
    _validate_receipt(args.cutover_receipt, CUTOVER_SCHEMA)
    audit = _validate_receipt(args.full_input_audit, dedup.FULL_INPUT_AUDIT_SCHEMA)
    plan = _validate_receipt(args.chunk_plan, CHUNK_PLAN_SCHEMA)
    recovery = _validate_receipt(args.recovery_receipt, RECOVERY_SCHEMA)
    observation = _passed(args.job_evidence, label="held array job evidence")
    job_id = str(args.array_job_id)
    if not job_id.isdigit():
        raise ValueError("array job ID must be numeric")
    workers = int(benchmark.get("selected_workers", -1))
    if workers not in (4, 5) or plan.get("selected_workers") != workers:
        raise ValueError("benchmark and chunk-plan worker selection drift")
    bindings = {
        "benchmark_receipt_sha256": sha256_file(args.benchmark_receipt),
        "cutover_receipt_sha256": sha256_file(args.cutover_receipt),
        "full_input_audit_sha256": sha256_file(args.full_input_audit),
        "chunk_plan_sha256": sha256_file(args.chunk_plan),
        "runner_sha256": sha256_file(args.runner),
    }
    if recovery.get("run_root") != str(args.run_root.resolve()):
        raise ValueError("recovery receipt run-root drift")
    if recovery.get("new_runner_sha256") != bindings["runner_sha256"]:
        raise ValueError("recovery receipt corrected-runner drift")
    if plan.get("benchmark_receipt_sha256") != bindings["benchmark_receipt_sha256"]:
        raise ValueError("chunk plan benchmark binding drift")
    if plan.get("cutover_receipt_sha256") != bindings["cutover_receipt_sha256"]:
        raise ValueError("chunk plan cutover binding drift")
    if plan.get("full_input_audit_sha256") != bindings["full_input_audit_sha256"]:
        raise ValueError("chunk plan audit binding drift")
    if plan.get("runner_sha256") != bindings["runner_sha256"]:
        raise ValueError("chunk plan runner binding drift")
    if plan.get("combined_manifest_sha256") != audit.get("combined_manifest_sha256"):
        raise ValueError("chunk plan manifest binding drift")
    if plan.get("attempt_id") != args.attempt_id:
        raise ValueError("attempt identity drift")
    if plan.get("recovery_receipt_sha256") != sha256_file(args.recovery_receipt):
        raise ValueError("chunk plan recovery binding drift")
    if str(observation.get("array_job_id")) != job_id or observation.get("submission_nonce") != args.submission_nonce:
        raise ValueError("held array evidence identity drift")
    if observation.get("attempt_id") != args.attempt_id:
        raise ValueError("held array evidence attempt drift")
    if observation.get("state") != "PENDING" or observation.get("reason") != "JobHeldUser":
        raise ValueError("array is not user-held")
    expected_identity = {
        "owner": "fffoivos",
        "account": "a0140",
        "partition": "normal",
        "job_name": f"a1v5-signature-normal-c{workers}",
        "array_spec": args.array_spec,
        "coord_root": str(args.coord_root.resolve()),
    }
    for key, expected in expected_identity.items():
        if observation.get(key) != expected:
            raise ValueError(f"held array evidence drift for {key}")
    value: dict[str, Any] = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "run_root": str(args.run_root.resolve()),
        "array_job_id": job_id,
        "submission_nonce": args.submission_nonce,
        "attempt_id": args.attempt_id,
        "recovery_receipt_sha256": sha256_file(args.recovery_receipt),
        "selected_workers": workers,
        "array_spec": args.array_spec,
        "job_evidence_sha256": sha256_file(args.job_evidence),
        **bindings,
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "array_job_id": job_id, "held": True}))
    return 0


def authorize_release(args: argparse.Namespace) -> int:
    """Write the final immutable authorization before releasing a held array."""

    submission = _validate_receipt(args.submission_receipt, SUBMISSION_SCHEMA)
    job_id = str(args.array_job_id)
    if submission.get("array_job_id") != job_id or submission.get("submission_nonce") != args.submission_nonce:
        raise ValueError("submission receipt job or nonce drift")
    value: dict[str, Any] = {
        "schema_version": RELEASE_AUTHORIZATION_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "submission_receipt_sha256": sha256_file(args.submission_receipt),
        "array_job_id": job_id,
        "submission_nonce": args.submission_nonce,
        "attempt_id": submission["attempt_id"],
        "recovery_receipt_sha256": submission["recovery_receipt_sha256"],
        "chunk_plan_sha256": submission["chunk_plan_sha256"],
        "runner_sha256": submission["runner_sha256"],
        "selected_workers": submission["selected_workers"],
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "array_job_id": job_id, "authorized": True}))
    return 0


def make_chunk_plan(args: argparse.Namespace) -> int:
    benchmark = _validate_receipt(args.benchmark_receipt, BENCHMARK_SCHEMA)
    workers = int(benchmark.get("selected_workers", -1))
    if benchmark.get("status") != "passed" or benchmark.get("approved") is not True or workers not in (4, 5):
        raise ValueError("benchmark receipt does not select four or five approved workers")
    boundary = _validate_receipt(args.cutover_receipt, CUTOVER_SCHEMA)
    audit = _validate_receipt(args.full_input_audit, dedup.FULL_INPUT_AUDIT_SCHEMA)
    recovery = _validate_receipt(args.recovery_receipt, RECOVERY_SCHEMA)
    attempt_id = _validate_identifier(args.attempt_id, label="attempt ID")
    first = int(boundary["first_missing_rank"])
    last = int(args.last_rank)
    chunk_size = int(args.chunk_size)
    if last < first or not 1 <= chunk_size <= 60:
        raise ValueError("invalid remaining rank range or chunk size")
    run_root = args.run_root.resolve()
    if recovery.get("run_root") != str(run_root):
        raise ValueError("recovery receipt run drift")
    if recovery.get("new_runner_sha256") != sha256_file(args.runner):
        raise ValueError("recovery receipt does not bind corrected runner")
    if recovery.get("new_pipeline_root") != str(args.pipeline_root.resolve()):
        raise ValueError("recovery receipt does not bind corrected pipeline")
    predecessor_execution_sha256 = None
    if args.predecessor_execution is not None:
        predecessor = _validate_receipt(args.predecessor_execution, EXECUTION_SCHEMA)
        if predecessor.get("run_root") != str(run_root):
            raise ValueError("predecessor execution run-root drift")
        if predecessor.get("recovery_receipt_sha256") != sha256_file(args.recovery_receipt):
            raise ValueError("predecessor execution recovery binding drift")
        predecessor_ranks = predecessor.get("ranks")
        if (
            predecessor.get("rank_count") != 1
            or not isinstance(predecessor_ranks, list)
            or len(predecessor_ranks) != 1
            or int(predecessor_ranks[0].get("rank", -1)) != first
        ):
            raise ValueError("predecessor execution is not the single boundary-rank canary")
        canary_receipt = run_root / "60-dedup" / "minhash-signatures" / "receipts" / f"{first:06d}.json"
        if predecessor_ranks[0].get("receipt_sha256") != sha256_file(canary_receipt):
            raise ValueError("predecessor canary receipt changed after execution validation")
        predecessor_execution_sha256 = sha256_file(args.predecessor_execution)
    elif last > first:
        raise ValueError("multi-rank production planning requires a passed boundary-rank canary")
    pending: list[int] = []
    reused: list[dict[str, Any]] = []
    for rank in range(first, last + 1):
        receipt = run_root / "60-dedup" / "minhash-signatures" / "receipts" / f"{rank:06d}.json"
        if receipt.exists():
            completed = _read(receipt)
            if completed.get("schema_version") != dedup.SIGNATURE_RECEIPT_SCHEMA or completed.get("status") != "passed":
                raise ValueError(f"unexpected incomplete signature receipt: {receipt}")
            outputs = completed.get("outputs")
            if not isinstance(outputs, list) or len(outputs) != 32:
                raise ValueError(f"completed signature receipt has incomplete outputs: {receipt}")
            for output in outputs:
                dedup.validate_file_receipt(output, root=run_root)
            reused.append({"rank": rank, "receipt_sha256": sha256_file(receipt)})
        else:
            pending.append(rank)
    if not pending:
        raise ValueError("no unsigned ranks remain after benchmark")
    chunks = []
    for index, start in enumerate(range(0, len(pending), chunk_size)):
        chunks.append({"index": index, "ranks": pending[start : start + chunk_size]})
    value: dict[str, Any] = {
        "schema_version": CHUNK_PLAN_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "first_rank": first,
        "last_rank": last,
        "last_chunk": len(chunks) - 1,
        "chunk_size": chunk_size,
        "selected_workers": workers,
        "attempt_id": attempt_id,
        "recovery_receipt_sha256": sha256_file(args.recovery_receipt),
        "predecessor_execution_sha256": predecessor_execution_sha256,
        "benchmark_receipt_sha256": sha256_file(args.benchmark_receipt),
        "cutover_receipt_sha256": sha256_file(args.cutover_receipt),
        "full_input_audit_sha256": sha256_file(args.full_input_audit),
        "combined_manifest_sha256": audit["combined_manifest_sha256"],
        "deployed_code_root": str(args.pipeline_root.resolve()),
        "runner_sha256": sha256_file(args.runner),
        "reused_benchmark_or_completed_ranks": reused,
        "chunks": chunks,
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "chunks": len(chunks), "workers": workers, "ranks": len(pending), "reused": len(reused)}))
    return 0


def validate_attempt_execution(args: argparse.Namespace) -> int:
    submission = _validate_receipt(args.submission_receipt, SUBMISSION_SCHEMA)
    authorization = _validate_receipt(args.release_authorization, RELEASE_AUTHORIZATION_SCHEMA)
    plan = _validate_receipt(args.chunk_plan, CHUNK_PLAN_SCHEMA)
    evidence = _validate_receipt(args.scheduler_evidence, ARRAY_EXECUTION_EVIDENCE_SCHEMA)
    job_id = str(submission.get("array_job_id", ""))
    if not job_id.isdigit() or evidence.get("array_job_id") != job_id or evidence.get("attempt_id") != submission.get("attempt_id"):
        raise ValueError("execution evidence job identity drift")
    authorization_bindings = {
        "submission_receipt_sha256": sha256_file(args.submission_receipt),
        "array_job_id": job_id,
        "submission_nonce": submission.get("submission_nonce"),
        "attempt_id": submission.get("attempt_id"),
        "recovery_receipt_sha256": submission.get("recovery_receipt_sha256"),
        "chunk_plan_sha256": submission.get("chunk_plan_sha256"),
        "runner_sha256": submission.get("runner_sha256"),
        "selected_workers": submission.get("selected_workers"),
    }
    for key, expected in authorization_bindings.items():
        if authorization.get(key) != expected:
            raise ValueError(f"execution authorization binding drift: {key}")
    if submission.get("chunk_plan_sha256") != sha256_file(args.chunk_plan):
        raise ValueError("execution submission does not bind chunk plan")
    if submission.get("attempt_id") != plan.get("attempt_id"):
        raise ValueError("execution attempt identity drift")
    if submission.get("run_root") != str(args.run_root.resolve()):
        raise ValueError("execution submission run-root drift")
    if submission.get("recovery_receipt_sha256") != plan.get("recovery_receipt_sha256"):
        raise ValueError("execution recovery binding drift")
    expected_tasks = _array_indices(str(submission.get("array_spec", "")))
    if (
        evidence.get("array_spec") != submission.get("array_spec")
        or evidence.get("expected_state") != "COMPLETED"
        or evidence.get("expected_exit_code") != "0:0"
    ):
        raise ValueError("execution scheduler evidence binding drift")
    tasks = evidence.get("tasks")
    if not isinstance(tasks, list) or sorted(int(task.get("task_index", -1)) for task in tasks) != expected_tasks:
        raise ValueError("execution evidence does not close every array task")
    for task in tasks:
        if task.get("state") != "COMPLETED" or task.get("exit_code") != "0:0":
            raise ValueError("array execution did not complete successfully")
        if task.get("account") != "a0140" or task.get("partition") != "normal":
            raise ValueError("array execution scheduler identity drift")
        if task.get("job_id") != f"{job_id}_{int(task['task_index'])}":
            raise ValueError("array execution scheduler task ID drift")
    chunks = plan.get("chunks")
    if not isinstance(chunks, list) or [int(chunk.get("index", -1)) for chunk in chunks] != expected_tasks:
        raise ValueError("execution chunk indices do not match scheduler array")
    raw_expected_ranks = [
        int(rank)
        for chunk in chunks
        for rank in chunk.get("ranks", [])
    ]
    expected_ranks = sorted(raw_expected_ranks)
    if not expected_ranks or len(set(expected_ranks)) != len(expected_ranks):
        raise ValueError("execution chunk plan has empty or duplicate ranks")
    run_root = args.run_root.resolve()
    receipt_root = run_root / "60-dedup" / "minhash-signatures" / "receipts"
    metric_root = args.metrics_root.resolve()
    all_metric_paths = list(metric_root.glob("*.json"))
    if len(all_metric_paths) != len(expected_ranks):
        raise ValueError("attempt metric directory does not exactly close planned ranks")
    metrics: list[dict[str, Any]] = []
    for rank in expected_ranks:
        receipt_path = receipt_root / f"{rank:06d}.json"
        receipt = _read(receipt_path)
        if (
            receipt.get("schema_version") != dedup.SIGNATURE_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or int(receipt.get("task_index", -1)) != rank
        ):
            raise ValueError(f"attempt rank {rank} lacks a passed receipt")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 32:
            raise ValueError(f"attempt rank {rank} has incomplete output closure")
        for output in outputs:
            dedup.validate_file_receipt(output, root=run_root)
        matches = list(metric_root.glob(f"*-rank-{rank:06d}.json"))
        if len(matches) != 1:
            raise ValueError(f"attempt rank {rank} does not have exactly one metric")
        metric = _read(matches[0])
        if (
            metric.get("status") != "passed"
            or metric.get("attempt_id") != submission.get("attempt_id")
            or metric.get("array_job_id") != job_id
            or metric.get("submission_nonce") != submission.get("submission_nonce")
            or metric.get("chunk_plan_sha256") != sha256_file(args.chunk_plan)
            or metric.get("benchmark_plan_sha256") is not None
            or int(metric.get("rank", -1)) != rank
        ):
            raise ValueError(f"attempt metric binding drift for rank {rank}")
        metrics.append({"rank": rank, "metric_sha256": sha256_file(matches[0]), "receipt_sha256": sha256_file(receipt_path)})
    value: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "passed",
        "created_at": _now(),
        "attempt_id": submission["attempt_id"],
        "run_root": str(run_root),
        "array_job_id": job_id,
        "submission_nonce": submission["submission_nonce"],
        "recovery_receipt_sha256": submission["recovery_receipt_sha256"],
        "selected_workers": submission["selected_workers"],
        "submission_receipt_sha256": sha256_file(args.submission_receipt),
        "release_authorization_sha256": sha256_file(args.release_authorization),
        "chunk_plan_sha256": sha256_file(args.chunk_plan),
        "scheduler_evidence_sha256": sha256_file(args.scheduler_evidence),
        "rank_count": len(expected_ranks),
        "ranks": metrics,
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "attempt_id": submission["attempt_id"], "ranks": len(expected_ranks)}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("build-preflight")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--pipeline-root", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--runtime-receipt", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--acquisition-audit", type=Path, required=True)
    command.add_argument("--exact-manifest", type=Path, required=True)
    command.add_argument("--publication-pre", type=Path, required=True)
    command.add_argument("--publication-metadata", type=Path, required=True)
    command.add_argument("--evidence", action="append", default=[], metavar="LABEL=PATH")
    command.add_argument("--scheduler-snapshot", type=Path, required=True)
    command.add_argument("--legacy-helper", type=Path, required=True)
    command.add_argument("--approve-fence", action="store_true")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=build_preflight)

    command = commands.add_parser("validate-boundary")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--through-rank", type=int, required=True)
    command.add_argument("--fence-job-id", required=True)
    command.add_argument("--final-legacy-job-id", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=validate_boundary)

    command = commands.add_parser("finalize-cutover")
    command.add_argument("--preflight", type=Path, required=True)
    command.add_argument("--boundary", type=Path, required=True)
    command.add_argument("--fence-evidence", type=Path, required=True)
    command.add_argument("--successor-evidence", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=finalize_cutover)

    command = commands.add_parser("finalize-sentinel-cutover")
    command.add_argument("--request", type=Path, required=True)
    command.add_argument("--arm-receipt", type=Path, required=True)
    command.add_argument("--stop-receipt", type=Path, required=True)
    command.add_argument("--queue-evidence", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=finalize_sentinel_cutover)

    command = commands.add_parser("make-benchmark-plan")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--cutover-receipt", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--rank-count", type=int, default=24)
    command.add_argument("--phase-ranks", action="append", default=[], metavar="RANK[,RANK...]")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=make_benchmark_plan)

    command = commands.add_parser("approve-benchmark")
    command.add_argument("--benchmark-plan", type=Path, required=True)
    command.add_argument("--metrics-root", type=Path, required=True)
    command.add_argument("--observations", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=approve_benchmark)

    command = commands.add_parser("make-chunk-plan")
    command.add_argument("--benchmark-receipt", type=Path, required=True)
    command.add_argument("--cutover-receipt", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--pipeline-root", type=Path, required=True)
    command.add_argument("--runner", type=Path, required=True)
    command.add_argument("--recovery-receipt", type=Path, required=True)
    command.add_argument("--attempt-id", required=True)
    command.add_argument("--predecessor-execution", type=Path)
    command.add_argument("--last-rank", type=int, required=True)
    command.add_argument("--chunk-size", type=int, default=60)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=make_chunk_plan)

    command = commands.add_parser("validate-worker-authorization")
    command.add_argument("--submission-receipt", type=Path, required=True)
    command.add_argument("--release-authorization", type=Path, required=True)
    command.add_argument("--array-job-id", required=True)
    command.add_argument("--submission-nonce", required=True)
    command.add_argument("--chunk-plan-sha256", required=True)
    command.add_argument("--runner", type=Path, required=True)
    command.add_argument("--workers", type=int, required=True)
    command.add_argument("--attempt-id", required=True)
    command.add_argument("--recovery-receipt-sha256", required=True)
    command.set_defaults(func=validate_worker_authorization)

    command = commands.add_parser("record-submission")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--benchmark-receipt", type=Path, required=True)
    command.add_argument("--cutover-receipt", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--chunk-plan", type=Path, required=True)
    command.add_argument("--recovery-receipt", type=Path, required=True)
    command.add_argument("--attempt-id", required=True)
    command.add_argument("--runner", type=Path, required=True)
    command.add_argument("--job-evidence", type=Path, required=True)
    command.add_argument("--array-job-id", required=True)
    command.add_argument("--submission-nonce", required=True)
    command.add_argument("--array-spec", required=True)
    command.add_argument("--coord-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=record_submission)

    command = commands.add_parser("authorize-release")
    command.add_argument("--submission-receipt", type=Path, required=True)
    command.add_argument("--array-job-id", required=True)
    command.add_argument("--submission-nonce", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=authorize_release)

    command = commands.add_parser("record-failed-array-recovery")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--failed-submission", type=Path, required=True)
    command.add_argument("--failed-release-authorization", type=Path, required=True)
    command.add_argument("--failed-release-observation", type=Path, required=True)
    command.add_argument("--failed-chunk-plan", type=Path, required=True)
    command.add_argument("--scheduler-evidence", type=Path, required=True)
    command.add_argument("--metrics-root", type=Path, required=True)
    command.add_argument("--expected-receipt-count", type=int, required=True)
    command.add_argument("--new-pipeline-root", type=Path, required=True)
    command.add_argument("--new-runner", type=Path, required=True)
    command.add_argument("--recovery-id", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=record_failed_array_recovery)

    command = commands.add_parser("validate-attempt-execution")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--submission-receipt", type=Path, required=True)
    command.add_argument("--release-authorization", type=Path, required=True)
    command.add_argument("--chunk-plan", type=Path, required=True)
    command.add_argument("--scheduler-evidence", type=Path, required=True)
    command.add_argument("--metrics-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=validate_attempt_execution)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

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
    phases = []
    cursor = 0
    for index, (workers, count, name) in enumerate(
        ((1, 2, "baseline"), (2, 4, "two_worker"), (4, 8, "four_worker"), (5, 10, "five_worker"))
    ):
        ranks = [int(row["rank"]) for row in selected[cursor : cursor + count]]
        phases.append({"index": index, "name": name, "workers": workers, "ranks": ranks})
        cursor += count
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
        "rank_inventory": [
            {"rank": row["rank"], "bytes": row["bytes"], "rows": row["rows"], "sha256": row["sha256"]}
            for row in selected
        ],
    }
    _write_immutable(args.output, value)
    print(canonical_json({"ok": True, "phases": len(phases), "ranks": rank_count}))
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
    expected = {
        "submission_receipt_sha256": sha256_file(args.submission_receipt),
        "array_job_id": job_id,
        "submission_nonce": args.submission_nonce,
        "chunk_plan_sha256": args.chunk_plan_sha256,
        "runner_sha256": sha256_file(args.runner),
    }
    for key, value in expected.items():
        if submission.get(key) != value and authorization.get(key) != value:
            raise ValueError(f"submission/authorization binding drift: {key}")
    if int(submission.get("selected_workers", -1)) != int(args.workers):
        raise ValueError("worker count differs from submission authorization")
    print(canonical_json({"ok": True, "array_job_id": job_id}))
    return 0


def record_submission(args: argparse.Namespace) -> int:
    """Record an already identity-checked held normal array submission."""

    benchmark = _validate_receipt(args.benchmark_receipt, BENCHMARK_SCHEMA)
    cutover = _validate_receipt(args.cutover_receipt, CUTOVER_SCHEMA)
    audit = _validate_receipt(args.full_input_audit, dedup.FULL_INPUT_AUDIT_SCHEMA)
    plan = _validate_receipt(args.chunk_plan, CHUNK_PLAN_SCHEMA)
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
    if str(observation.get("array_job_id")) != job_id or observation.get("submission_nonce") != args.submission_nonce:
        raise ValueError("held array evidence identity drift")
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
    first = int(boundary["first_missing_rank"])
    last = int(args.last_rank)
    chunk_size = int(args.chunk_size)
    if last < first or not 1 <= chunk_size <= 60:
        raise ValueError("invalid remaining rank range or chunk size")
    run_root = args.run_root.resolve()
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

    command = commands.add_parser("make-benchmark-plan")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--cutover-receipt", type=Path, required=True)
    command.add_argument("--combined-manifest", type=Path, required=True)
    command.add_argument("--rank-count", type=int, default=24)
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
    command.set_defaults(func=validate_worker_authorization)

    command = commands.add_parser("record-submission")
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--benchmark-receipt", type=Path, required=True)
    command.add_argument("--cutover-receipt", type=Path, required=True)
    command.add_argument("--full-input-audit", type=Path, required=True)
    command.add_argument("--chunk-plan", type=Path, required=True)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

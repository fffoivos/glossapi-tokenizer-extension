from __future__ import annotations

import json
import shlex
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import agent1_v5_datatrove as dedup  # noqa: E402
import agent1_v5_dedup_acceleration as acceleration  # noqa: E402


TAKEOVER_SPEC = importlib.util.spec_from_file_location(
    "agent1_v5_signature_takeover", ROOT / "scripts" / "agent1_v5_signature_takeover.py"
)
assert TAKEOVER_SPEC and TAKEOVER_SPEC.loader
takeover = importlib.util.module_from_spec(TAKEOVER_SPEC)
sys.modules[TAKEOVER_SPEC.name] = takeover
TAKEOVER_SPEC.loader.exec_module(takeover)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def binding(path: Path, root: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"signature")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": dedup.sha256_file(path),
    }


def test_boundary_requires_all_32_receipted_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output_root = run / "60-dedup" / "minhash-signatures"
    outputs = [binding(output_root / f"bucket_{bucket:03d}" / "00000.minhash.sig", run) for bucket in range(32)]
    receipt = {
        "schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA,
        "status": "passed",
        "task_index": 0,
        "outputs": outputs,
    }
    write_json(output_root / "receipts" / "000000.json", receipt)
    manifest = tmp_path / "combined.json"
    write_json(manifest, {"files": [{"rank": 0}, {"rank": 1}]})
    target = tmp_path / "boundary.json"

    assert acceleration.main(
        [
            "validate-boundary",
            "--run-root",
            str(run),
            "--combined-manifest",
            str(manifest),
            "--through-rank",
            "0",
            "--fence-job-id",
            "123",
            "--final-legacy-job-id",
            "122",
            "--output",
            str(target),
        ]
    ) == 0
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["first_missing_rank"] == 1
    assert value["legacy_receipt_count"] == 1

    (output_root / "bucket_031" / "00000.minhash.sig").unlink()
    target.unlink()
    try:
        acceleration.main(
            [
                "validate-boundary",
                "--run-root",
                str(run),
                "--combined-manifest",
                str(manifest),
                "--through-rank",
                "0",
                "--fence-job-id",
                "123",
                "--final-legacy-job-id",
                "122",
                "--output",
                str(target),
            ]
        )
    except ValueError as error:
        assert "file receipt mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing signature output was accepted")


def test_chunk_plan_requires_approved_four_or_five_worker_benchmark(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "schema_version": dedup.FULL_INPUT_AUDIT_SCHEMA,
            "status": "passed",
            "combined_manifest_sha256": "manifest",
        },
    )
    boundary = tmp_path / "boundary.json"
    write_json(
        boundary,
        {
            "schema_version": acceleration.CUTOVER_SCHEMA,
            "status": "passed",
            "first_missing_rank": 7,
        },
    )
    benchmark = tmp_path / "benchmark.json"
    write_json(
        benchmark,
        {
            "schema_version": acceleration.BENCHMARK_SCHEMA,
            "status": "passed",
            "approved": True,
            "selected_workers": 5,
        },
    )
    output = tmp_path / "chunks.json"
    run_root = tmp_path / "run"
    attempt_id = "attempt-002-canary"
    runner = ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh"
    recovery = tmp_path / "recovery.json"
    write_json(
        recovery,
        {
            "schema_version": acceleration.RECOVERY_SCHEMA,
            "status": "passed",
            "run_root": str(run_root.resolve()),
            "recovery_id": "recovery-002",
            "new_runner_sha256": dedup.sha256_file(runner),
            "new_pipeline_root": str(ROOT.resolve()),
            "failed_pending_ranks": list(range(7, 15)),
        },
    )
    canary_outputs = [
        binding(run_root / "60-dedup" / "minhash-signatures" / f"bucket_{bucket:03d}" / "00007.sig", run_root)
        for bucket in range(32)
    ]
    canary_receipt = run_root / "60-dedup" / "minhash-signatures" / "receipts" / "000007.json"
    write_json(
        canary_receipt,
        {"schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA, "status": "passed", "task_index": 7, "outputs": canary_outputs},
    )
    predecessor = tmp_path / "canary-execution.json"
    write_json(
        predecessor,
        {
            "schema_version": acceleration.EXECUTION_SCHEMA,
            "status": "passed",
            "run_root": str(run_root.resolve()),
            "recovery_receipt_sha256": dedup.sha256_file(recovery),
            "rank_count": 1,
            "ranks": [{"rank": 7, "receipt_sha256": dedup.sha256_file(canary_receipt)}],
        },
    )

    assert acceleration.main(
        [
            "make-chunk-plan",
            "--benchmark-receipt",
            str(benchmark),
            "--cutover-receipt",
            str(boundary),
            "--full-input-audit",
            str(audit),
            "--run-root",
            str(run_root),
            "--pipeline-root",
            str(ROOT),
            "--runner",
            str(runner),
            "--recovery-receipt",
            str(recovery),
            "--attempt-id",
            attempt_id,
            "--predecessor-execution",
            str(predecessor),
            "--last-rank",
            "14",
            "--chunk-size",
            "3",
            "--output",
            str(output),
        ]
    ) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["selected_workers"] == 5
    assert value["attempt_id"] == attempt_id
    assert value["recovery_receipt_sha256"] == dedup.sha256_file(recovery)
    assert value["canary_rank"] == 7
    assert value["chunks"] == [
        {"index": 0, "ranks": [8, 9, 10]},
        {"index": 1, "ranks": [11, 12, 13]},
        {"index": 2, "ranks": [14]},
    ]


def test_canary_plan_uses_first_recovery_pending_rank_after_sparse_benchmark_receipts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    runner = ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh"
    manifest = tmp_path / "combined.json"
    write_json(manifest, {"files": [{"rank": rank} for rank in range(12)]})
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "schema_version": dedup.FULL_INPUT_AUDIT_SCHEMA,
            "status": "passed",
            "combined_manifest_sha256": dedup.sha256_file(manifest),
        },
    )
    cutover = tmp_path / "cutover.json"
    write_json(
        cutover,
        {
            "schema_version": acceleration.CUTOVER_SCHEMA,
            "status": "passed",
            "first_missing_rank": 7,
            "combined_manifest_sha256": dedup.sha256_file(manifest),
        },
    )
    benchmark = tmp_path / "benchmark.json"
    write_json(
        benchmark,
        {"schema_version": acceleration.BENCHMARK_SCHEMA, "status": "passed", "approved": True, "selected_workers": 5},
    )
    for rank in (7, 8, 9):
        outputs = [
            binding(run_root / "60-dedup" / "minhash-signatures" / f"bucket_{bucket:03d}" / f"{rank:05d}.sig", run_root)
            for bucket in range(32)
        ]
        write_json(
            run_root / "60-dedup" / "minhash-signatures" / "receipts" / f"{rank:06d}.json",
            {"schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA, "status": "passed", "task_index": rank, "outputs": outputs},
        )
    recovery = tmp_path / "recovery.json"
    write_json(
        recovery,
        {
            "schema_version": acceleration.RECOVERY_SCHEMA,
            "status": "passed",
            "run_root": str(run_root.resolve()),
            "new_runner_sha256": dedup.sha256_file(runner),
            "new_pipeline_root": str(ROOT.resolve()),
            "failed_pending_ranks": [10, 11],
        },
    )
    output = tmp_path / "canary-chunks.json"
    assert acceleration.main(
        [
            "make-chunk-plan",
            "--benchmark-receipt", str(benchmark),
            "--cutover-receipt", str(cutover),
            "--full-input-audit", str(audit),
            "--run-root", str(run_root),
            "--pipeline-root", str(ROOT),
            "--runner", str(runner),
            "--recovery-receipt", str(recovery),
            "--attempt-id", "attempt-003-canary",
            "--last-rank", "10",
            "--chunk-size", "1",
            "--output", str(output),
        ]
    ) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["first_rank"] == 7
    assert value["canary_rank"] == 10
    assert value["chunks"] == [{"index": 0, "ranks": [10]}]
    assert [entry["rank"] for entry in value["reused_benchmark_or_completed_ranks"]] == [7, 8, 9]


def test_worker_authorization_rejects_drift_in_either_receipt(tmp_path: Path) -> None:
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    expected = {
        "array_job_id": "123",
        "submission_nonce": "nonce-123456789",
        "chunk_plan_sha256": "plan-sha",
        "runner_sha256": dedup.sha256_file(runner),
        "attempt_id": "attempt-002-canary",
        "recovery_receipt_sha256": "recovery-sha",
    }
    submission = tmp_path / "submission.json"
    authorization = tmp_path / "authorization.json"
    write_json(submission, {"schema_version": acceleration.SUBMISSION_SCHEMA, "status": "passed", "selected_workers": 5, **expected})
    write_json(
        authorization,
        {
            "schema_version": acceleration.RELEASE_AUTHORIZATION_SCHEMA,
            "status": "passed",
            "submission_receipt_sha256": dedup.sha256_file(submission),
            "selected_workers": 5,
            **expected,
        },
    )
    args = [
        "validate-worker-authorization",
        "--submission-receipt", str(submission),
        "--release-authorization", str(authorization),
        "--array-job-id", "123",
        "--submission-nonce", "nonce-123456789",
        "--chunk-plan-sha256", "plan-sha",
        "--runner", str(runner),
        "--workers", "5",
        "--attempt-id", "attempt-002-canary",
        "--recovery-receipt-sha256", "recovery-sha",
    ]
    assert acceleration.main(args) == 0
    value = json.loads(authorization.read_text(encoding="utf-8"))
    value["runner_sha256"] = "drift"
    write_json(authorization := tmp_path / "authorization-drift.json", value)
    drift_args = list(args)
    drift_args[drift_args.index(str(tmp_path / "authorization.json"))] = str(authorization)
    with pytest.raises(ValueError, match="binding drift"):
        acceleration.main(drift_args)


def test_failed_array_recovery_proves_prework_failure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    outputs = [binding(run / "60-dedup" / "minhash-signatures" / f"bucket_{bucket:03d}" / "00000.sig", run) for bucket in range(32)]
    write_json(
        run / "60-dedup" / "minhash-signatures" / "receipts" / "000000.json",
        {"schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA, "status": "passed", "task_index": 0, "outputs": outputs},
    )
    old_runner = tmp_path / "old-runner.sh"
    new_runner = tmp_path / "new-runner.sh"
    old_runner.write_text("old", encoding="utf-8")
    new_runner.write_text("new", encoding="utf-8")
    plan = tmp_path / "failed-plan.json"
    write_json(
        plan,
        {"schema_version": acceleration.CHUNK_PLAN_SCHEMA, "status": "passed", "last_rank": 2, "chunks": [{"index": 0, "ranks": [1]}, {"index": 1, "ranks": [2]}]},
    )
    submission = tmp_path / "failed-submission.json"
    write_json(
        submission,
        {
            "schema_version": acceleration.SUBMISSION_SCHEMA,
            "status": "passed",
            "array_job_id": "123",
            "run_root": str(run.resolve()),
            "array_spec": "0-1%1",
            "submission_nonce": "nonce-123456789",
            "chunk_plan_sha256": dedup.sha256_file(plan),
            "runner_sha256": dedup.sha256_file(old_runner),
        },
    )
    authorization = tmp_path / "failed-authorization.json"
    write_json(
        authorization,
        {
            "schema_version": acceleration.RELEASE_AUTHORIZATION_SCHEMA,
            "status": "passed",
            "array_job_id": "123",
            "submission_nonce": "nonce-123456789",
            "submission_receipt_sha256": dedup.sha256_file(submission),
        },
    )
    release = tmp_path / "failed-release.json"
    write_json(release, {"schema_version": "agent1_v5_dedup_acceleration_release_observation_v1", "status": "passed", "array_job_id": "123", "submission_nonce": "nonce-123456789", "release_requested": True})
    scheduler = tmp_path / "scheduler.json"
    tasks = [
        {"job_id": f"123_{index}", "task_index": index, "state": "FAILED", "exit_code": "127:0", "account": "a0140", "partition": "normal"}
        for index in (0, 1)
    ]
    write_json(
        scheduler,
        {
            "schema_version": acceleration.ARRAY_EXECUTION_EVIDENCE_SCHEMA,
            "status": "passed",
            "array_job_id": "123",
            "attempt_id": "recovery-002",
            "array_spec": "0-1%1",
            "expected_state": "FAILED",
            "expected_exit_code": "127:0",
            "tasks": tasks,
        },
    )
    metrics = run / "60-dedup" / "minhash-signatures" / "accelerated-metrics"
    metrics.mkdir(parents=True)
    output = tmp_path / "recovery.json"
    args = [
        "record-failed-array-recovery",
        "--run-root", str(run),
        "--failed-submission", str(submission),
        "--failed-release-authorization", str(authorization),
        "--failed-release-observation", str(release),
        "--failed-chunk-plan", str(plan),
        "--scheduler-evidence", str(scheduler),
        "--metrics-root", str(metrics),
        "--expected-receipt-count", "1",
        "--new-pipeline-root", str(tmp_path),
        "--new-runner", str(new_runner),
        "--recovery-id", "recovery-002",
        "--output", str(output),
    ]
    assert acceleration.main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["production_metric_count"] == 0
    write_json(metrics / "unexpected.json", {"benchmark_plan_sha256": None})
    retry_args = list(args)
    retry_args[-1] = str(tmp_path / "recovery-with-metric.json")
    with pytest.raises(ValueError, match="production metrics"):
        acceleration.main(retry_args)


def test_attempt_execution_closes_scheduler_authorization_metrics_and_receipt(tmp_path: Path) -> None:
    run = tmp_path / "run"
    outputs = [
        binding(run / "60-dedup" / "minhash-signatures" / f"bucket_{bucket:03d}" / "00007.sig", run)
        for bucket in range(32)
    ]
    receipt = run / "60-dedup" / "minhash-signatures" / "receipts" / "000007.json"
    write_json(
        receipt,
        {"schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA, "status": "passed", "task_index": 7, "outputs": outputs},
    )
    plan = tmp_path / "chunks.json"
    write_json(
        plan,
        {
            "schema_version": acceleration.CHUNK_PLAN_SCHEMA,
            "status": "passed",
            "attempt_id": "attempt-002-canary",
            "recovery_receipt_sha256": "recovery-sha",
            "chunks": [{"index": 0, "ranks": [7]}],
        },
    )
    submission = tmp_path / "submission.json"
    submission_value = {
        "schema_version": acceleration.SUBMISSION_SCHEMA,
        "status": "passed",
        "run_root": str(run.resolve()),
        "array_job_id": "456",
        "array_spec": "0-0%1",
        "submission_nonce": "nonce-123456789",
        "attempt_id": "attempt-002-canary",
        "recovery_receipt_sha256": "recovery-sha",
        "chunk_plan_sha256": dedup.sha256_file(plan),
        "runner_sha256": "runner-sha",
        "selected_workers": 5,
    }
    write_json(submission, submission_value)
    authorization = tmp_path / "authorization.json"
    write_json(
        authorization,
        {
            "schema_version": acceleration.RELEASE_AUTHORIZATION_SCHEMA,
            "status": "passed",
            "submission_receipt_sha256": dedup.sha256_file(submission),
            "array_job_id": "456",
            "submission_nonce": "nonce-123456789",
            "attempt_id": "attempt-002-canary",
            "recovery_receipt_sha256": "recovery-sha",
            "chunk_plan_sha256": dedup.sha256_file(plan),
            "runner_sha256": "runner-sha",
            "selected_workers": 5,
        },
    )
    scheduler = tmp_path / "scheduler.json"
    write_json(
        scheduler,
        {
            "schema_version": acceleration.ARRAY_EXECUTION_EVIDENCE_SCHEMA,
            "status": "passed",
            "array_job_id": "456",
            "array_spec": "0-0%1",
            "attempt_id": "attempt-002-canary",
            "expected_state": "COMPLETED",
            "expected_exit_code": "0:0",
            "tasks": [
                {"job_id": "456_0", "task_index": 0, "state": "COMPLETED", "exit_code": "0:0", "account": "a0140", "partition": "normal"}
            ],
        },
    )
    metrics = tmp_path / "metrics"
    write_json(
        metrics / "chunk-0000-rank-000007.json",
        {
            "status": "passed",
            "rank": 7,
            "attempt_id": "attempt-002-canary",
            "array_job_id": "456",
            "submission_nonce": "nonce-123456789",
            "chunk_plan_sha256": dedup.sha256_file(plan),
            "benchmark_plan_sha256": None,
        },
    )
    execution = tmp_path / "execution.json"
    args = [
        "validate-attempt-execution",
        "--run-root", str(run),
        "--submission-receipt", str(submission),
        "--release-authorization", str(authorization),
        "--chunk-plan", str(plan),
        "--scheduler-evidence", str(scheduler),
        "--metrics-root", str(metrics),
        "--output", str(execution),
    ]
    assert acceleration.main(args) == 0
    value = json.loads(execution.read_text(encoding="utf-8"))
    assert value["run_root"] == str(run.resolve())
    assert value["recovery_receipt_sha256"] == "recovery-sha"
    assert value["ranks"] == [
        {"rank": 7, "metric_sha256": dedup.sha256_file(metrics / "chunk-0000-rank-000007.json"), "receipt_sha256": dedup.sha256_file(receipt)}
    ]
    write_json(metrics / "unexpected.json", {"status": "passed"})
    retry_args = list(args)
    retry_args[-1] = str(tmp_path / "execution-with-extra-metric.json")
    with pytest.raises(ValueError, match="exactly close"):
        acceleration.main(retry_args)


def test_signature_shells_use_single_uenv_and_safe_cleanup() -> None:
    runner = (ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh").read_text(encoding="utf-8")
    submitter = (ROOT / "slurm" / "agent1_v5_eiger" / "submit_accelerated_signature_array.sh").read_text(encoding="utf-8")
    assert runner.count("uenv run pytorch/v2.6.0:v1 --view=default") == 2
    assert '"${uenv_python[@]}" "$acceleration" validate-worker-authorization' in runner
    assert '"$python" "$dedup" accelerated-signature-task' in runner
    assert '"${uenv_python[@]}" "$dedup"' not in runner
    assert 'if [[ "$armed" != 1 || -z "$array_job_id" ]]; then' in submitter
    assert 'scontrol_array_spec="${array_first}%${array_throttle}"' in submitter
    assert '"ArrayTaskId=${scontrol_array_spec} "' in submitter


def test_signature_runner_refills_the_first_available_worker(tmp_path: Path) -> None:
    runner = ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh"
    event_log = tmp_path / "events.log"
    script = f"""
source {shlex.quote(str(runner))}
should_drain() {{ return 1; }}
run_rank() {{
  printf 'start:%s\\n' "$1" >> {shlex.quote(str(event_log))}
  case "$1" in
    1) sleep 0.30 ;;
    *) sleep 0.05 ;;
  esac
  printf 'done:%s\\n' "$1" >> {shlex.quote(str(event_log))}
}}
run_bounded_rank_pool 2 1 2 3
"""
    subprocess.run(["bash", "-c", script], check=True)
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert events.index("start:3") < events.index("done:1")


def test_signature_runner_stops_new_launches_after_worker_failure(tmp_path: Path) -> None:
    runner = ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh"
    event_log = tmp_path / "events.log"
    script = f"""
source {shlex.quote(str(runner))}
should_drain() {{ return 1; }}
run_rank() {{
  printf 'start:%s\\n' "$1" >> {shlex.quote(str(event_log))}
  case "$1" in
    1) sleep 0.15 ;;
    2) sleep 0.03; return 7 ;;
  esac
  printf 'done:%s\\n' "$1" >> {shlex.quote(str(event_log))}
}}
set +e
run_bounded_rank_pool 2 1 2 3
rc=$?
set -e
printf 'rc:%s\\n' "$rc" >> {shlex.quote(str(event_log))}
"""
    subprocess.run(["bash", "-c", script], check=True)
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert "start:3" not in events
    assert "rc:7" in events


def test_benchmark_selection_prefers_five_after_all_gates(tmp_path: Path) -> None:
    phases = [
        {"index": 0, "name": "baseline", "workers": 1, "ranks": [0, 1]},
        {"index": 1, "name": "two_worker", "workers": 2, "ranks": [2, 3, 4, 5]},
        {"index": 2, "name": "four_worker", "workers": 4, "ranks": list(range(6, 14))},
        {"index": 3, "name": "five_worker", "workers": 5, "ranks": list(range(14, 24))},
    ]
    plan = tmp_path / "benchmark-plan.json"
    write_json(
        plan,
        {"schema_version": acceleration.BENCHMARK_PLAN_SCHEMA, "status": "passed", "phases": phases},
    )
    plan_sha = dedup.sha256_file(plan)
    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    for phase in phases:
        for position, rank in enumerate(phase["ranks"]):
            wave = position // phase["workers"]
            start = wave * 100 + 1
            write_json(
                metrics_root / f"metric-{phase['index']}-{rank}.json",
                {
                    "schema_version": "agent1_v5_accelerated_signature_metric_v1",
                    "status": "passed",
                    "benchmark_plan_sha256": plan_sha,
                    "phase_index": phase["index"],
                    "rank": rank,
                    "workers": phase["workers"],
                    "input_bytes": 100,
                    "input_rows": 10,
                    "started_epoch": start,
                    "finished_epoch": start + 100,
                    "elapsed_seconds": 100,
                },
            )
    observations = tmp_path / "observations.json"
    write_json(
        observations,
        {
            "schema_version": "agent1_v5_dedup_benchmark_observations_v1",
            "status": "passed",
            "benchmark_plan_sha256": plan_sha,
            "phases": [
                {
                    "phase_index": phase["index"],
                    "sample_count": 2,
                    "aggregate_rss_bytes": 1,
                    "read_peak_5m_bps": 1,
                    "write_peak_5m_bps": 1,
                    "max_cpu_cores": 1,
                    "warnings": [],
                    "errors": [],
                }
                for phase in phases
            ],
        },
    )
    output = tmp_path / "decision.json"
    assert acceleration.main(
        [
            "approve-benchmark",
            "--benchmark-plan",
            str(plan),
            "--metrics-root",
            str(metrics_root),
            "--observations",
            str(observations),
            "--output",
            str(output),
        ]
    ) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["approved"] is True
    assert value["selected_workers"] == 5


def test_explicit_benchmark_skips_only_partial_nanochat_shard(tmp_path: Path) -> None:
    run = tmp_path / "run"
    manifest = tmp_path / "manifest.json"
    files = [
        {
            "rank": rank,
            "origin": "nanochat_base",
            "rows": 118001 if rank == 76 else 196608,
            "bytes": 10,
            "sha256": f"sha-{rank}",
        }
        for rank in range(68, 93)
    ]
    write_json(manifest, {"files": files})
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {
            "schema_version": dedup.FULL_INPUT_AUDIT_SCHEMA,
            "status": "passed",
            "combined_manifest_sha256": dedup.sha256_file(manifest),
        },
    )
    cutover = tmp_path / "cutover.json"
    write_json(
        cutover,
        {
            "schema_version": acceleration.CUTOVER_SCHEMA,
            "status": "passed",
            "first_missing_rank": 68,
            "combined_manifest_sha256": dedup.sha256_file(manifest),
        },
    )
    output = tmp_path / "benchmark.json"
    assert acceleration.main(
        [
            "make-benchmark-plan",
            "--run-root",
            str(run),
            "--full-input-audit",
            str(audit),
            "--cutover-receipt",
            str(cutover),
            "--combined-manifest",
            str(manifest),
            "--phase-ranks",
            "68,69",
            "--phase-ranks",
            "70,71,72,73",
            "--phase-ranks",
            "74,75,77,78,79,80,81,82",
            "--phase-ranks",
            "83,84,85,86,87,88,89,90,91,92",
            "--output",
            str(output),
        ]
    ) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["phases"][2]["ranks"] == [74, 75, 77, 78, 79, 80, 81, 82]
    assert value["explicit_nonbenchmark_exclusions"] == [
        {"rank": 76, "bytes": 10, "rows": 118001, "sha256": "sha-76", "reason": "non_full_nanochat_shard"}
    ]


def test_takeover_request_fails_closed_on_wrong_rank_and_checksum_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    coord = tmp_path / "coord"
    legacy = tmp_path / "legacy"
    active = tmp_path / "active-helper.sh"
    guarded = tmp_path / "guarded-helper.sh"
    tool = tmp_path / "agent1_v5_signature_takeover.py"
    for path, contents in {
        run / "run_contract.json": b"contract",
        run / "release-pre-dedup" / "manifests" / "combined_manifest.json": b"manifest",
        run / "datatrove_runtime.json": b"runtime",
        active: b"original helper",
        guarded: b"guarded helper",
        tool: b"tool",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    write_json(
        run / "dedup_full_input_audit.json",
        {
            "schema_version": dedup.FULL_INPUT_AUDIT_SCHEMA,
            "status": "passed",
            "combined_manifest_sha256": dedup.sha256_file(run / "release-pre-dedup" / "manifests" / "combined_manifest.json"),
        },
    )
    request = run / "request.json"
    assert takeover.main(
        [
            "create-request",
            "--run-root",
            str(run),
            "--coord-root",
            str(coord),
            "--legacy-pipeline-root",
            str(legacy),
            "--active-helper",
            str(active),
            "--original-helper",
            str(active),
            "--guarded-helper",
            str(guarded),
            "--takeover-tool",
            str(tool),
            "--stop-after-rank",
            "67",
            "--output",
            str(request),
        ]
    ) == 0
    active.write_bytes(guarded.read_bytes())
    active_tool = active.parent / tool.name
    active_tool.write_bytes(tool.read_bytes())
    valid_args = [
        "validate-request",
        "--request",
        str(request),
        "--run-root",
        str(run),
        "--coord-root",
        str(coord),
        "--legacy-pipeline-root",
        str(legacy),
        "--active-helper",
        str(active),
        "--takeover-tool",
        str(active_tool),
        "--task-index",
        "67",
    ]
    assert takeover.main(valid_args) == 0
    wrong_rank = valid_args[:-1] + ["68"]
    try:
        takeover.main(wrong_rank)
    except ValueError as error:
        assert "stop rank" in str(error)
    else:  # pragma: no cover
        raise AssertionError("wrong stop rank was accepted")
    active.write_bytes(b"drift")
    try:
        takeover.main(valid_args)
    except ValueError as error:
        assert "checksum drift" in str(error)
    else:  # pragma: no cover
        raise AssertionError("guarded helper checksum drift was accepted")

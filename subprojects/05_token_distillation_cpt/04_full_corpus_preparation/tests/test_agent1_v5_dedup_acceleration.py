from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import agent1_v5_datatrove as dedup  # noqa: E402
import agent1_v5_dedup_acceleration as acceleration  # noqa: E402


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
            str(tmp_path / "run"),
            "--pipeline-root",
            str(ROOT),
            "--runner",
            str(ROOT / "slurm" / "agent1_v5_eiger" / "normal_signature_runner.sh"),
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
    assert value["chunks"] == [
        {"index": 0, "ranks": [7, 8, 9]},
        {"index": 1, "ranks": [10, 11, 12]},
        {"index": 2, "ranks": [13, 14]},
    ]


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

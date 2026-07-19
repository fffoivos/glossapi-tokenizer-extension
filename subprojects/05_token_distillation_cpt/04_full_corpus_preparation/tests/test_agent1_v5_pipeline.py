from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import agent1_v5_datatrove as dedup  # noqa: E402
import agent1_v5_pipeline as pipeline  # noqa: E402
import publish_private_agent1_v5 as publisher  # noqa: E402
import prototype_agent1_v4_gfm_normalization as gfm  # noqa: E402
import submit_agent1_v5_eiger as submitter  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def canonical_row(source: str, document: str, text: str, quality: float = 0.0) -> dict[str, object]:
    row = {name: None for name in pipeline.CANONICAL_FIELD_NAMES}
    row.update(
        {
            "source_dataset": source,
            "source_doc_id": document,
            "text": text,
            "title": None,
            "author": None,
            "source_metadata_json": "{}",
            "greek_badness_score": quality,
            "mojibake_badness_score": 0.0,
            "cleaner_chars_before": len(text),
            "cleaner_chars_after": len(text),
        }
    )
    return row


def make_combined(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    release = run_root / "release-pre-dedup"
    data = release / "data"
    data.mkdir(parents=True)
    duplicate_text = "Αυτό είναι ένα αρκετά μεγάλο ίδιο ελληνικό κείμενο για δοκιμή."
    base_rows = [canonical_row("nanochat", "base-duplicate", duplicate_text, 99.0)]
    candidate_rows = [
        canonical_row("candidate", "candidate-duplicate", duplicate_text, 0.0),
        canonical_row("candidate", "candidate-unique", "Ένα τελείως διαφορετικό κείμενο.", 0.0),
    ]
    paths = [data / "000000.parquet", data / "000001.parquet"]
    pq.write_table(pa.Table.from_pylist(base_rows, schema=pipeline.canonical_schema()), paths[0])
    pq.write_table(pa.Table.from_pylist(candidate_rows, schema=pipeline.canonical_schema()), paths[1])
    files = []
    for rank, (origin, path, rows) in enumerate(
        (("nanochat_base", paths[0], 1), ("candidate", paths[1], 2))
    ):
        files.append(
            {
                "rank": rank,
                "origin": origin,
                **pipeline.file_receipt(path, root=release, rows=rows),
            }
        )
    manifest = release / "manifests" / "combined_manifest.json"
    write_json(
        manifest,
        {
            "schema_version": pipeline.COMBINED_MANIFEST_SCHEMA,
            "status": "passed",
            "root": str(release),
            "rows": 3,
            "files": files,
        },
    )
    contract = run_root / "run_contract.json"
    write_json(
        contract,
        {
            "schema_version": pipeline.CONTRACT_SCHEMA,
            "status": "passed",
            "run_root": str(run_root),
            "run_id": "agent1-v5-test",
            "private_repositories": {
                "pre_dedup": "owner/pre",
                "deduplicated": "owner/dedup",
            },
        },
    )
    write_json(run_root / "license_override_receipt.json", {"status": "passed"})
    return run_root, contract, manifest


def test_config_locks_all_eighteen_source_adapters_and_schema() -> None:
    config = pipeline.load_config(ROOT / "configs" / "agent1_v5_eiger_pipeline.json")
    assert len(config["sources"]) == 18
    assert all(source["text_path"] for source in config["sources"].values())
    assert config["dedup"]["preserve_greek_diacritics"] is True
    assert config["dedup"]["num_buckets"] * config["dedup"]["hashes_per_bucket"] == 128
    assert pipeline.canonical_schema().names == list(pipeline.CANONICAL_FIELD_NAMES)
    assert pipeline.canonical_schema().names[:6] == list(pipeline.ENVELOPE_FIELDS)
    assert config["execution"]["cluster"] == "clariden"
    assert config["execution"]["production_partition"] == "debug"
    assert config["execution"]["transfer_partition"] == "debug"
    assert config["execution"]["max_walltime"] == "01:25:00"
    assert config["execution"]["max_array_parallelism"] == 2
    assert config["pins"]["glossapi_bundle"].endswith("glossapi-a2aace04fbae.bundle")


def test_clariden_wrapper_isolates_venv_and_setup_avoids_unused_glossapi_extras() -> None:
    slurm = ROOT / "slurm" / "agent1_v5_eiger"
    for name in ("clariden_debug_stage.sh", "clariden_debug_bundle.sh"):
        wrapper = (slurm / name).read_text(encoding="utf-8")
        assert "env -u PYTHONPATH -u PYTHONHOME" in wrapper
    setup = (slurm / "stage.sh").read_text(encoding="utf-8")
    assert 'pip install --no-deps -e "${GLOSSAPI_ROOT}"' not in setup
    assert '"${GLOSSAPI_ROOT}/rust/glossapi_rs_cleaner"' in setup
    assert '"${GLOSSAPI_ROOT}/rust/glossapi_rs_noise"' in setup
    bundle = (slurm / "bundle.sh").read_text(encoding="utf-8")
    assert 'mkdir -p "${RUN_ROOT}/slurm"' in bundle
    assert 'pids+=("$!")' in bundle
    assert 'wait "${pid}"' in bundle
    assert "jobs -pr" not in bundle


def test_debug_bundle_batches_preserve_task_width_and_qos_limit() -> None:
    assert submitter.bundle_batches(158, 32, 2) == [(0, 1), (2, 3), (4, 4)]
    assert submitter.bundle_batches(273, 16, 2)[-1] == (16, 17)
    assert submitter.bundle_batches(32, 1, 2)[-1] == (30, 31)
    assert submitter.bundle_batches(3, 8, 2) == [(0, 0)]


def test_bounded_stage_runner_executes_each_task_once_and_emits_metrics(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "pipeline"
    runner_dir = pipeline_root / "slurm" / "agent1_v5_eiger"
    runner_dir.mkdir(parents=True)
    source_runner = ROOT / "slurm" / "agent1_v5_eiger" / "bounded_stage_runner.sh"
    runner = runner_dir / "bounded_stage_runner.sh"
    runner.write_text(source_runner.read_text(encoding="utf-8"), encoding="utf-8")
    runner.chmod(0o755)
    stage = runner_dir / "stage.sh"
    stage.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$TASK_INDEX\" >> \"$RUN_ROOT/tasks.txt\"\n",
        encoding="utf-8",
    )
    stage.chmod(0o755)
    fake_time = tmp_path / "time"
    fake_time.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test \"$1\" = -v; test \"$2\" = -o; report=$3; shift 3\n"
        "printf 'Maximum resident set size (kbytes): 42\\n' > \"$report\"\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    fake_time.chmod(0o755)
    run_root = tmp_path / "run"
    run_root.mkdir()
    attempt = tmp_path / "attempt"
    environment = {
        **os.environ,
        "STAGE": "bucket",
        "PIPELINE_ROOT": str(pipeline_root),
        "RUN_ROOT": str(run_root),
        "TASK_START": "2",
        "TASK_END": "5",
        "TASK_CONCURRENCY": "2",
        "ATTEMPT_ROOT": str(attempt),
        "ATTEMPT_ID": "test-bounded",
        "TIME_BIN": str(fake_time),
    }
    subprocess.run([str(runner)], check=True, env=environment)
    assert sorted((run_root / "tasks.txt").read_text().splitlines()) == ["2", "3", "4"]
    metrics = sorted((attempt / "metrics").glob("*.json"))
    assert len(metrics) == 3
    values = [json.loads(path.read_text()) for path in metrics]
    assert [row["task_index"] for row in values] == [2, 3, 4]
    assert all(row["status"] == "passed" for row in values)
    assert all(row["workers"] == 2 for row in values)
    assert all(row["attempt_id"] == "test-bounded" for row in values)


def test_submitter_resume_reuses_only_matching_persisted_jobs(tmp_path: Path) -> None:
    run_root = (tmp_path / "run").resolve()
    pipeline_root = tmp_path / "pipeline"
    config_path = tmp_path / "config.json"
    acquisition_receipt = tmp_path / "acquisition.json"
    write_json(config_path, {"execution": {"cluster": "clariden"}})
    write_json(acquisition_receipt, {"status": "passed"})
    bindings = {
        "pipeline_root": str(pipeline_root),
        "config_path": str(config_path),
        "config_sha256": submitter.sha256_file(config_path),
        "acquisition_receipt_path": str(acquisition_receipt),
        "acquisition_receipt_sha256": submitter.sha256_file(acquisition_receipt),
        "account": "a0140",
    }
    state_path = tmp_path / ".resume-test.coord" / "submission_state.json"
    write_json(
        state_path,
        {
            "schema_version": "agent1_v5_eiger_submission_state_v1",
            "run_id": "resume-test",
            "run_root": str(run_root),
            "bindings": bindings,
            "jobs": {
                "setup": {
                    "job_id": "12345",
                    "stage": "setup",
                    "partition": "debug",
                    "walltime": "01:25:00",
                    "dependency": [],
                    "array": None,
                }
            },
        },
    )
    args = argparse.Namespace(
        pipeline_root=pipeline_root,
        config=config_path,
        acquisition_receipt=acquisition_receipt,
        run_root=run_root,
        run_id="resume-test",
        resume=True,
        account="a0140",
    )
    config = {"execution": {"cluster": "clariden"}}
    resumed = submitter.Submitter(args, config)
    assert resumed.jobs["setup"]["job_id"] == "12345"
    assert resumed.submit(
        "setup",
        "setup",
        partition="debug",
        walltime="01:25:00",
    ) == "12345"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["run_id"] = "different-run"
    write_json(state_path, state)
    try:
        submitter.Submitter(args, config)
    except ValueError as error:
        assert "run_id" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched resume state was accepted")


def test_completed_dependencies_are_not_resubmitted_to_slurm(monkeypatch) -> None:
    states = {
        "complete": ("COMPLETED", "0:0"),
        "active": ("RUNNING", ""),
        "failed": ("FAILED", "1:0"),
    }
    monkeypatch.setattr(submitter, "root_job_state", states.__getitem__)
    assert submitter.unresolved_dependencies(["complete", "active"]) == ["active"]
    try:
        submitter.unresolved_dependencies(["failed"])
    except RuntimeError as error:
        assert "FAILED" in str(error)
    else:  # pragma: no cover
        raise AssertionError("failed dependency was accepted")


def test_root_job_state_excludes_diagnostic_job_steps(monkeypatch) -> None:
    commands = []

    class QueueResult:
        stdout = ""

    monkeypatch.setattr(submitter.subprocess, "run", lambda *args, **kwargs: QueueResult())

    def accounting(*command: str) -> str:
        commands.append(command)
        return "123|COMPLETED|0:0"

    monkeypatch.setattr(submitter, "command_output", accounting)
    assert submitter.root_job_state("123") == ("COMPLETED", "0:0")
    assert "-X" in commands[0]


def test_unterminated_known_tag_prefix_is_escaped_before_later_php_close() -> None:
    source = "<a href=](images/example\nlater ?>)"
    metrics = gfm.NormalizationMetrics()
    escaped = gfm._escape_residual_angles(source, metrics)
    assert escaped == "&lt;a href=](images/example\nlater ?>)"
    assert gfm.KNOWN_HTML_TAG_RE.search(escaped) is None
    assert metrics.pseudo_tags_escaped == 1


def test_bundle_runs_every_task_and_propagates_child_failure(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "pipeline"
    stage = pipeline_root / "slurm" / "agent1_v5_eiger" / "stage.sh"
    stage.parent.mkdir(parents=True)
    stage.write_text(
        "#!/usr/bin/env bash\n"
        "mkdir -p \"${RUN_ROOT}/seen\"\n"
        "printf '%s\\n' \"${TASK_INDEX}\" > \"${RUN_ROOT}/seen/${TASK_INDEX}\"\n"
        "if [[ \"${FAIL_TASK:-}\" == \"${TASK_INDEX}\" ]]; then exit 9; fi\n",
        encoding="utf-8",
    )
    stage.chmod(0o755)
    bundle = ROOT / "slurm" / "agent1_v5_eiger" / "bundle.sh"
    base_env = {
        **os.environ,
        "TASK_COUNT": "5",
        "TASKS_PER_NODE": "5",
        "TASK_CONCURRENCY": "2",
        "SLURM_ARRAY_TASK_ID": "0",
        "PIPELINE_ROOT": str(pipeline_root),
        "RUN_ROOT": str(tmp_path / "run"),
        "STAGE": "test",
    }
    assert subprocess.run([bundle], env=base_env, check=False).returncode == 0
    assert sorted(path.name for path in (tmp_path / "run" / "seen").iterdir()) == [
        "0", "1", "2", "3", "4"
    ]
    failed_env = {**base_env, "RUN_ROOT": str(tmp_path / "failed"), "FAIL_TASK": "2"}
    assert subprocess.run([bundle], env=failed_env, check=False).returncode == 1
    assert sorted(path.name for path in (tmp_path / "failed" / "seen").iterdir()) == [
        "0", "1", "2", "3", "4"
    ]


def test_bundled_stage_requeues_timeout_while_receipts_advance(
    tmp_path: Path, monkeypatch
) -> None:
    instance = object.__new__(submitter.Submitter)
    instance.args = argparse.Namespace(run_root=tmp_path / "run")
    instance.config = {
        "execution": {
            "production_partition": "debug",
            "max_walltime": "01:25:00",
            "max_array_parallelism": 2,
            "max_transient_retries": 2,
        }
    }
    instance.bundle_script = tmp_path / "bundle.sh"
    submitted = []

    def fake_submit(self, name, stage, **kwargs):
        submitted.append((name, stage, kwargs))
        return str(100 + len(submitted))

    progress = iter([154, 154, 160])
    monkeypatch.setattr(submitter.Submitter, "submit", fake_submit)
    monkeypatch.setattr(submitter.Submitter, "checkpoint_progress", lambda self, stage: next(progress))
    outcomes = iter([False, False, True])

    def fake_wait(job_ids, poll_seconds):
        if not next(outcomes):
            raise RuntimeError("timeout")
        return {job_ids[0]: ("COMPLETED", "0:0")}

    monkeypatch.setattr(submitter, "wait_for_jobs", fake_wait)
    monkeypatch.setattr(submitter, "root_job_state", lambda job_id: ("TIMEOUT", "0:0"))
    assert instance.bundled(
        "transform",
        "transform",
        32,
        32,
        ["canary"],
        poll_seconds=1,
    )[-1] == "103"
    assert [row[0] for row in submitted] == [
        "transform-part000",
        "transform-part000-retry001",
        "transform-part000-retry002",
    ]


def test_transform_task_checkpoints_and_resumes_without_recleaning(
    tmp_path: Path, monkeypatch
) -> None:
    config = json.loads(
        (ROOT / "configs" / "agent1_v5_eiger_pipeline.json").read_text(encoding="utf-8")
    )
    config["execution"]["transform_batch_rows"] = 2
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [f"doc-{index}" for index in range(5)],
                "markdown_text": [f"κείμενο {index}" for index in range(5)],
                "metadata_json": [json.dumps({"subject": f"τίτλος {index}"}) for index in range(5)],
            }
        ),
        input_path,
    )
    run_root = tmp_path / "run"
    contract_path = run_root / "run_contract.json"
    write_json(
        contract_path,
        {
            "schema_version": pipeline.CONTRACT_SCHEMA,
            "status": "passed",
            "run_root": str(run_root),
            "glossapi": {"root": str(tmp_path / "glossapi")},
        },
    )
    task = {
        "task_index": 0,
        "source_id": "diavgeia",
        "repo_id": "glossAPI/diavgeia",
        "revision": "a" * 40,
        "artifact_path": "input.parquet",
        "input_path": str(input_path),
        "input_expected_hash": pipeline.sha256_file(input_path),
        "input_hash_kind": "sha256",
        "row_groups": [0],
        "row_start": 0,
        "rows": 5,
        "uncompressed_bytes": 100,
    }
    tasks_path = run_root / "transform_tasks.json"
    write_json(
        tasks_path,
        {
            "schema_version": pipeline.TASK_MANIFEST_SCHEMA,
            "task_count": 1,
            "tasks": [task],
        },
    )
    repetition_path = tmp_path / "repetition.py"
    repetition_path.write_text("# pinned test module\n", encoding="utf-8")

    class Repetition:
        @staticmethod
        def replace_complex_repetitions(text, metrics=None):
            return text

    monkeypatch.setattr(
        pipeline.gfm,
        "_load_repetition_module",
        lambda root: (Repetition(), repetition_path),
    )
    clean_calls = []

    def fake_clean(text, *, repetition_cleaner):
        clean_calls.append(text)
        return {
            "normalized_markdown": text,
            "repetition_metrics": {
                "complex_repetition_replacements": 0,
                "complex_repetition_characters_removed": 0,
                "complex_repetition_rule_counts": {},
            },
            "generated_image_metrics": {
                "generated_image_artifact_count": 0,
                "generated_image_characters_removed": 0,
                "image_description_comments_emitted": 0,
            },
            "markup_metrics": {"tag_counts": {}, "transformations": {}},
        }

    monkeypatch.setattr(pipeline.gfm, "clean_then_normalize_to_gfm", fake_clean)
    args = argparse.Namespace(
        config=config_path,
        contract=contract_path,
        tasks=tasks_path,
        task_index=0,
    )
    assert pipeline.transform_task(args) == 0
    assert clean_calls == [f"κείμενο {index}" for index in range(5)]
    checkpoint_root = run_root / "10-transform" / "checkpoints" / "task-000000"
    assert len(list(checkpoint_root.glob("part-*.receipt.json"))) == 3
    output_path = run_root / "10-transform" / "shards" / "task-000000.parquet"
    receipt_path = run_root / "10-transform" / "receipts" / "task-000000.json"
    assert pq.read_table(output_path).column("source_row_index").to_pylist() == list(range(5))

    receipt_path.unlink()
    output_path.unlink()
    (run_root / "10-transform" / "audits" / "task-000000.jsonl.gz").unlink()
    (run_root / "10-transform" / "issues" / "task-000000.jsonl.gz").unlink()
    clean_calls.clear()
    assert pipeline.transform_task(args) == 0
    assert clean_calls == []
    assert pq.read_table(output_path).num_rows == 5


def test_nested_source_mapping_extracts_fields_and_keeps_remaining_metadata() -> None:
    config = pipeline.load_config(ROOT / "configs" / "agent1_v5_eiger_pipeline.json")
    mapping = config["sources"]["diavgeia"]
    row = {
        "id": "decision-1",
        "markdown_text": "# Κείμενο",
        "metadata_json": json.dumps({"subject": "Απόφαση", "ada": "XYZ"}),
        "extra": 7,
    }
    assert pipeline.read_path(row, mapping["text_path"]) == "# Κείμενο"
    assert pipeline.read_path(row, mapping["title_path"]) == "Απόφαση"
    assert pipeline.read_path(row, mapping["author_path"]) is None
    metadata = json.loads(pipeline.metadata_json(row, mapping))
    assert "markdown_text" not in metadata
    assert metadata["metadata_json"] == {"ada": "XYZ"}
    assert metadata["id"] == "decision-1"
    assert metadata["extra"] == 7


def test_acquisition_identity_detects_stat_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    path.write_bytes(b"pinned")
    stat_result = path.stat()
    binding = {
        "local_path": str(path),
        "expected_hash": pipeline.sha256_file(path),
        "hash_kind": "sha256",
        "size": stat_result.st_size,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }
    assert pipeline.validate_acquired_file_identity(binding) == path.resolve()
    binding["mtime_ns"] += 1
    try:
        pipeline.validate_acquired_file_identity(binding)
    except ValueError as error:
        assert "mtime_ns" in str(error)
    else:  # pragma: no cover
        raise AssertionError("stat drift was not detected")


def test_relaxed_exact_preserves_greek_diacritic_and_number_distinctions() -> None:
    assert dedup.relaxed_exact_text("  Άλφα — 12  ") == "άλφα 12"
    assert dedup.relaxed_exact_text("Άλφα") != dedup.relaxed_exact_text("Αλφα")
    assert dedup.relaxed_exact_text("τιμή 12") != dedup.relaxed_exact_text("τιμή 13")


def test_actual_jaccard_uses_full_unique_shingle_sets() -> None:
    import numpy as np

    left = np.array([1, 2, 3, 5], dtype="<u8")
    right = np.array([2, 3, 4, 5], dtype="<u8")
    assert dedup._jaccard_sorted(left, right) == 3 / 5
    assert dedup._jaccard_sorted(np.array([], dtype="<u8"), right) == 0.0


def test_merge_lsh_pairs_blocks_unresolved_oversized_groups(tmp_path: Path) -> None:
    run_root, contract, combined = make_combined(tmp_path)
    config = json.loads((ROOT / "configs" / "agent1_v5_eiger_pipeline.json").read_text())
    config["dedup"].update(
        {
            "num_buckets": 1,
            "hashes_per_bucket": 128,
            "max_bucket_documents": 2,
        }
    )
    config_path = tmp_path / "config.json"
    write_json(config_path, config)

    bucket = run_root / "60-dedup" / "minhash-buckets" / "00000_00.dups"
    bucket.parent.mkdir(parents=True)
    bucket.write_bytes(
        dedup.PAIR_STRUCT.pack(0, 0, 0, 1)
        + dedup.PAIR_STRUCT.pack(0, 1, 0, 2)
    )
    bucket_receipt = bucket.parent / "receipts" / "000000.json"
    write_json(
        bucket_receipt,
        {
            "schema_version": dedup.BUCKET_RECEIPT_SCHEMA,
            "status": "passed",
            "output": pipeline.file_receipt(bucket, root=run_root),
        },
    )

    database = run_root / "lsh-pairs.sqlite"
    try:
        dedup.merge_lsh_pairs(
            argparse.Namespace(
                config=config_path,
                contract=contract,
                combined_manifest=combined,
                output=database,
            )
        )
    except RuntimeError as error:
        assert "unresolved oversized LSH group" in str(error)
    else:  # pragma: no cover
        raise AssertionError("oversized LSH group did not block pair-manifest release")

    assert not database.exists()
    blocked_database = database.with_suffix(database.suffix + ".blocked")
    assert blocked_database.is_file()
    oversized = json.loads(database.with_suffix(".oversized.json").read_text())
    assert oversized["status"] == "blocked"
    assert oversized["groups"] == [{"bucket": 0, "documents": 3, "edges": 2}]
    manifest = json.loads(database.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "blocked"
    assert manifest["reason"] == "unresolved_oversized_lsh_groups"
    assert manifest["pairs_excluding_oversized_groups"] == 0
    try:
        dedup._pair_manifest(database)
    except ValueError as error:
        assert "not passed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("downstream stage accepted a blocked pair manifest")


def test_merge_signatures_reuses_full_input_audit_without_payload_rehash(
    monkeypatch, tmp_path: Path
) -> None:
    run_root, contract, combined_path = make_combined(tmp_path)
    combined = json.loads(combined_path.read_text())
    runtime = run_root / "datatrove_runtime.json"
    write_json(runtime, {"status": "passed", "runtime": "test"})
    full_audit = run_root / "dedup_full_input_audit.json"
    write_json(
        full_audit,
        {
            "schema_version": dedup.FULL_INPUT_AUDIT_SCHEMA,
            "status": "passed",
            "run_contract_sha256": pipeline.sha256_file(contract),
            "combined_manifest_sha256": pipeline.sha256_file(combined_path),
            "runtime_receipt_sha256": pipeline.sha256_file(runtime),
            "run_id": "agent1-v5-test",
            "files": combined["files"],
            "rows": combined["rows"],
            "task_count": len(combined["files"]),
        },
    )
    for inventory in combined["files"]:
        rank = int(inventory["rank"])
        signature = run_root / "60-dedup" / "minhash-signatures" / f"sig-{rank}.bin"
        signature.parent.mkdir(parents=True, exist_ok=True)
        signature.write_bytes(f"rank-{rank}".encode())
        binding = pipeline.file_receipt(signature, root=run_root)
        write_json(
            signature.parent / "receipts" / f"{rank:06d}.json",
            {
                "schema_version": dedup.SIGNATURE_RECEIPT_SCHEMA,
                "status": "passed",
                "task_index": rank,
                "input": inventory,
                "outputs": [binding] * 32,
            },
        )

    def reject_legacy_payload_validation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("merge-signatures reread the combined payload")

    monkeypatch.setattr(dedup, "_load_release", reject_legacy_payload_validation)
    output = run_root / "signature_manifest.json"
    assert (
        dedup.merge_signatures(
            argparse.Namespace(
                config=ROOT / "configs" / "agent1_v5_eiger_pipeline.json",
                contract=contract,
                combined_manifest=combined_path,
                runtime_receipt=runtime,
                full_input_audit=full_audit,
                output=output,
            )
        )
        == 0
    )
    manifest = json.loads(output.read_text())
    assert manifest["status"] == "passed"
    assert manifest["task_count"] == 2
    assert manifest["bucket_count"] == 32


def test_merge_lsh_pairs_promotes_candidate_when_groups_are_bounded(tmp_path: Path) -> None:
    run_root, contract, combined = make_combined(tmp_path)
    config = json.loads((ROOT / "configs" / "agent1_v5_eiger_pipeline.json").read_text())
    config["dedup"].update(
        {
            "num_buckets": 1,
            "hashes_per_bucket": 128,
            "max_bucket_documents": 3,
        }
    )
    config_path = tmp_path / "config.json"
    write_json(config_path, config)

    bucket = run_root / "60-dedup" / "minhash-buckets" / "00000_00.dups"
    bucket.parent.mkdir(parents=True)
    bucket.write_bytes(
        dedup.PAIR_STRUCT.pack(0, 0, 0, 1)
        + dedup.PAIR_STRUCT.pack(0, 1, 0, 2)
    )
    write_json(
        bucket.parent / "receipts" / "000000.json",
        {
            "schema_version": dedup.BUCKET_RECEIPT_SCHEMA,
            "status": "passed",
            "output": pipeline.file_receipt(bucket, root=run_root),
        },
    )

    database = run_root / "lsh-pairs.sqlite"
    assert (
        dedup.merge_lsh_pairs(
            argparse.Namespace(
                config=config_path,
                contract=contract,
                combined_manifest=combined,
                output=database,
            )
        )
        == 0
    )
    assert database.is_file()
    assert not database.with_suffix(database.suffix + ".partial").exists()
    assert not database.with_suffix(database.suffix + ".blocked").exists()
    manifest = dedup._pair_manifest(database)
    assert manifest["status"] == "passed"
    assert manifest["pairs"] == 2
    oversized = json.loads(database.with_suffix(".oversized.json").read_text())
    assert oversized["status"] == "passed"
    assert oversized["groups"] == []


def test_exact_cluster_protects_nanochat_and_filters_candidate(tmp_path: Path) -> None:
    run_root, contract, combined = make_combined(tmp_path)
    for rank in range(2):
        dedup.exact_index_task(
            argparse.Namespace(contract=contract, combined_manifest=combined, task_index=rank)
        )
    exact_manifest = run_root / "exact_manifest.json"
    dedup.merge_exact_index(
        argparse.Namespace(
            contract=contract,
            combined_manifest=combined,
            output=exact_manifest,
        )
    )
    verified_shard = run_root / "verified-empty.parquet"
    pq.write_table(
        pa.table(
            {
                "pair_id": pa.array([], type=pa.uint64()),
                "left_ref": pa.array([], type=pa.uint64()),
                "right_ref": pa.array([], type=pa.uint64()),
                "jaccard": pa.array([], type=pa.float64()),
            }
        ),
        verified_shard,
    )
    verified_manifest = run_root / "verified_manifest.json"
    write_json(
        verified_manifest,
        {
            "schema_version": dedup.VERIFY_MANIFEST_SCHEMA,
            "status": "passed",
            "counters": {"verified": 0},
            "shards": [pipeline.file_receipt(verified_shard, root=run_root, rows=0)],
        },
    )
    removals = run_root / "removals.sqlite"
    dedup.cluster_duplicates(
        argparse.Namespace(
            contract=contract,
            combined_manifest=combined,
            exact_manifest=exact_manifest,
            verified_manifest=verified_manifest,
            output=removals,
        )
    )
    cluster_manifest = json.loads(removals.with_suffix(".manifest.json").read_text())
    assert cluster_manifest["removed_rows"] == 1
    for rank in range(2):
        dedup.filter_task(
            argparse.Namespace(
                contract=contract,
                combined_manifest=combined,
                removal_database=removals,
                task_index=rank,
            )
        )
    output = run_root / "release-deduplicated"
    dedup.merge_filtered_release(
        argparse.Namespace(
            contract=contract,
            combined_manifest=combined,
            removal_database=removals,
            output=output,
        )
    )
    result = pq.read_table(output / "data" / "000000.parquet").to_pylist()
    candidate = pq.read_table(output / "data" / "000001.parquet").to_pylist()
    assert [row["source_doc_id"] for row in result] == ["base-duplicate"]
    assert [row["source_doc_id"] for row in candidate] == ["candidate-unique"]


def test_merge_transform_records_quarantined_blank_rows_without_blocking(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    contract = run_root / "run_contract.json"
    write_json(
        contract,
        {
            "schema_version": pipeline.CONTRACT_SCHEMA,
            "status": "passed",
            "run_root": str(run_root),
        },
    )
    tasks = {
        "task_count": 2,
        "tasks": [
            {"task_index": 0, "source_id": "complete_source"},
            {"task_index": 1, "source_id": "partially_blank_source"},
        ],
    }
    task_path = run_root / "transform_tasks.json"
    write_json(task_path, tasks)
    for task, counters in zip(
        tasks["tasks"],
        (
            {"input_rows": 1, "output_rows": 1},
            {"input_rows": 2, "output_rows": 1, "quarantined_blank_text": 1},
        ),
    ):
        index = task["task_index"]
        shard, receipt_path = pipeline._task_output_paths(run_root, "10-transform", index)
        shard.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([{"source_id": task["source_id"]}], schema=pipeline.transform_schema()), shard
        )
        audit = run_root / "10-transform" / "audits" / f"task-{index:06d}.jsonl.gz"
        issues = run_root / "10-transform" / "issues" / f"task-{index:06d}.jsonl.gz"
        audit.parent.mkdir(parents=True, exist_ok=True)
        issues.parent.mkdir(parents=True, exist_ok=True)
        audit.write_bytes(b"")
        issues.write_bytes(b"")
        write_json(
            receipt_path,
            {
                "schema_version": pipeline.TRANSFORM_RECEIPT_SCHEMA,
                "task_sha256": pipeline.sha256_json(task),
                "output": pipeline.file_receipt(shard, root=run_root, rows=1),
                "audit": pipeline.file_receipt(audit, root=run_root),
                "issues": pipeline.file_receipt(issues, root=run_root),
                "counters": counters,
            },
        )

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda _path: {"sources": {"complete_source": {}, "partially_blank_source": {}}},
    )
    output = run_root / "transform_manifest.json"
    assert pipeline.merge_transform(
        argparse.Namespace(config=tmp_path / "config.json", contract=contract, tasks=task_path, output=output)
    ) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["blocking_issues"] == []
    assert manifest["quarantined_rows"] == 1
    assert manifest["quarantine_issues"] == [
        {
            "source_id": "partially_blank_source",
            "reason": "missing_or_empty_text_rows_quarantined",
            "rows": 1,
        }
    ]


def test_merge_glossapi_records_empty_rows_without_blocking(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    contract = run_root / "run_contract.json"
    write_json(
        contract,
        {
            "schema_version": pipeline.CONTRACT_SCHEMA,
            "status": "passed",
            "run_root": str(run_root),
        },
    )
    transform_shard = run_root / "10-transform" / "shards" / "task-000000.parquet"
    transform_shard.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{"source_id": "candidate"}], schema=pipeline.transform_schema()), transform_shard
    )
    transform_manifest = run_root / "transform_manifest.json"
    write_json(
        transform_manifest,
        {
            "schema_version": pipeline.TRANSFORM_MANIFEST_SCHEMA,
            "status": "passed",
            "output_rows": 2,
            "shards": [pipeline.file_receipt(transform_shard, root=run_root, rows=2)],
        },
    )
    output_shard, receipt_path = pipeline._task_output_paths(run_root, "20-glossapi", 0)
    output_shard.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([canonical_row("candidate", "doc", "κείμενο")], schema=pipeline.canonical_schema()),
        output_shard,
    )
    issues = run_root / "20-glossapi" / "issues" / "task-000000.jsonl.gz"
    issues.parent.mkdir(parents=True, exist_ok=True)
    issues.write_bytes(b"")
    write_json(
        receipt_path,
        {
            "schema_version": pipeline.GLOSSAPI_RECEIPT_SCHEMA,
            "input": {"sha256": pipeline.file_receipt(transform_shard, root=run_root, rows=2)["sha256"]},
            "output": pipeline.file_receipt(output_shard, root=run_root, rows=1),
            "issues": pipeline.file_receipt(issues, root=run_root),
            "counters": {"input_rows": 2, "output_rows": 1, "quarantined_empty_after_glossapi": 1},
        },
    )
    output = run_root / "glossapi_manifest.json"
    assert pipeline.merge_glossapi(
        argparse.Namespace(contract=contract, transform_manifest=transform_manifest, output=output)
    ) == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["blocking_issues"] == []
    assert manifest["quarantined_rows"] == 1
    assert manifest["quarantine_issues"] == [
        {"reason": "empty_after_glossapi_rows_quarantined", "rows": 1}
    ]


def test_publisher_waits_out_hub_rate_limit_without_requeueing(monkeypatch) -> None:
    class Response:
        status_code = 429
        headers = {"Retry-After": "1"}

    class RateLimited(Exception):
        response = Response()

    class Api:
        def __init__(self) -> None:
            self.commit_calls = 0
            self.upload_kwargs: dict[str, object] = {}

        def create_commit(self, *_args: object, **_kwargs: object) -> str:
            self.commit_calls += 1
            if self.commit_calls == 1:
                raise RateLimited("rate limited")
            return "commit-sha"

        def upload_large_folder(self, **kwargs: object) -> None:
            self.upload_kwargs = kwargs
            assert self.create_commit() == "commit-sha"

    pauses: list[int] = []
    monkeypatch.setattr(publisher.time, "sleep", pauses.append)
    api = Api()
    publisher.upload_large_folder_with_rate_limit_backoff(
        api,
        repo_id="owner/private",
        repo_type="dataset",
        folder_path="/tmp/staging",
    )

    assert pauses == [publisher.HUB_RATE_LIMIT_BACKOFF_SECONDS]
    assert api.commit_calls == 2
    assert api.upload_kwargs["num_workers"] == publisher.HUB_UPLOAD_WORKERS

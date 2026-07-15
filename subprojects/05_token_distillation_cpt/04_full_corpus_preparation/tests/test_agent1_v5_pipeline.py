from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import agent1_v5_datatrove as dedup  # noqa: E402
import agent1_v5_pipeline as pipeline  # noqa: E402
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


def test_debug_bundle_batches_preserve_task_width_and_qos_limit() -> None:
    assert submitter.bundle_batches(158, 32, 2) == [(0, 1), (2, 3), (4, 4)]
    assert submitter.bundle_batches(273, 16, 2)[-1] == (16, 17)
    assert submitter.bundle_batches(32, 1, 2)[-1] == (30, 31)
    assert submitter.bundle_batches(3, 8, 2) == [(0, 0)]


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

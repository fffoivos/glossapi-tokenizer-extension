from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from contract_utils import geometry, nearest_replay_targets, sha256_file, write_json_atomic  # noqa: E402
import decontaminate_targeted_corpus as targeted_decontam  # noqa: E402
import audit_targeted_decontamination as decontam_audit  # noqa: E402
import audit_validation_exclusion as validation_audit  # noqa: E402
import exclude_frozen_validation_content as validation_exclusion  # noqa: E402
import search_existing_poly_artifact as poly_search  # noqa: E402
import search_renamed_poly_artifact as renamed_poly_search  # noqa: E402
import search_archived_poly_artifact as archived_poly_search  # noqa: E402
import audit_release_polytonic_sources as release_poly_audit  # noqa: E402
import prepare_corrected_initial_hf as corrected_initial_hf  # noqa: E402
import finalize_targeted_initial_greekmmlu as targeted_greekmmlu  # noqa: E402
import build_native_suite_training_exclusions as native_exclusions  # noqa: E402
from extract_release_sources import deterministic_bucket  # noqa: E402
from finalize_targeted_pool_catalogs import digest16  # noqa: E402
from freeze_experiment_contract import validate_static  # noqa: E402
from run_parallel_task_batch import task_contract  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_atomic_receipts_refuse_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    write_json_atomic(receipt, {"status": "first"})
    try:
        write_json_atomic(receipt, {"status": "replacement"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable receipt was overwritten")
    assert read(receipt) == {"status": "first"}


def test_native_suite_exclusions_select_only_published_recommendations(tmp_path: Path) -> None:
    columns = {name: [] for name in native_exclusions.MATCH_COLUMNS}
    rows = [
        {
            "benchmark": "a", "dataset_row_index": 3, "dataset_shard": "data/000001.parquet",
            "document_key_sha256": "k", "document_text_sha256": "t", "example_id": "a:1",
            "evaluation_unit_id": "a:1", "match_category": "strict", "match_strength": "strong",
            "recommended_exclusion": True, "source_dataset": "source", "source_doc_id": "doc",
        },
        {
            "benchmark": "b", "dataset_row_index": 3, "dataset_shard": "data/000001.parquet",
            "document_key_sha256": "k", "document_text_sha256": "t", "example_id": "b:1",
            "evaluation_unit_id": "b:1", "match_category": "strict", "match_strength": "strong",
            "recommended_exclusion": True, "source_dataset": "source", "source_doc_id": "doc",
        },
        {
            "benchmark": "a", "dataset_row_index": 4, "dataset_shard": "data/000001.parquet",
            "document_key_sha256": "q", "document_text_sha256": "u", "example_id": "a:2",
            "evaluation_unit_id": "a:2", "match_category": "question_only", "match_strength": "candidate",
            "recommended_exclusion": False, "source_dataset": "source", "source_doc_id": "doc2",
        },
    ]
    for row in rows:
        for name in columns:
            columns[name].append(row[name])
    path = tmp_path / "matches.parquet"
    pq.write_table(pa.table(columns), path)
    exclusions, counts = native_exclusions.load_exact_exclusions(path)
    assert list(exclusions) == [("data/000001.parquet", 3)]
    assert exclusions[("data/000001.parquet", 3)]["benchmarks"] == {"a", "b"}
    assert counts == {"a": 1, "b": 1}
    assert native_exclusions.raw_text_sha256("λόγος") == hashlib.sha256("λόγος".encode()).hexdigest()


def test_poly_artifact_search_is_discovery_only_and_hashes_matches(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    exact = root / "cache/sha256-blob"
    related = root / "scholarios_source.parquet"
    linked = root / "nested/poly_train.parquet"
    ignored = root / "unrelated.parquet"
    exact.parent.mkdir()
    exact.write_bytes(b"exact-existing-split")
    related.write_bytes(b"related-source")
    ignored.write_bytes(b"not-a-candidate")
    linked.parent.mkdir()
    linked.symlink_to(exact)
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["search", "--root", str(root), "--output", str(output)],
    )
    poly_search.main()
    receipt = read(output)
    assert receipt["status"] == "completed"
    assert receipt["contract"]["discovery_only"] is True
    assert receipt["contract"]["reconstruction_performed"] is False
    assert receipt["contract"]["substitution_performed"] is False
    assert receipt["candidate_count"] == 2
    assert receipt["exact_name_candidate_count"] == 1
    exact_candidates = [item for item in receipt["candidates"] if item["exact_expected_name"]]
    assert len(exact_candidates) == 1
    assert exact_candidates[0]["sha256"] == hashlib.sha256(b"exact-existing-split").hexdigest()
    assert any(Path(value).name == "poly_train.parquet" for value in exact_candidates[0]["discovered_paths"])
    by_name = {item["basename"]: item for item in receipt["candidates"]}
    assert "unrelated.parquet" not in by_name


def test_renamed_poly_search_is_discovery_only() -> None:
    script = (ROOT / "scripts/search_renamed_poly_artifact.py").read_text()
    sbatch = (ROOT / "clariden/search_renamed_poly_artifact_debug.sbatch").read_text()
    assert '"discovery_only": True' in script
    assert '"reconstruction_performed": False' in script
    assert '"substitution_performed": False' in script
    assert '"corpus_files_written": 0' in script
    assert "EXPECTED_ROWS = 14_929" in script
    assert "EXPECTED_TEXT_CHARS = 409_101_812" in script
    assert "EXPECTED_UTF8_BYTES = 802_061_905" in script
    assert "#SBATCH --partition=debug" in sbatch
    assert "verify_code_bundle.py" in sbatch


def test_renamed_poly_search_matches_all_frozen_content_facts(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "renamed.parquet"
    texts = ["ἀρχή", "λόγος"]
    sources = ["source_a", "source_b"]
    pq.write_table(
        pa.table({"source_dataset": sources, "text": texts}),
        path,
        row_group_size=1,
    )
    monkeypatch.setattr(renamed_poly_search, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(
        renamed_poly_search, "EXPECTED_TEXT_CHARS", sum(map(len, texts))
    )
    monkeypatch.setattr(
        renamed_poly_search,
        "EXPECTED_UTF8_BYTES",
        sum(len(value.encode("utf-8")) for value in texts),
    )
    monkeypatch.setattr(
        renamed_poly_search,
        "EXPECTED_SOURCES",
        {"source_a": 1, "source_b": 1},
    )
    result = renamed_poly_search.inspect_candidate(path)
    assert result["classification"] == "exact_frozen_split_match"
    assert result["rows"] == 2
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_renamed_poly_search_discovers_extensionless_parquet_blob(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    blob = root / "content-addressed-blob-without-extension"
    pq.write_table(pa.table({"source_dataset": ["s"], "text": ["λόγος"]}), blob)
    non_parquet = root / "large-enough.bin"
    non_parquet.write_bytes(b"not parquet")

    discovered = renamed_poly_search.parquet_paths(
        root,
        minimum_bytes=1,
        maximum_bytes=1_000_000,
    )

    assert discovered == [blob.resolve()]
    assert renamed_poly_search.has_parquet_signature(blob) is True
    assert renamed_poly_search.has_parquet_signature(non_parquet) is False


def test_archived_poly_search_is_discovery_only() -> None:
    script = (ROOT / "scripts/search_archived_poly_artifact.py").read_text()
    sbatch = (ROOT / "clariden/search_archived_poly_artifact_debug.sbatch").read_text()
    assert '"discovery_only": True' in script
    assert '"archive_members_extracted": 0' in script
    assert '"corpus_files_written": 0' in script
    assert '"reconstruction_performed": False' in script
    assert '"substitution_performed": False' in script
    assert '"candidate_is_not_authority_until_exact_content_audit": True' in script
    assert "#SBATCH --partition=debug" in sbatch
    assert "verify_code_bundle.py" in sbatch
    assert "TARGET8_ARCHIVE_MINIMUM_BYTES" in sbatch
    assert "TARGET8_ARCHIVE_MAXIMUM_BYTES" in sbatch
    assert "TARGET8_ARCHIVE_TIMEOUT_SECONDS" in sbatch
    assert "tar -xf" not in script
    assert "unzip -p" not in script


def test_archived_poly_search_lists_tar_and_zip_without_extracting(tmp_path: Path) -> None:
    import tarfile
    import zipfile

    payload = tmp_path / "poly_train.parquet"
    payload.write_bytes(b"existing artifact bytes")
    tar_path = tmp_path / "backup.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(payload, arcname="c3p_polytonic_20260518T_impl/splits/poly_train.parquet")
    zip_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(payload, "splits/poly_train.parquet")

    for archive in (tar_path, zip_path):
        result = archived_poly_search.inspect_archive(archive, timeout_seconds=30)
        assert result["classification"] == "candidate_members_found"
        assert result["candidate_members"]
    assert payload.read_bytes() == b"existing artifact bytes"


def test_archived_poly_inventory_preserves_out_of_bound_files(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.zip"
    tiny.write_bytes(b"PK")
    discovered = archived_poly_search.discover_archive_paths(tmp_path)
    assert discovered == [tiny]


def test_release_polytonic_audit_signal_is_distinctive_not_plain_tonos() -> None:
    chars, greek, distinctive, present = release_poly_audit.inspect_text("ά λόγος ἀρχή α\u0313")
    assert chars == len("ά λόγος ἀρχή α\u0313")
    assert greek > 0
    assert distinctive == 2
    assert present is True


def test_decontamination_audit_accepts_canonical_omitted_zero_dropped_count() -> None:
    from collections import Counter

    totals = Counter(input=7088, kept=7088, dropped=0, ledger=7088)
    assert decontam_audit.declared_counts_match(totals, {"input": 7088, "kept": 7088})
    assert not decontam_audit.declared_counts_match(totals, {"input": 7088, "kept": 7087})


def test_selected_pool_finalizer_accepts_canonical_omitted_zero_dropped_count() -> None:
    finalizer = (ROOT / "scripts/finalize_selected_pools.py").read_text()
    assert 'counts.get("dropped", 0)' in finalizer
    assert 'manifest["counts"].get("dropped", 0)' in finalizer
    academic_finalizer = (ROOT / "scripts/finalize_academic_pool.py").read_text()
    assert 'decontam["counts"].get("dropped", 0)' in academic_finalizer


def test_release_polytonic_audit_and_extraction_are_debug_only() -> None:
    script = (ROOT / "scripts/audit_release_polytonic_sources.py").read_text()
    audit_sbatch = (ROOT / "clariden/audit_release_polytonic_sources_debug.sbatch").read_text()
    extract_sbatch = (ROOT / "clariden/extract_release_polytonic_sources_debug.sbatch").read_text()
    assert '"selection_authority": "pinned_hf_release_only"' in script
    assert '"external_dataset_used": False' in script
    assert '"rows_written": 0' in script
    assert "#SBATCH --partition=debug" in audit_sbatch + extract_sbatch
    assert "extract_release_sources.py" in extract_sbatch


def test_release_polytonic_preparation_is_debug_only_and_has_no_external_input_path() -> None:
    wrapper = (ROOT / "clariden/prepare_release_polytonic_debug.sbatch").read_text()
    binary_freezer = (ROOT / "scripts/freeze_targeted_binary_inputs.py").read_text()
    asset_freezer = (ROOT / "scripts/freeze_training_assets.py").read_text()
    catalog_finalizer = (ROOT / "scripts/finalize_targeted_pool_catalogs.py").read_text()
    poly_finalizer = (ROOT / "scripts/finalize_poly_pool.py").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "count_release_polytonic_sources_debug.sbatch" in wrapper
    assert "decontaminate_poly_debug.sbatch" in wrapper
    assert "audit_poly_validation_debug.sbatch" in wrapper
    assert "finalize_poly_pool_debug.sbatch" in wrapper
    assert '"release_polytonic_sources"' in binary_freezer
    assert '"poly_train"' not in binary_freezer
    assert '"release_polytonic_sources_passes": 1' in asset_freezer
    assert '"release_polytonic_sources_passes": 1' in catalog_finalizer
    assert "counted_files == extracted_files" in poly_finalizer
    assert "poly token receipt did not count the release extraction" in poly_finalizer
    assert "poly decontamination input is not the release extraction root" in poly_finalizer
    assert 'decontam["counts"].get("dropped", 0)' in poly_finalizer
    assert 'exclusion["counts"].get("excluded", 0)' in poly_finalizer
    resume = (ROOT / "clariden/resume_release_polytonic_after_decontam_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in resume
    assert "audit_poly_decontamination_debug.sbatch" in resume
    assert "decontaminate_poly_debug.sbatch" not in resume


def test_release_polytonic_final_receipt_only_resume_is_debug_and_does_not_reprocess() -> None:
    final_only = (ROOT / "clariden/finalize_release_polytonic_only_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in final_only
    assert "finalize_poly_pool.py" in final_only
    assert "count_parquet_tokens.py" not in final_only
    assert "decontaminate_poly_debug.sbatch" not in final_only
    assert "exclude_frozen_validation_content.py" not in final_only


def test_static_configs_and_allocation_arithmetic() -> None:
    a = read(ROOT / "configs/experiment_a_recipe.json")
    b = read(ROOT / "configs/experiment_b_recipe.json")
    allocation = read(ROOT / "configs/allocation_plan.json")
    validate_static(a, b, allocation)
    assert b["status"] == "retired_by_owner_20260812"
    assert b["launch_authorized"] is False
    assert b["retirement"]["allow_new_gpu_submissions"] is False
    assert allocation["experiment_b"]["launch_authorized"] is False
    poly = a["modern_data"]["polytonic"]
    assert poly["selection_authority"] == "pinned_hf_release_only"
    assert poly["external_dataset_allowed"] is False
    assert poly["row_level_reconstruction_allowed"] is False
    assert poly["additional_deduplication_allowed"] is False
    assert poly["source_datasets"] == [
        "1000_prwta_xronia_ellhnikhs",
        "Ekklisiastika_Keimena",
        "Wikisource_Greek_texts",
        "klasikh_arx_ell_grammateia",
    ]
    assert poly["historical_split_manifest_role"] == "provenance_only_not_an_input_requirement"
    scoped_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md",
            ROOT / "configs/experiment_a_recipe.json",
        )
    )
    assert "home_path" not in scoped_text
    assert "/home/" not in scoped_text
    assert allocation["experiment_a"]["maximum_hold_seconds"] == 6000
    assert allocation["experiment_a"]["source_trigger_seconds"] == 30000
    assert allocation["distributed_prelaunch_restart_smoke"] == {
        "partition": "normal",
        "profile_id": "dp32_16node",
        "nodes": 16,
        "leaf_switches": 1,
        "wall_seconds": 3600,
        "allocations_per_experiment": 1,
        "uninterrupted_updates": 2,
        "resume_source": "exact_uninterrupted_control_checkpoint",
        "resumed_updates": 1,
        "submit_only_after_debug_data_and_training_assets_are_frozen": True,
        "production_horizon_submission_still_forbidden_until_receipt_passes": True,
    }


def test_planning_geometry_is_exact() -> None:
    modern = 20_182_673_760
    foreign, old = nearest_replay_targets(modern)
    assert (foreign, old) == (5_109_537_661, 255_476_883)
    result = geometry(modern, foreign, old)
    assert result["total_active_tokens"] == 25_547_688_304
    assert result["updates"] == 6092
    assert result["loss_inactive_tail_slots"] == 4_011_664


def test_planning_freezer_fails_closed_with_named_blockers(tmp_path: Path) -> None:
    output = tmp_path / "planning.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "freeze_experiment_contract.py"),
            "--experiment-a-config",
            str(ROOT / "configs/experiment_a_recipe.json"),
            "--experiment-b-config",
            str(ROOT / "configs/experiment_b_recipe.json"),
            "--allocation-config",
            str(ROOT / "configs/allocation_plan.json"),
            "--mode",
            "planning",
            "--output",
            str(output),
        ],
        check=True,
    )
    value = read(output)
    assert value["status"] == "blocked"
    assert "release_internal_polytonic_token_receipt_missing" in value["blockers"]["experiment_a"]
    assert "continuation_b_schedule_receipt_missing" in value["blockers"]["experiment_b"]
    failed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "freeze_experiment_contract.py"),
            "--experiment-a-config",
            str(ROOT / "configs/experiment_a_recipe.json"),
            "--experiment-b-config",
            str(ROOT / "configs/experiment_b_recipe.json"),
            "--allocation-config",
            str(ROOT / "configs/allocation_plan.json"),
            "--mode",
            "launch",
            "--output",
            str(tmp_path / "must_not_exist.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert failed.returncode != 0
    assert not (tmp_path / "must_not_exist.json").exists()


def test_launch_freezer_scopes_blockers_to_the_selected_experiment(tmp_path: Path) -> None:
    # A remains blocked on polytonic data, but a valid B schedule must be able
    # to freeze independently rather than inheriting A's operational blocker.
    ids = np.array([1, 2, 3, 4], dtype="<u8")
    active = np.array([1, 1, 1, 1], dtype="<u2")
    ids_path = tmp_path / "ids.u64"
    active_path = tmp_path / "active.u16"
    ids.tofile(ids_path)
    active.tofile(active_path)
    packed = tmp_path / "packed.json"
    packed.write_text(json.dumps({"global": {"duplicate_sequence_ids": 0}}))
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "schema_version": "apertus_data_order_schedules_v1",
                "status": "completed",
                "packed_corpus_receipt": {"path": str(packed), "sha256": sha256_file(packed)},
                "common_contract": {"same_exact_sequence_multiset": True},
                "arms": [
                    {
                        "arm_id": "D0_mixed",
                        "training_slots": 4,
                        "sequence_ids": {"path": str(ids_path), "sha256": sha256_file(ids_path)},
                        "active_tokens": {"path": str(active_path), "sha256": sha256_file(active_path)},
                    }
                ],
            }
        )
    )
    # The production-sized B schedule is tested separately; this assertion
    # checks selection semantics directly without manufacturing 9B tokens.
    script = (SCRIPTS / "freeze_experiment_contract.py").read_text()
    assert 'selected_names = ("experiment_a", "experiment_b")' in script
    assert "selected_blockers" in script
    assert "any(selected_blockers.values())" in script


def test_continuation_schedule_uses_only_parent_suffix(tmp_path: Path) -> None:
    ids = np.array(
        [
            1,
            2,
            3,
            4,
            (1 << 62) | 10,
            (2 << 62) | 20,
            (1 << 62) | 11,
            (3 << 62) | 30,
            (2 << 62) | 21,
            (2 << 62) | 22,
            (3 << 62) | 31,
            (1 << 62) | 12,
        ],
        dtype="<u8",
    )
    active = np.array([1, 1, 1, 1, 10, 4, 8, 2, 4, 5, 2, 12], dtype="<u2")
    ids_path = tmp_path / "parent.ids"
    active_path = tmp_path / "parent.active"
    ids.tofile(ids_path)
    active.tofile(active_path)
    packed_path = tmp_path / "packed.json"
    packed_path.write_text(json.dumps({"global": {"duplicate_sequence_ids": 0}}), encoding="utf-8")
    parent_path = tmp_path / "schedule.json"
    parent = {
        "schema_version": "apertus_data_order_schedules_v1",
        "status": "completed",
        "packed_corpus_receipt": {"path": str(packed_path), "sha256": sha256_file(packed_path)},
        "common_contract": {"same_exact_sequence_multiset": True},
        "arms": [
            {
                "arm_id": "D0_mixed",
                "training_slots": len(ids),
                "sequence_ids": {"path": str(ids_path), "bytes": ids_path.stat().st_size, "sha256": sha256_file(ids_path)},
                "active_tokens": {"path": str(active_path), "bytes": active_path.stat().st_size, "sha256": sha256_file(active_path)},
            }
        ],
    }
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints/iter_0000001"
    checkpoint_dir.mkdir(parents=True)
    metadata = checkpoint_dir / ".metadata"
    metadata.write_bytes(b"checkpoint-metadata")
    checkpoint_receipt = tmp_path / "checkpoint-receipt.json"
    checkpoint_receipt.write_text(
        json.dumps(
            {
                "schema_version": "megatron_exact_checkpoint_view_v1",
                "iteration": 1,
                "source_checkpoint_root": str(checkpoint_dir.parent),
                "source_files": [
                    {
                        "relative_path": ".metadata",
                        "bytes": metadata.stat().st_size,
                        "sha256": sha256_file(metadata),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_continuation_b_schedule.py"),
            "--parent-schedule-manifest",
            str(parent_path),
            "--parent-checkpoint-receipt",
            str(checkpoint_receipt),
            "--parent-checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-iteration",
            "1",
            "--global-batch-sequences",
            "4",
            "--expected-modern-active-tokens",
            "30",
            "--window-count",
            "4",
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    manifest = read(output / "schedule_manifest.json")
    continuation = manifest["continuation_contract"]
    assert continuation["realized_pool_active_tokens"] == {"F": 8, "G": 30, "O": 0}
    assert continuation["parent_prefix_overlap_selected_sequences"] == 0
    arm = manifest["arms"][0]
    assert arm["arm_id"] == "D0_mixed"
    hybrid = np.fromfile(arm["sequence_ids"]["path"], dtype="<u8")
    assert np.array_equal(hybrid[:4], ids[:4])
    selected = hybrid[4 : 4 + manifest["continuation_contract"]["continuation_real_sequences"]]
    assert not set(map(int, selected)).intersection(map(int, ids[:4]))
    assert manifest["continuation_contract"]["continuation_optimizer_updates"] == 2
    assert arm["optimizer_updates"] == 3


def test_continuation_pool_view_materializes_parent_packed_receipts(tmp_path: Path) -> None:
    packed = tmp_path / "parent-packed.json"
    packed.write_text(json.dumps({"global": {"duplicate_sequence_ids": 0}}), encoding="utf-8")
    parent_pool = tmp_path / "parent-pool.json"
    parent_pool.write_text(
        json.dumps(
            {
                "schema_version": "apertus_schedule_pool_corpus_v1",
                "status": "completed",
                "source_root": str(tmp_path / "parent-data"),
                "tokenizer": {"sha256": "tokenizer"},
                "sorted_training_catalogs": {"H": "catalog"},
            }
        ),
        encoding="utf-8",
    )
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema_version": "apertus_data_order_schedules_v1",
                "status": "completed",
                "packed_corpus_receipt": {
                    "path": str(packed),
                    "bytes": packed.stat().st_size,
                    "sha256": sha256_file(packed),
                },
                "continuation_contract": {
                    "parent_prefix_is_byte_exact": True,
                    "parent_prefix_overlap_selected_sequences": 0,
                },
                "arms": [
                    {
                        "arm_id": "D0_mixed",
                        "pool_active_tokens": {"H": 4, "G": 79, "F": 20, "O": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    integrity = tmp_path / "parent-integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": "apertus_full_8b_packed_payload_integrity_v1",
                "status": "passed",
                "packed_receipt": {"sha256": sha256_file(packed)},
            }
        ),
        encoding="utf-8",
    )
    pool_output = tmp_path / "stage/inventory/pool.json"
    packed_output = tmp_path / "stage/packed.json"
    integrity_output = tmp_path / "stage/integrity.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_continuation_b_pool_view.py"),
            "--parent-pool-receipt",
            str(parent_pool),
            "--hybrid-schedule-manifest",
            str(schedule),
            "--packed-output",
            str(packed_output),
            "--parent-packed-integrity",
            str(integrity),
            "--packed-integrity-output",
            str(integrity_output),
            "--output",
            str(pool_output),
        ],
        check=True,
    )
    value = read(pool_output)
    assert value["integer_79_20_1_geometry"]["active_tokens"] == 104
    assert value["invariants"]["global_deduplication_performed"] is False
    assert packed_output.read_bytes() == packed.read_bytes()
    assert integrity_output.read_bytes() == integrity.read_bytes()


def test_continuation_builder_v1_snapshot_is_preserved() -> None:
    snapshot = read(ROOT / "configs/continuation_data_builder_v1.json")
    assert snapshot["schema_version"] == "apertus_targeted_continuation_data_builder_v1"
    assert snapshot["status"] == "frozen"
    for implementation in snapshot["implementations"]:
        assert sha256_file(ROOT / implementation["path"]) == implementation["sha256"]
    policy = snapshot["fixed_v1_policy"]
    assert policy["selected_suffix_pools"] == ["G", "F", "O"]
    assert policy["parent_prefix_byte_exact"] is True
    assert policy["parent_prefix_overlap_allowed"] is False
    assert policy["global_deduplication_performed"] is False
    versioning = snapshot["versioning_policy"]
    assert versioning["edit_v1_in_place_for_new_mix"] is False
    assert versioning["new_mix_requires_new_policy_id"] is True
    assert versioning["new_mix_requires_new_tests_receipts_and_launch_gate"] is True


def test_named_source_extraction_preserves_order_and_duplicates(tmp_path: Path) -> None:
    release = tmp_path / "release-root"
    data = release / "release/data"
    manifests = release / "release/manifests"
    data.mkdir(parents=True)
    manifests.mkdir(parents=True)
    table = pa.table(
        {
            "source_dataset": ["openarchives.gr", "other", "openarchives.gr"],
            "source_doc_id": ["same", "x", "same"],
            "text": ["alpha", "skip", "alpha"],
        }
    )
    shard = data / "000000.parquet"
    pq.write_table(table, shard)
    (manifests / "anonymization_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "data/000000.parquet",
                        "rows": 3,
                        "bytes": shard.stat().st_size,
                        "sha256": sha256_file(shard),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (manifests / "token_counts.json").write_text(
        json.dumps({"source_rows": {"openarchives.gr": 2}}), encoding="utf-8"
    )
    output = tmp_path / "selected"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "extract_release_sources.py"),
            "--release-root",
            str(release),
            "--source",
            "openarchives.gr",
            "--output-root",
            str(output),
            "--workers",
            "1",
        ],
        check=True,
    )
    selected = pq.read_table(output / "data/000000.parquet").to_pydict()
    assert selected["source_doc_id"] == ["same", "same"]
    assert selected["text"] == ["alpha", "alpha"]
    receipt = read(output / "extraction_manifest.json")
    assert receipt["invariants"]["global_deduplication_performed"] is False
    assert receipt["rows"] == 2
    assert receipt["outputs"][0]["output"]["path"] == str((output / "data/000000.parquet").resolve())
    assert not list(tmp_path.glob(".selected.*.partial"))


def test_source_sampling_is_identity_deterministic_and_near_quarter() -> None:
    seed = 20260811
    first = deterministic_bucket(seed, "HPLT/ell_Grek_ge8_no_mt_clean60", "same", 4)
    second = deterministic_bucket(seed, "HPLT/ell_Grek_ge8_no_mt_clean60", "same", 4)
    assert first == second
    selected = sum(
        deterministic_bucket(seed, "HPLT/ell_Grek_ge8_no_mt_clean60", f"doc-{index}", 4) < 1
        for index in range(10_000)
    )
    assert 2300 <= selected <= 2700


def test_anonymized_release_decontamination_adapter_preserves_kept_duplicates(tmp_path: Path) -> None:
    class FakeCanonical:
        @staticmethod
        def match_document(text: str, _index: object) -> tuple[str, str, list[dict]]:
            if text == "benchmark leak":
                return "drop", "greekmmlu_exact_prompt", [{"benchmark_item_id": "q1"}]
            return "keep", "no_high_confidence_match", []

    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "source_dataset": ["openarchives.gr"] * 3,
                "source_doc_id": ["duplicate", "duplicate", "leak"],
                "text": ["same text", "same text", "benchmark leak"],
            }
        ),
        source,
    )
    targeted_decontam._CANONICAL = FakeCanonical()
    targeted_decontam._INDEX = object()
    result = targeted_decontam.process_file(
        (
            str(source),
            "part.parquet",
            str(tmp_path / "kept"),
            str(tmp_path / "dropped"),
            str(tmp_path / "ledger"),
        )
    )
    assert result["counts"] == {"input": 3, "kept": 2, "dropped": 1}
    kept = pq.read_table(tmp_path / "kept/part.parquet").to_pydict()
    dropped = pq.read_table(tmp_path / "dropped/part.parquet").to_pydict()
    assert kept["source_doc_id"] == ["duplicate", "duplicate"]
    assert kept["text"] == ["same text", "same text"]
    assert dropped["source_doc_id"] == ["leak"]
    ledger = pq.read_table(tmp_path / "ledger/part.parquet").to_pydict()
    assert ledger["action"] == ["keep", "keep", "drop"]


def test_decontamination_wrappers_do_not_precreate_poisonable_output_roots() -> None:
    for name in ("academic", "hplt", "poly"):
        wrapper = (ROOT / f"clariden/decontaminate_{name}_debug.sbatch").read_text()
        assert 'mkdir -p "$root"' not in wrapper
    adapter = (SCRIPTS / "decontaminate_targeted_corpus.py").read_text()
    assert "args.manifest.parent.rmdir()" in adapter


def test_decontamination_audit_reconciles_each_shard_without_collapsing_duplicates(tmp_path: Path) -> None:
    class FakeCanonical:
        @staticmethod
        def match_document(text: str, _index: object) -> tuple[str, str, list[dict]]:
            if text == "benchmark leak":
                return "drop", "greekmmlu_exact_prompt", [{"benchmark_item_id": "q1"}]
            return "keep", "no_high_confidence_match", []

    identity_columns = {
        "source_dataset": ["openarchives.gr", "openarchives.gr", "openarchives.gr"],
        "source_doc_id": ["duplicate", "duplicate", "leak"],
        "text": ["same text", "same text", "benchmark leak"],
    }
    source = tmp_path / "input.parquet"
    kept = tmp_path / "kept.parquet"
    dropped = tmp_path / "dropped.parquet"
    ledger = tmp_path / "ledger.parquet"
    pq.write_table(pa.table(identity_columns), source)
    pq.write_table(pa.table({key: value[:2] for key, value in identity_columns.items()}), kept)
    pq.write_table(pa.table({key: value[2:] for key, value in identity_columns.items()}), dropped)
    pq.write_table(
        pa.table(
            {
                "source_dataset": identity_columns["source_dataset"],
                "source_doc_id": identity_columns["source_doc_id"],
                "input_text_sha256": [
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for text in identity_columns["text"]
                ],
                "action": ["keep", "keep", "drop"],
            }
        ),
        ledger,
    )
    decontam_audit._CANONICAL = FakeCanonical()
    decontam_audit._INDEX = object()
    result = decontam_audit.reconcile_and_rescan(
        {
            "relative_path": "part.parquet",
            "input": str(source),
            "output": decontam_audit.file_binding(kept),
            "dropped": decontam_audit.file_binding(dropped),
            "ledger": decontam_audit.file_binding(ledger),
        }
    )
    assert result["input"] == 3
    assert result["kept"] == 2
    assert result["dropped"] == 1
    assert result["ledger"] == 3
    assert result["actions"] == {"keep": 2, "drop": 1}
    assert result["remaining"] == []


def test_bundle_contains_importable_canonical_greekmmlu_policy() -> None:
    canonical_path = (
        ROOT.parents[1]
        / "subprojects/05_token_distillation_cpt/04_full_corpus_preparation/scripts/decontaminate_full_corpus.py"
    )
    assert canonical_path.is_file()
    canonical = targeted_decontam.load_canonical(canonical_path)
    assert canonical.POLICY_VERSION == "greekmmlu_decontamination_v1"
    assert canonical.DEFAULT_K == 8
    assert canonical.DEFAULT_MIN_COVERAGE == 0.85
    assert canonical.DEFAULT_MINHASH_THRESHOLD == 0.85


def test_validation_exclusion_is_not_training_deduplication(tmp_path: Path) -> None:
    heldout_text = "held out"
    validation_exclusion._HELDOUT_HASHES = {
        hashlib.sha256(heldout_text.encode("utf-8")).hexdigest()
    }
    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "source_dataset": ["greek_phd"] * 3,
                "source_doc_id": ["duplicate", "duplicate", "heldout"],
                "text": ["same training text", "same training text", heldout_text],
            }
        ),
        source,
    )
    result = validation_exclusion.process_file(
        (
            str(source),
            "part.parquet",
            str(tmp_path / "eligible"),
            str(tmp_path / "excluded"),
            str(tmp_path / "ledger"),
        )
    )
    assert result["counts"] == {"input": 3, "kept": 2, "excluded": 1}
    kept = pq.read_table(tmp_path / "eligible/part.parquet").to_pydict()
    assert kept["source_doc_id"] == ["duplicate", "duplicate"]
    excluded = pq.read_table(tmp_path / "excluded/part.parquet").to_pydict()
    assert excluded["source_doc_id"] == ["heldout"]
    validation_audit._HELDOUT_HASHES = set(validation_exclusion._HELDOUT_HASHES)
    audited = validation_audit.reconcile_shard(
        {
            "relative_path": "part.parquet",
            "input": str(source),
            "kept": validation_audit.file_binding(tmp_path / "eligible/part.parquet"),
            "excluded": validation_audit.file_binding(tmp_path / "excluded/part.parquet"),
            "ledger": validation_audit.file_binding(tmp_path / "ledger/part.parquet"),
        }
    )
    assert audited == {
        "relative_path": "part.parquet",
        "input": 3,
        "kept": 2,
        "excluded": 1,
        "ledger": 3,
        "kept_overlap": 0,
        "excluded_nonoverlap": 0,
    }


def test_catalog_identity_preserves_duplicate_rows_as_distinct_records() -> None:
    first = digest16("targeted8b-record-v1", 845, 0, "same-doc", "same-text-sha")
    second = digest16("targeted8b-record-v1", 845, 1, "same-doc", "same-text-sha")
    retry = digest16("targeted8b-record-v1", 845, 0, "same-doc", "same-text-sha")
    assert first != second
    assert first == retry


def test_hplt_pipeline_consumers_use_the_candidate_pool_paths() -> None:
    expected = "hplt_candidates_training_eligible"
    freeze_inputs = (ROOT / "clariden/freeze_binary_inputs_debug.sbatch").read_text()
    finalize_selection = (ROOT / "clariden/finalize_selected_pools_debug.sbatch").read_text()
    assert expected in freeze_inputs
    assert expected in finalize_selection
    assert "hplt_candidates_decontaminated" in finalize_selection
    assert "experiment_a/hplt_training_eligible" not in freeze_inputs
    assert "$stage/hplt_training_eligible" not in finalize_selection
    assert "$stage/hplt_decontaminated" not in finalize_selection


def test_binary_bridge_preserves_positional_parent_task_slots() -> None:
    freezer = (SCRIPTS / "freeze_targeted_binary_inputs.py").read_text()
    catalog = (SCRIPTS / "finalize_targeted_pool_catalogs.py").read_text()
    assert '"task_origin": "inherited_parent_binary"' in freezer
    assert '"task_origin": "targeted_new_modern"' in freezer
    assert "list(range(len(payload[\"tasks\"])))" in freezer
    assert 'task.get("task_origin") != "targeted_new_modern"' in catalog
    assert 'require(not catalog_root.exists()' in catalog
    assert "tempfile.mkdtemp" in catalog
    assert "os.rename(temporary_catalog_root, catalog_root)" in catalog


def test_parallel_binary_batch_rejects_inherited_task_positions(tmp_path: Path) -> None:
    contract = tmp_path / "experiment_a/binary_contract/input_receipt.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_index": 0, "task_origin": "inherited_parent_binary", "output_prefix": "/parent/zero"},
                    {"task_index": 1, "task_origin": "targeted_new_modern", "output_prefix": "source_binary/one"},
                ]
            }
        )
    )
    common = {
        "kind": "binary",
        "code_root": tmp_path / "code",
        "stage_root": tmp_path,
        "data_python": Path(sys.executable),
        "end": 2,
        "parallel": 1,
        "workers_per_task": 1,
    }
    inherited = Namespace(**common, start=0)
    try:
        task_contract(inherited)
    except ValueError as error:
        assert "inherited task" in str(error)
    else:
        raise AssertionError("inherited task range was accepted")
    targeted = Namespace(**common, start=1)
    selected, frozen_contract, _ = task_contract(targeted)
    assert [row["task_index"] for row in selected] == [1]
    assert frozen_contract == contract


def test_targeted_launcher_preserves_allocation_and_resume_contracts() -> None:
    launcher = (ROOT / "clariden/submit_targeted_production.sh").read_text()
    assert "--partition=normal --time=12:00:00 --switches=1" in launcher
    assert '--nodes="$nodes"' in launcher
    assert "sbatch --test-only" in launcher
    assert "--eligible-after-minutes 500" in launcher
    assert "--maximum-hold-seconds 6000" in launcher
    assert "test-only mutated prequeue graph" in launcher
    assert 'ln -s "$source_checkpoint" "$FULL8_RUN_ROOT/checkpoints/iter_0009536"' in launcher
    assert "printf '9536\\n' >\"$FULL8_RUN_ROOT/checkpoints/latest_checkpointed_iteration.txt\"" in launcher
    assert "FULL8_LOAD_CHECKPOINT=$launch_initial_megatron" in launcher
    assert "FULL8_EXACT_LOAD_ITERATION" not in launcher


def test_experiment_a_materializes_only_the_consumed_stationary_schedule() -> None:
    wrapper = (ROOT / "clariden/build_experiment_a_schedule_debug.sbatch").read_text()
    builder = (ROOT.parent / "06_dataset_scheduling_experiments/dataset/build_five_schedules.py").read_text()
    assert "--arm D0_mixed" in wrapper
    assert 'action="append"' in builder
    assert "os.rename(output_root, args.output_dir)" in builder


def test_b_asset_chain_preserves_gate_order_and_stays_on_debug() -> None:
    wrapper = (ROOT / "clariden/prepare_continuation_b_assets_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    pool = wrapper.index("build_continuation_b_pool_view_debug.sbatch")
    training = wrapper.index("freeze_training_assets_debug.sbatch")
    contract = wrapper.index("freeze_experiment_contract_debug.sbatch")
    assert pool < training < contract
    assert "build_prequeue_schedule" not in wrapper


def test_all_panel_validation_reuses_proven_groups_transactionally_on_debug() -> None:
    wrapper = (ROOT / "clariden/run_all_per_document_groups_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "#SBATCH --nodes=1" in wrapper
    assert "#SBATCH --gpus-per-node=4" in wrapper
    assert 'for group in 0 1 2 3' in wrapper
    assert 'run_per_document_group_resource_aware.sh' in wrapper
    assert 'final_output.partial.${SLURM_JOB_ID' in wrapper
    assert 'mv "$staging" "$final_output"' in wrapper
    assert '{"completed", "frozen", "passed"}' in wrapper
    assert 'len(receipts) != 13 or len(documents) != 13' in wrapper


def test_nested_submit_rebind_uses_debug_for_parent_and_child() -> None:
    parent = (ROOT / "clariden/prove_nested_sbatch_debug.sbatch").read_text()
    child = (ROOT / "clariden/nested_sbatch_child_debug.sbatch").read_text()
    submitter = (SCRIPTS / "submit_nested_sbatch_probe_debug.py").read_text()
    assert "#SBATCH --partition=debug" in parent
    assert "#SBATCH --partition=debug" in child
    assert "#SBATCH --partition=normal" not in parent + child
    assert '"--partition=debug"' in submitter
    assert '"--uenv-passthrough=ignore"' in submitter
    assert "uenv run pytorch/v2.9.1:v2 --view=default -- python3" in parent
    assert "uenv run pytorch/v2.9.1:v2 --view=default -- torchrun" in child
    assert "verify_code_bundle.py" in child + submitter


def test_b_checkpoint_authority_is_the_existing_exact_tree_receipt() -> None:
    config = read(ROOT / "configs/experiment_b_recipe.json")
    assert config["parent"]["checkpoint_receipt"].endswith(
        "/checkpoint_evaluations/iter_0009536/attempt_0/export/source_checkpoint_receipt.json"
    )
    builder = (SCRIPTS / "build_continuation_b_schedule.py").read_text()
    freezer = (SCRIPTS / "freeze_experiment_contract.py").read_text()
    assets = (SCRIPTS / "freeze_training_assets.py").read_text()
    assert "--parent-checkpoint-receipt" in builder
    assert 'checkpoint_rows[0].get("relative_path") == ".metadata"' in builder
    assert "B parent checkpoint receipt binding drift" in freezer
    assert 'recipe["initialization"]["targeted_resume"]' in assets


def test_frozen_validation_and_launch_evidence_fail_closed() -> None:
    assets = (SCRIPTS / "freeze_training_assets.py").read_text()
    gate = (SCRIPTS / "build_targeted_launch_gate.py").read_text()
    assert 'validation.get("status") == "frozen"' in assets
    assert "verify_code_bundle_receipt(args.code_bundle_receipt" in gate
    assert '"apertus_mini_immutable_code_bundle_v1"' not in gate
    assert "validation overlap-audit binding drift" in gate
    assert "per-document evidence binding drift" in gate
    assert "initial checkpoint tree hash drift" in gate
    assert '"position_embedding_type": "rope"' in gate
    assert '"base": 500000' in gate
    assert '"scaling_factor": 8.0' in gate
    assert 'binding_matches(initial_validation_manifest, initial_validation_manifest_path)' in gate
    assert 'initial_validation_manifest.get("sha256") == sha256_file(args.validation_manifest)' in gate
    assert "first_post_checkpoint_update_within_frozen_bounds" in gate
    assert 'source-validation cadence receipt/executable drift' in gate
    assert 'GreekMMLU cadence receipt/checkpoint-plan drift' in gate


def test_training_asset_receipt_matches_the_executable_validation_cadence() -> None:
    freezer = (SCRIPTS / "freeze_training_assets.py").read_text()
    trainer = (ROOT.parent / "07_full_8b_cpt/clariden/train_segment.sbatch").read_text()
    assert "SOURCE_CONDITIONED_INTERVAL_UPDATES = 25" in freezer
    assert 'source_validation["interval_updates"] = SOURCE_CONDITIONED_INTERVAL_UPDATES' in freezer
    assert "104_857_600" not in freezer
    assert '"cadence_active_tokens": 2_000_000_000' in freezer
    assert '"cadence_active_tokens": 1_000_000_000' in freezer
    literal_cadence = "--eval-interval 25" in trainer
    receipt_bound_cadence = (
        '--eval-interval "${PROFILE[eval_interval]}"' in trainer
        and 'print(f"eval_interval\\t{r[\'evaluation\'][\'source_conditioned\'][\'interval_updates\']}")' in trainer
    )
    assert literal_cadence or receipt_bound_cadence


def test_bundle_deployer_overlays_only_the_targeted_subproject() -> None:
    deployer = (ROOT / "clariden/deploy_targeted_bundle.sh").read_text()
    assert "cp -a \"$PROVEN_BASE\" \"$REMOTE_ROOT\"" in deployer
    assert 'relative=subprojects/08_targeted_8b_cpt_experiments' in deployer
    assert 'rsync -a --delete' in deployer
    assert "train_segment_targeted_benchmark_offset.patch" in deployer
    assert 'chmod -R a-w "$REMOTE_ROOT"' in deployer


def test_retired_experiment_b_is_rejected_by_production_submitter() -> None:
    submitter = (ROOT / "clariden/submit_targeted_production.sh").read_text()
    assert 'recipe.get("launch_authorized") is not True' in submitter
    assert "Experiment B was retired by the owner" in submitter


def test_corrected_initial_hf_view_changes_only_geometry_and_hardlinks_payloads(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "roundtrip"
    source.mkdir()
    source_config = {
        "rope_theta": 12_000_000,
        "max_position_embeddings": 65_536,
        "tie_word_embeddings": False,
        "vocab_size": 148_992,
        "architectures": ["ApertusForCausalLM"],
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    tokenizer = {"model": {"vocab": {"λόγος": 1}}, "normalizer": None}
    for name, value in {
        "generation_config.json": {"do_sample": False},
        "model.safetensors.index.json": {"weight_map": {}},
        "special_tokens_map.json": {"pad_token": "<pad>"},
        "tokenizer_config.json": {"model_max_length": 4096},
        "tokenizer.json": tokenizer,
    }.items():
        (source / name).write_text(json.dumps(value), encoding="utf-8")
    for index in range(1, 5):
        (source / f"model-{index:05d}-of-00004.safetensors").write_bytes(f"tensor-{index}".encode())
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2), encoding="utf-8")
    monkeypatch.setattr(corrected_initial_hf, "TOKENIZER_SHA256", sha256_file(frozen / "tokenizer.json"))
    roundtrip = tmp_path / "roundtrip.json"
    roundtrip.write_text(
        json.dumps(
            {
                "standard_max_abs_diff": 0.0,
                "r17_max_abs_diff": 0.0,
                "xielu_max_abs_diff": 0.0,
                "qk_norm_max_abs_diff": 0.0,
                "orig_only": [],
                "trip_only": [],
                "shape_mismatches": [],
                "r17_changed_over_tol_count": 0,
                "standard_changed_over_tol_count": 0,
                "logits": {"logit_max_abs_diff": 0.0, "logit_mean_abs_diff_max": 0.0, "per_prompt": [{"top_id_match": True}]},
            }
        ),
        encoding="utf-8",
    )
    td = tmp_path / "td.json"
    td.write_text(
        json.dumps(
            {
                "schema_version": "production_polytonic_td_init_verification_v1",
                "status": "passed",
                "existing_input_rows_exact": True,
                "existing_output_rows_exact": True,
                "non_embedding_tensors_exact": True,
                "new_rows_finite": True,
                "new_rows_nonzero": True,
                "target_layer": 11,
                "tokenizer_json_sha256": sha256_file(frozen / "tokenizer.json"),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "model"
    output.parent.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare", "--source-model-root", str(source), "--roundtrip-verification", str(roundtrip),
            "--td-initialization-verification", str(td), "--frozen-tokenizer-dir", str(frozen),
            "--output-root", str(output),
        ],
    )
    assert corrected_initial_hf.main() == 0
    target_config = read(output / "config.json")
    assert {key for key in target_config if target_config[key] != source_config.get(key)} == {
        "rope_theta", "max_position_embeddings"
    }
    assert target_config["rope_theta"] == 500_000
    assert target_config["max_position_embeddings"] == 4_096
    assert (source / "model-00001-of-00004.safetensors").stat().st_ino == (output / "model-00001-of-00004.safetensors").stat().st_ino
    receipt = read(output.parent / "corrected_initial_hf_receipt.json")
    assert receipt["zero_tensor_and_logit_drift"] is True
    assert receipt["model_and_support_files_hardlinked_to_zero_drift_source"] is True
    assert receipt["tokenizer_semantically_identical_to_roundtrip"] is True


def test_corrected_initial_hf_builder_is_debug_only_and_receipt_bound() -> None:
    wrapper = (ROOT / "clariden/prepare_corrected_initial_hf_debug.sbatch").read_text()
    script = (ROOT / "scripts/prepare_corrected_initial_hf.py").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "verify_code_bundle.py" in wrapper
    assert "TARGET8_CORRECTED_INITIAL_HF_ROOT" in wrapper
    assert '"schema_version": "apertus_full_8b_corrected_initial_hf_v1"' in script
    assert "os.link(source_path, target_path)" in script
    assert "corrected_config_keys" in script


def test_initial_greekmmlu_wrapper_is_debug_only_and_verifies_scientific_bundle() -> None:
    wrapper = (ROOT / "clariden/run_initial_greekmmlu_debug.sbatch").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "#SBATCH --gpus-per-node=1" in wrapper
    assert "verify_code_bundle.py" in wrapper
    assert "TARGET8_CODE_RECEIPT" in wrapper
    assert 'exec bash "$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch"' in wrapper


def test_targeted_initial_greekmmlu_finalizer_binds_pinned_dataset_content() -> None:
    wrapper = (ROOT / "clariden/finalize_initial_greekmmlu_debug.sbatch").read_text()
    script = (ROOT / "scripts/finalize_targeted_initial_greekmmlu.py").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "verify_code_bundle.py" in wrapper
    assert "TARGET8_EVALUATION_CODE_RECEIPT" in wrapper
    assert '"source": DATASET' in script
    assert '"revision": REVISION' in script
    assert "content_fingerprint(source_rows)" in script
    assert "prediction/dataset binding drift" in script
    assert "legacy/decontaminated GreekMMLU subset binding drift" in script
    assert "evaluated_scientific_bundle" in script


def test_targeted_greekmmlu_content_fingerprint_is_canonical_and_order_sensitive() -> None:
    first = [{"question": "α", "answer": 0}, {"question": "β", "answer": 1}]
    equivalent = [{"answer": 0, "question": "α"}, {"answer": 1, "question": "β"}]
    reversed_rows = list(reversed(first))
    assert targeted_greekmmlu.content_fingerprint(first) == targeted_greekmmlu.content_fingerprint(equivalent)
    assert targeted_greekmmlu.content_fingerprint(first) != targeted_greekmmlu.content_fingerprint(reversed_rows)


def test_targeted_restart_smoke_is_exact_dp32_and_bounded_to_three_updates() -> None:
    runner = (ROOT / "clariden/run_targeted_restart_smoke.sbatch").read_text()
    submitter = (ROOT / "clariden/submit_targeted_restart_smoke.sh").read_text()
    finalizer = (SCRIPTS / "finalize_targeted_restart_smoke.py").read_text()
    patch = (ROOT / "patches/train_segment_targeted_benchmark_offset.patch").read_text()
    assert "#SBATCH --partition=normal" in runner
    assert "#SBATCH --nodes=16" in runner
    assert runner.count("run_train ") == 2
    assert "FULL8_BENCHMARK_BASE_ITERATION" in runner
    assert '"$TARGET8_INITIAL_CHECKPOINT_ROOT" "$checkpoint" "$exact_initial"' in runner
    assert '--validation-manifest "$stage/validation/validation_manifest.json"' in runner
    assert "sbatch --test-only" in submitter
    assert 'if [[ "$TARGET8_LEAF_SWITCH" != auto ]]' in submitter
    assert '--switches=1 "${exclude_args[@]}"' in submitter
    assert "distributed smoke submitted before debug asset freeze" in submitter
    assert '"total_optimizer_updates_executed": 3' in finalizer
    assert '"comparison_design": "resume_from_exact_uninterrupted_control_checkpoint"' in finalizer
    assert '"code_bundle_receipt": file_binding(args.code_bundle_receipt)' in finalizer
    assert '--code-bundle-receipt "$TARGET8_CODE_RECEIPT"' in runner
    assert '"control_checkpoint_boundary_proven"' in finalizer
    assert '"all_frozen_panels_ran_before_control_checkpoint"' in finalizer
    assert "checkpoint_sample_cursor_exact" in finalizer
    assert "benchmark_base <= start < end <= benchmark_base + 288" in patch


def test_b_after_restart_controller_is_debug_only_and_fail_closed() -> None:
    wrapper = (ROOT / "clariden/finalize_and_submit_b_after_restart_debug.sbatch").read_text()
    derive = (SCRIPTS / "derive_completed_job_leaf.py").read_text()
    assert "#SBATCH --partition=debug" in wrapper
    assert "#SBATCH --time=01:30:00" in wrapper
    assert "derive_completed_job_leaf.py" in wrapper
    assert "capture_launch_environment.py" in wrapper
    launch = wrapper.index("build_targeted_launch_gate.py")
    operational = wrapper.index("build_targeted_operational_gate.py")
    submit = wrapper.index("submit_targeted_production.sh")
    assert launch < operational < submit
    assert "DRY_RUN=0 CONFIRM_GPU_LAUNCH=TARGETED8_CPT_B" in wrapper
    assert 'state == "COMPLETED" and exit_code == "0:0"' in derive
    assert 'len(nodes) == 16' in derive
    assert 'len(matches) == 1' in derive


def test_packed_payload_audit_hashes_every_output(tmp_path: Path) -> None:
    outputs = {}
    for name, payload in (("bin", b"bin"), ("idx", b"idx"), ("active_counts", b"active")):
        path = tmp_path / name
        path.write_bytes(payload)
        outputs[name] = {"path": str(path), "bytes": len(payload), "sha256": sha256_file(path)}
    manifest_path = tmp_path / "bucket.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "apertus_fixed_sequence_bucket_v1",
                "status": "completed",
                "task_index": 0,
                "outputs": outputs,
            }
        )
    )
    packed_path = tmp_path / "packed.json"
    packed_path.write_text(
        json.dumps(
            {
                "schema_version": "apertus_packed_sequence_corpus_v1",
                "status": "completed",
                "packing_task_manifests": [
                    {
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": sha256_file(manifest_path),
                    }
                ],
            }
        )
    )
    output = tmp_path / "integrity.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_packed_payload_integrity.py"),
            "--packed-receipt",
            str(packed_path),
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        check=True,
    )
    receipt = read(output)
    assert receipt["status"] == "passed"
    assert receipt["manifest_count"] == 1
    assert receipt["payload_count"] == 3

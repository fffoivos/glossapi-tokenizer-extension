from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINALIZE = load_module("phase04_finalize_acquisition", HERE / "scripts" / "finalize_acquisition.py")
INPUT = load_module("phase04_validate_input_receipt", HERE / "scripts" / "validate_input_receipt.py")
BUILD = load_module("phase04_write_detector_build_receipt", HERE / "scripts" / "write_detector_build_receipt.py")
VALIDATE_BUILD = load_module(
    "phase04_validate_detector_build_receipt",
    HERE / "scripts" / "validate_detector_build_receipt.py",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def acquisition_fixture(tmp_path: Path) -> dict[str, Path]:
    revision = "1" * 40
    config_path = tmp_path / "sources.json"
    config = {
        "schema_version": "full_cpt_sources_v1",
        "base": {
            "repo_id": "owner/base",
            "repo_type": "dataset",
            "revision": revision,
            "include_globs": ["data/*.parquet"],
            "role": "base",
            "text_columns": ["text"],
            "id_columns": ["source_doc_id"],
            "source_column": "source_dataset",
        },
        "apertus_overlap_overlay": {
            "repo_id": "owner/overlay",
            "repo_type": "dataset",
            "revision": "2" * 40,
            "include_globs": ["summary.json"],
            "role": "base_overlay",
        },
        "tokenizer": {
            "repo_id": "owner/tokenizer",
            "repo_type": "model",
            "revision": "3" * 40,
            "include_globs": ["tokenizer.json"],
        },
        "embedded_structural_routes": [
            {
                "source_id": "greek_phd",
                "acquisition_source_id": "nanochat_base",
                "acquisition_include_globs": ["data/greek_phd*.parquet"],
                "input_scope": "canonical_mixed",
                "source_regex": "^greek_phd",
                "text_columns": ["text"],
                "id_columns": ["source_doc_id"],
                "source_column": "source_dataset",
            }
        ],
        "sources": [],
    }
    write_json(config_path, config)

    destination = tmp_path / "staged"
    parquet = destination / "nanochat_base" / revision / "data" / "greek_phd-00000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"fixture parquet payload")
    parquet_stat = parquet.stat()
    payload_hash = hashlib.sha256(parquet.read_bytes()).hexdigest()
    selected = {
        "path": "data/greek_phd-00000.parquet",
        "size": parquet.stat().st_size,
        "blob_id": "f" * 40,
        "lfs_sha256": payload_hash,
        "lfs_size": parquet.stat().st_size,
    }
    lock_path = tmp_path / "sources.lock.json"
    lock = {
        "schema_version": "full_cpt_sources_lock_v1",
        "sources_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sources": [
            {
                "source_id": "nanochat_base",
                "repo_id": "owner/base",
                "repo_type": "dataset",
                "revision": revision,
                "role": "base",
                "selected_files": [selected],
                "selected_file_count": 1,
                "selected_bytes": parquet.stat().st_size,
            }
        ],
    }
    write_json(lock_path, lock)

    download_path = tmp_path / "sources.download.json"
    download = {
        "schema_version": "full_cpt_download_manifest_v1",
        "source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "destination": str(destination.resolve()),
        "sources": [
            {
                "source_id": "nanochat_base",
                "repo_id": "owner/base",
                "repo_type": "dataset",
                "revision": revision,
                "local_dir": str(parquet.parents[1].resolve()),
                "files": 1,
                "bytes": parquet.stat().st_size,
                "lfs_sha256_verified": 1,
                "git_blob_ids_verified": 0,
                "verified_files": [
                    {
                        "path": selected["path"],
                        "size": selected["size"],
                        "hash_kind": "lfs_sha256",
                        "expected_hash": payload_hash,
                        "actual_hash": payload_hash,
                        "device": parquet_stat.st_dev,
                        "inode": parquet_stat.st_ino,
                        "mtime_ns": parquet_stat.st_mtime_ns,
                        "ctime_ns": parquet_stat.st_ctime_ns,
                    }
                ],
            }
        ],
    }
    write_json(download_path, download)

    schema_path = tmp_path / "sources.schemas.json"
    schema = {
        "schema_version": "full_cpt_staged_schema_audit_v1",
        "ok": True,
        "sources_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "destination": str(destination.resolve()),
        "sources": [{"source_id": "nanochat_base", "status": "ok"}],
        "skipped_artifacts": [],
    }
    write_json(schema_path, schema)

    receipt_path = tmp_path / "sources.receipt.json"
    receipt = FINALIZE.build_receipt(
        sources_path=config_path,
        lock_path=lock_path,
        download_manifest_path=download_path,
        schema_audit_path=schema_path,
        destination=destination,
        code_commit="a" * 40,
    )
    write_json(receipt_path, receipt)
    return {
        "config": config_path,
        "lock": lock_path,
        "receipt": receipt_path,
        "parquet": parquet,
    }


def validate_fixture(paths: dict[str, Path], *inputs: Path) -> dict:
    return INPUT.validate_input(
        receipt_path=paths["receipt"],
        sources_path=paths["config"],
        source_id="greek_phd",
        input_values=[str(path) for path in inputs],
        text_column="text",
        id_column="source_doc_id",
        source_column="source_dataset",
    )


def test_acquisition_receipt_binds_verified_payload_and_exact_route(tmp_path: Path) -> None:
    paths = acquisition_fixture(tmp_path)
    check = validate_fixture(paths, paths["parquet"])
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    recorded = receipt["sources"][0]["files"][0]
    assert check["ok"] is True
    assert check["paths"] == [str(paths["parquet"].resolve())]
    assert recorded["hash_kind"] == "lfs_sha256"
    assert recorded["expected_hash"] == hashlib.sha256(paths["parquet"].read_bytes()).hexdigest()


def test_launch_gate_rejects_extra_path_size_and_metadata_drift(tmp_path: Path) -> None:
    paths = acquisition_fixture(tmp_path)
    extra = paths["parquet"].with_name("greek_phd-extra.parquet")
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="path set"):
        validate_fixture(paths, paths["parquet"].parent)
    extra.unlink()

    paths["parquet"].write_bytes(paths["parquet"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size drifted"):
        validate_fixture(paths, paths["parquet"])

    paths = acquisition_fixture(tmp_path / "mtime")
    file_stat = paths["parquet"].stat()
    os.utime(paths["parquet"], ns=(file_stat.st_atime_ns, file_stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="mtime_ns drifted"):
        validate_fixture(paths, paths["parquet"])


def test_finalizer_rejects_lock_metadata_not_bound_to_config(tmp_path: Path) -> None:
    paths = acquisition_fixture(tmp_path)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    lock["sources"][0]["repo_id"] = "attacker/redirect"
    write_json(paths["lock"], lock)
    with pytest.raises(ValueError, match="locked repo_id"):
        FINALIZE.build_receipt(
            sources_path=paths["config"],
            lock_path=paths["lock"],
            download_manifest_path=tmp_path / "sources.download.json",
            schema_audit_path=tmp_path / "sources.schemas.json",
            destination=tmp_path / "staged",
            code_commit="a" * 40,
        )


def fake_aarch64_elf(path: Path) -> None:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[18:20] = (183).to_bytes(2, "little")
    path.write_bytes(header + b"fixture executable")


def test_detector_build_receipt_binds_commit_cargo_and_aarch64_binary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    detector = repo / "detector"
    detector.mkdir(parents=True)
    binary = tmp_path / "published" / "reference_detect"
    binary.parent.mkdir()
    fake_aarch64_elf(binary)
    cargo_lock = detector / "Cargo.lock"
    cargo_toml = detector / "Cargo.toml"
    cargo_lock.write_text("version = 4\n", encoding="utf-8")
    cargo_toml.write_text("[package]\nname='fixture'\n", encoding="utf-8")
    commit = "b" * 40
    receipt = BUILD.build_receipt(
        repo=repo,
        binary=binary,
        cargo_lock=cargo_lock,
        cargo_toml=cargo_toml,
        code_commit=commit,
        architecture="aarch64",
        cargo_version="cargo fixture",
        rustc_version="rustc fixture verbose",
    )
    receipt_path = tmp_path / "build_receipt.json"
    write_json(receipt_path, receipt)

    check = VALIDATE_BUILD.validate_receipt(
        receipt_path=receipt_path,
        repo=repo,
        binary=binary,
        cargo_lock=cargo_lock,
        cargo_toml=cargo_toml,
        current_commit=commit,
        current_architecture="aarch64",
    )
    assert check["ok"] is True
    with pytest.raises(ValueError, match="execution commit|current checkout commit"):
        VALIDATE_BUILD.validate_receipt(
            receipt_path=receipt_path,
            repo=repo,
            binary=binary,
            cargo_lock=cargo_lock,
            cargo_toml=cargo_toml,
            current_commit="c" * 40,
            current_architecture="aarch64",
        )
    binary.write_bytes(binary.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        VALIDATE_BUILD.validate_receipt(
            receipt_path=receipt_path,
            repo=repo,
            binary=binary,
            cargo_lock=cargo_lock,
            cargo_toml=cargo_toml,
            current_commit=commit,
            current_architecture="aarch64",
        )


def test_sbatch_spool_contract_uses_exported_repo_directory_and_commit() -> None:
    sbatches = sorted((HERE / "clariden").glob("*.sbatch"))
    assert sbatches
    for script in sbatches:
        body = script.read_text(encoding="utf-8")
        assert "HERE=${PHASE04_CLARIDEN_DIR:?" in body
        assert "dirname \"${BASH_SOURCE[0]}\"" not in body

    submit = (HERE / "clariden" / "submit.sh").read_text(encoding="utf-8")
    for exported in (
        "REPO_ROOT=$REPO_ROOT",
        "PHASE04_DIR=$PHASE04_DIR",
        "PHASE04_CLARIDEN_DIR=$HERE",
        "PHASE04_EXPECTED_COMMIT=$PHASE04_EXPECTED_COMMIT",
    ):
        assert exported in submit

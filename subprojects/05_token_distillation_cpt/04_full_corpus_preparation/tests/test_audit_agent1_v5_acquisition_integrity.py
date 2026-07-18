from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "audit_agent1_v5_acquisition_integrity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_agent1_v5_acquisition_integrity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_row(root: Path, name: str, payload: bytes) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    stat = path.stat()
    return {
        "path": name,
        "local_path": str(path),
        "hash_kind": "sha256",
        "expected_hash": hashlib.sha256(payload).hexdigest(),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def test_resolve_under_local_root_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.bin"
    outside = tmp_path / "outside.bin"
    inside.write_bytes(b"inside")
    outside.write_bytes(b"outside")

    resolved_root, resolved_path = AUDIT.resolve_under_local_root(root, inside)
    assert resolved_root == root.resolve()
    assert resolved_path == inside.resolve()
    with pytest.raises(ValueError, match="escapes"):
        AUDIT.resolve_under_local_root(root, outside)


def test_file_audit_detects_sha_and_stat_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "acquired"
    row = file_row(root, "part.parquet", b"current bytes")
    row["expected_hash"] = hashlib.sha256(b"different bytes").hexdigest()
    row["mtime_ns"] = int(row["mtime_ns"]) + 1

    result = AUDIT.audit_acquired_file("candidate", root, row)

    assert result["verified"] is False
    assert result["hashed_bytes"] == len(b"current bytes")
    assert result["counters"]["hash_mismatches"] == 1
    assert result["counters"]["mtime_ns_mismatches"] == 1
    assert {item["code"] for item in result["issues"]} == {
        "content_sha256_mismatch",
        "stat_identity_mismatch",
    }


def test_task_closure_requires_acquired_path_hash_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "part.parquet"
    path.write_bytes(b"bytes")
    path_key = str(path.resolve())
    acquired = {
        path_key: {
            "source_id": "candidate",
            "artifact_path": "part.parquet",
            "artifact_path_raw": "part.parquet",
            "expected_hash": "a" * 64,
            "hash_kind": "sha256",
            "repo_id": "owner/repo",
            "revision": "b" * 40,
        }
    }
    tasks = {
        "task_count": 1,
        "tasks": [
            {
                "task_index": 0,
                "source_id": "candidate",
                "artifact_path": "part.parquet",
                "input_path": str(path),
                "input_expected_hash": "c" * 64,
                "input_hash_kind": "sha256",
                "repo_id": "owner/repo",
                "revision": "b" * 40,
            }
        ],
    }

    result = AUDIT.audit_task_closure(
        tasks,
        task_kind="transform_tasks",
        acquired_by_path=acquired,
    )

    assert result["matched_tasks"] == 1
    assert result["unique_acquired_inputs"] == 1
    assert result["counters"]["task_input_hash_mismatches"] == 1
    assert result["counters"]["task_identity_mismatches"] == 0


def test_run_contract_binding_detects_acquisition_receipt_size_and_sha_drift(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    acquisition_path = tmp_path / "acquisition.json"
    write_json(
        acquisition_path,
        {
            "schema_version": AUDIT.ACQUISITION_SCHEMA,
            "status": "passed",
            "sources": [],
        },
    )
    contract_path = run_root / "run_contract.json"
    write_json(
        contract_path,
        {
            "schema_version": AUDIT.RUN_CONTRACT_SCHEMA,
            "status": "passed",
            "run_id": "binding-drift-test",
            "run_root": str(run_root),
            "source_ids": [],
            "acquisition_receipt": {
                "path": str(acquisition_path),
                "bytes": acquisition_path.stat().st_size,
                "sha256": AUDIT.sha256_file(acquisition_path),
            },
        },
    )
    acquisition_path.write_bytes(acquisition_path.read_bytes() + b" \n")

    receipt = AUDIT.audit_run(
        contract_path=contract_path,
        transform_tasks_path=run_root / "transform_tasks.json",
        transform_manifest_path=run_root / "transform_manifest.json",
        base_tasks_path=run_root / "base_tasks.json",
        base_manifest_path=run_root / "base_manifest.json",
        workers=1,
        expected_files=1,
    )

    assert receipt["status"] == "blocked"
    assert receipt["mismatch_counters"]["acquisition_receipt_size_mismatches"] == 1
    assert receipt["mismatch_counters"]["acquisition_receipt_sha256_mismatches"] == 1
    assert receipt["inputs"]["acquisition_receipt"]["binding_valid"] is False


def test_small_end_to_end_audit_passes_but_never_claims_pretraining_ready(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    candidate_root = tmp_path / "acquired" / "candidate"
    base_root = tmp_path / "acquired" / "nanochat"
    candidate = file_row(candidate_root, "candidate.parquet", b"candidate")
    base = file_row(base_root, "base.parquet", b"base")
    candidate_revision = "1" * 40
    base_revision = "2" * 40
    acquisition = {
        "schema_version": AUDIT.ACQUISITION_SCHEMA,
        "status": "passed",
        "sources": [
            {
                "source_id": "candidate",
                "repo_id": "owner/candidate",
                "revision": candidate_revision,
                "local_root": str(candidate_root),
                "selected_file_count": 1,
                "selected_bytes": candidate["size"],
                "files": [candidate],
            },
            {
                "source_id": "nanochat_base",
                "repo_id": "owner/base",
                "revision": base_revision,
                "local_root": str(base_root),
                "selected_file_count": 1,
                "selected_bytes": base["size"],
                "files": [base],
            },
        ],
        "staging": {
            "files": 2,
            "bytes": int(candidate["size"]) + int(base["size"]),
        },
    }
    acquisition_path = tmp_path / "acquisition.json"
    write_json(acquisition_path, acquisition)
    contract_path = run_root / "run_contract.json"
    contract = {
        "schema_version": AUDIT.RUN_CONTRACT_SCHEMA,
        "status": "passed",
        "run_id": "test-run",
        "run_root": str(run_root),
        "source_ids": ["candidate"],
        "acquisition_receipt": {
            "path": str(acquisition_path),
            "bytes": acquisition_path.stat().st_size,
            "sha256": AUDIT.sha256_file(acquisition_path),
        },
    }
    write_json(contract_path, contract)
    contract_sha = AUDIT.sha256_file(contract_path)

    transform_tasks_path = run_root / "transform_tasks.json"
    write_json(
        transform_tasks_path,
        {
            "schema_version": AUDIT.TRANSFORM_TASKS_SCHEMA,
            "status": "passed",
            "run_contract_sha256": contract_sha,
            "task_count": 1,
            "tasks": [
                {
                    "task_index": 0,
                    "source_id": "candidate",
                    "repo_id": "owner/candidate",
                    "revision": candidate_revision,
                    "artifact_path": candidate["path"],
                    "input_path": candidate["local_path"],
                    "input_expected_hash": candidate["expected_hash"],
                    "input_hash_kind": candidate["hash_kind"],
                }
            ],
        },
    )
    base_tasks_path = run_root / "base_tasks.json"
    write_json(
        base_tasks_path,
        {
            "schema_version": AUDIT.BASE_TASKS_SCHEMA,
            "status": "passed",
            "run_contract_sha256": contract_sha,
            "task_count": 1,
            "tasks": [
                {
                    "task_index": 0,
                    "source_id": "nanochat_base",
                    "repo_id": "owner/base",
                    "revision": base_revision,
                    "artifact_path": base["path"],
                    "input_path": base["local_path"],
                    "input_expected_hash": base["expected_hash"],
                    "input_hash_kind": base["hash_kind"],
                }
            ],
        },
    )
    transform_manifest_path = run_root / "transform_manifest.json"
    write_json(
        transform_manifest_path,
        {
            "schema_version": AUDIT.TRANSFORM_MANIFEST_SCHEMA,
            "status": "passed",
            "run_contract_sha256": contract_sha,
            "task_manifest_sha256": AUDIT.sha256_file(transform_tasks_path),
            "task_count": 1,
        },
    )
    base_manifest_path = run_root / "base_manifest.json"
    write_json(
        base_manifest_path,
        {
            "schema_version": AUDIT.BASE_MANIFEST_SCHEMA,
            "status": "passed",
            "run_contract_sha256": contract_sha,
            "base_plan_sha256": AUDIT.sha256_file(base_tasks_path),
            "task_count": 1,
        },
    )

    receipt = AUDIT.audit_run(
        contract_path=contract_path,
        transform_tasks_path=transform_tasks_path,
        transform_manifest_path=transform_manifest_path,
        base_tasks_path=base_tasks_path,
        base_manifest_path=base_manifest_path,
        workers=1,
        expected_files=2,
    )

    assert receipt["status"] == "passed"
    assert receipt["pretraining_ready"] is False
    assert receipt["counts"] == {
        "sources": 2,
        "acquired_files": 2,
        "receipt_bytes": len(b"candidate") + len(b"base"),
        "hashed_files": 2,
        "hashed_bytes": len(b"candidate") + len(b"base"),
        "verified_files": 2,
        "blocking_issues": 0,
    }
    assert not any(receipt["mismatch_counters"].values())
    AUDIT.assert_sanitized(receipt)

    output = run_root / "acquisition-integrity.json"
    AUDIT.write_json_immutable(output, receipt)
    with pytest.raises(FileExistsError, match="immutable"):
        AUDIT.write_json_immutable(output, receipt)

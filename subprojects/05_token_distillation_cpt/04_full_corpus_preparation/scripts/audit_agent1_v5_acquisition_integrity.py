#!/usr/bin/env python3
"""Prove Agent-1 v5 acquisition bytes and task-input lineage are intact.

The audit is deliberately read-only.  It validates the acquisition receipt
bound by a passed v5 run contract, hashes every acquired artifact, checks the
receipt's filesystem identity bindings, and closes the transform/base task
plans back to those artifacts.  The output contains hashes and aggregate
metadata only: it never includes corpus content or absolute filesystem paths.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "agent1_v5_acquisition_integrity_audit_v1"
RUN_CONTRACT_SCHEMA = "agent1_v5_run_contract_v1"
ACQUISITION_SCHEMA = "full_cpt_acquisition_receipt_v1"
TRANSFORM_TASKS_SCHEMA = "agent1_v5_transform_task_manifest_v1"
TRANSFORM_MANIFEST_SCHEMA = "agent1_v5_transform_manifest_v1"
BASE_TASKS_SCHEMA = "agent1_v5_base_plan_v1"
BASE_MANIFEST_SCHEMA = "agent1_v5_base_manifest_v1"
EXPECTED_ACQUIRED_FILES = 293
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STAT_FIELDS = ("device", "inode", "mtime_ns", "ctime_ns")
MISMATCH_COUNTERS = (
    "acquisition_receipt_size_mismatches",
    "acquisition_receipt_sha256_mismatches",
    "acquisition_schema_mismatches",
    "acquisition_status_mismatches",
    "source_roster_mismatches",
    "expected_file_count_mismatches",
    "source_count_mismatches",
    "source_byte_count_mismatches",
    "acquisition_count_mismatches",
    "acquisition_byte_count_mismatches",
    "unsafe_artifact_paths",
    "missing_files",
    "symlink_files",
    "path_containment_mismatches",
    "artifact_path_mapping_mismatches",
    "invalid_size_bindings",
    "size_mismatches",
    "device_mismatches",
    "inode_mismatches",
    "mtime_ns_mismatches",
    "ctime_ns_mismatches",
    "invalid_hash_bindings",
    "hash_mismatches",
    "files_changed_during_audit",
    "duplicate_acquired_paths",
    "run_root_mismatches",
    "run_input_missing",
    "run_input_path_mismatches",
    "run_input_schema_mismatches",
    "run_input_status_mismatches",
    "run_contract_link_mismatches",
    "task_manifest_link_mismatches",
    "task_count_mismatches",
    "task_index_mismatches",
    "task_input_missing_from_acquisition",
    "task_input_hash_mismatches",
    "task_identity_mismatches",
    "task_source_partition_mismatches",
    "task_non_parquet_inputs",
    "duplicate_base_task_inputs",
    "acquired_parquet_missing_from_tasks",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def sha256_file(path: Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: object) -> str:
    text = str(value)
    if SAFE_LABEL_RE.fullmatch(text):
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def artifact_ref(value: object) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if text and not pure.is_absolute() and ".." not in pure.parts:
        return pure.as_posix()
    return safe_label(text)


def issue(code: str, **context: object) -> dict[str, object]:
    result: dict[str, object] = {"code": code}
    for key, value in context.items():
        if value is None:
            continue
        if key in {"source_id", "observed_schema", "observed_status"}:
            result[key] = safe_label(value)
        elif key == "artifact_path":
            result[key] = artifact_ref(value)
        elif key in {"task_index", "expected", "observed"} and isinstance(
            value, (int, float)
        ):
            result[key] = value
        elif key in {"task_kind", "field"}:
            result[key] = safe_label(value)
        else:
            result[key] = safe_label(value)
    return result


def stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size": int(value.st_size),
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def stat_identity_mismatches(
    row: Mapping[str, Any], observed: Mapping[str, int]
) -> list[str]:
    mismatches: list[str] = []
    for name in ("size", *STAT_FIELDS):
        try:
            expected = int(row[name])
        except (KeyError, TypeError, ValueError):
            mismatches.append(name)
            continue
        if expected != int(observed[name]):
            mismatches.append(name)
    return mismatches


def resolve_under_local_root(
    local_root_value: object, local_path_value: object
) -> tuple[Path, Path]:
    """Resolve an acquired path and prove it remains under its declared root."""

    root = Path(str(local_root_value)).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local_root_not_directory")
    raw = Path(str(local_path_value))
    candidate = raw if raw.is_absolute() else root / raw
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("path_escapes_local_root")
    return root, path


def _valid_artifact_path(value: object) -> bool:
    text = str(value)
    pure = PurePosixPath(text)
    return bool(text) and not pure.is_absolute() and ".." not in pure.parts


def audit_acquired_file(
    source_id: str, local_root: object, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash one artifact and return metadata-only evidence and mismatch counters."""

    counters: Counter[str] = Counter()
    issues: list[dict[str, object]] = []
    artifact = artifact_ref(row.get("path", ""))
    context = {"source_id": source_id, "artifact_path": artifact}
    if not _valid_artifact_path(row.get("path", "")):
        counters["unsafe_artifact_paths"] += 1
        issues.append(issue("unsafe_artifact_path", **context))

    result: dict[str, Any] = {
        "source_id": source_id,
        "artifact_path": artifact,
        "expected_hash": str(row.get("expected_hash", "")),
        "hash_kind": str(row.get("hash_kind", "")),
        "repo_id": row.get("repo_id"),
        "revision": row.get("revision"),
        "_path_key": None,
        "receipt_bytes": 0,
        "hashed_bytes": 0,
        "hashed": False,
        "verified": False,
        "counters": counters,
        "issues": issues,
    }
    try:
        expected_size = int(row["size"])
        if expected_size < 0:
            raise ValueError
        result["receipt_bytes"] = expected_size
    except (KeyError, TypeError, ValueError):
        counters["invalid_size_bindings"] += 1
        issues.append(issue("invalid_size_binding", **context))

    try:
        root, path = resolve_under_local_root(local_root, row.get("local_path", ""))
    except FileNotFoundError:
        counters["missing_files"] += 1
        issues.append(issue("acquired_file_missing", **context))
        return result
    except (OSError, RuntimeError, ValueError):
        counters["path_containment_mismatches"] += 1
        issues.append(issue("local_path_not_contained", **context))
        return result

    result["_path_key"] = str(path)
    raw_local_path = Path(str(row.get("local_path", "")))
    declared_root = Path(str(local_root))
    declared_path = (
        raw_local_path
        if raw_local_path.is_absolute()
        else declared_root / raw_local_path
    )
    if declared_path.is_symlink():
        counters["symlink_files"] += 1
        issues.append(issue("acquired_file_is_symlink", **context))
    if _valid_artifact_path(row.get("path", "")):
        expected_path = (root / Path(*PurePosixPath(str(row["path"])).parts)).resolve(
            strict=False
        )
        if path != expected_path:
            counters["artifact_path_mapping_mismatches"] += 1
            issues.append(issue("artifact_path_mapping_mismatch", **context))

    before = stat_identity(path)
    for field in stat_identity_mismatches(row, before):
        counter = "size_mismatches" if field == "size" else f"{field}_mismatches"
        counters[counter] += 1
        issues.append(issue("stat_identity_mismatch", field=field, **context))

    expected_hash = str(row.get("expected_hash", ""))
    hash_kind = str(row.get("hash_kind", ""))
    valid_hash = hash_kind in {"sha256", "lfs_sha256"} and bool(
        SHA256_RE.fullmatch(expected_hash)
    )
    if not valid_hash:
        counters["invalid_hash_bindings"] += 1
        issues.append(issue("invalid_hash_binding", **context))
    try:
        observed_hash = sha256_file(path)
        after = stat_identity(path)
        result["hashed_bytes"] = after["size"]
        result["hashed"] = True
    except (FileNotFoundError, OSError):
        counters["files_changed_during_audit"] += 1
        issues.append(issue("file_changed_during_audit", **context))
        return result
    if before != after:
        counters["files_changed_during_audit"] += 1
        issues.append(issue("file_changed_during_audit", **context))
    if valid_hash and observed_hash != expected_hash:
        counters["hash_mismatches"] += 1
        issues.append(issue("content_sha256_mismatch", **context))
    result["verified"] = not counters
    return result


def _file_metadata(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _resolve_run_input(path: Path, run_root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or not resolved.is_relative_to(run_root)
    ):
        raise ValueError("invalid_run_input")
    return resolved


def _load_run_input(
    *,
    name: str,
    path: Path,
    run_root: Path,
    expected_schema: str,
    counters: Counter[str],
    issues: list[dict[str, object]],
    inputs: dict[str, object],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resolved = _resolve_run_input(path, run_root)
    except (FileNotFoundError, OSError, ValueError):
        counters["run_input_missing"] += 1
        issues.append(issue("run_input_missing_or_outside_run_root", task_kind=name))
        inputs[name] = {"available": False}
        return None, None
    metadata = _file_metadata(resolved)
    inputs[name] = {"available": True, **metadata}
    try:
        value = read_object(resolved)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        counters["run_input_schema_mismatches"] += 1
        issues.append(issue("run_input_not_json_object", task_kind=name))
        return None, str(metadata["sha256"])
    if value.get("schema_version") != expected_schema:
        counters["run_input_schema_mismatches"] += 1
        issues.append(
            issue(
                "run_input_schema_mismatch",
                task_kind=name,
                observed_schema=value.get("schema_version", "missing"),
            )
        )
    if value.get("status") != "passed":
        counters["run_input_status_mismatches"] += 1
        issues.append(
            issue(
                "run_input_status_mismatch",
                task_kind=name,
                observed_status=value.get("status", "missing"),
            )
        )
    return value, str(metadata["sha256"])


def _resolve_task_path(value: object) -> str:
    return str(Path(str(value)).resolve(strict=False))


def audit_task_closure(
    task_manifest: Mapping[str, Any],
    *,
    task_kind: str,
    acquired_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Close every task input to one acquisition receipt file binding."""

    counters: Counter[str] = Counter()
    issues: list[dict[str, object]] = []
    tasks_value = task_manifest.get("tasks")
    tasks: Sequence[Any] = tasks_value if isinstance(tasks_value, list) else []
    try:
        declared_count = int(task_manifest.get("task_count", -1))
    except (TypeError, ValueError):
        declared_count = -1
    if declared_count != len(tasks):
        counters["task_count_mismatches"] += 1
        issues.append(
            issue(
                "task_count_mismatch",
                task_kind=task_kind,
                expected=len(tasks),
                observed=declared_count,
            )
        )

    indexes: list[int] = []
    covered: set[str] = set()
    path_use_counts: Counter[str] = Counter()
    matched_tasks = 0
    for position, task_value in enumerate(tasks):
        if not isinstance(task_value, Mapping):
            counters["task_identity_mismatches"] += 1
            issues.append(
                issue("task_not_object", task_kind=task_kind, task_index=position)
            )
            continue
        task = task_value
        try:
            task_index = int(task.get("task_index", -1))
        except (TypeError, ValueError):
            task_index = -1
        indexes.append(task_index)
        task_source = str(task.get("source_id", ""))
        task_artifact = artifact_ref(task.get("artifact_path", ""))
        context = {
            "task_kind": task_kind,
            "task_index": task_index,
            "source_id": task_source,
            "artifact_path": task_artifact,
        }
        path_key = _resolve_task_path(task.get("input_path", ""))
        acquired = acquired_by_path.get(path_key)
        if acquired is None:
            counters["task_input_missing_from_acquisition"] += 1
            issues.append(issue("task_input_not_in_acquisition", **context))
            continue
        matched_tasks += 1
        covered.add(path_key)
        path_use_counts[path_key] += 1
        if str(task.get("input_expected_hash", "")) != str(
            acquired.get("expected_hash", "")
        ):
            counters["task_input_hash_mismatches"] += 1
            issues.append(issue("task_expected_hash_mismatch", **context))
        identity_fields = (
            ("source_id", task_source, str(acquired.get("source_id", ""))),
            (
                "artifact_path",
                str(task.get("artifact_path", "")),
                str(acquired.get("artifact_path_raw", "")),
            ),
            ("repo_id", str(task.get("repo_id", "")), str(acquired.get("repo_id", ""))),
            (
                "revision",
                str(task.get("revision", "")),
                str(acquired.get("revision", "")),
            ),
            (
                "input_hash_kind",
                str(task.get("input_hash_kind", "")),
                str(acquired.get("hash_kind", "")),
            ),
        )
        for field, observed, expected in identity_fields:
            if observed != expected:
                counters["task_identity_mismatches"] += 1
                issues.append(
                    issue("task_acquisition_identity_mismatch", field=field, **context)
                )
        is_base = str(acquired.get("source_id", "")) == "nanochat_base"
        if (task_kind == "base_tasks") != is_base:
            counters["task_source_partition_mismatches"] += 1
            issues.append(issue("task_source_partition_mismatch", **context))
        if (
            not str(acquired.get("artifact_path_raw", ""))
            .casefold()
            .endswith(".parquet")
        ):
            counters["task_non_parquet_inputs"] += 1
            issues.append(issue("task_input_not_parquet", **context))

    if indexes != list(range(len(tasks))):
        counters["task_index_mismatches"] += 1
        issues.append(issue("task_index_sequence_mismatch", task_kind=task_kind))
    if task_kind == "base_tasks":
        duplicates = sum(count - 1 for count in path_use_counts.values() if count > 1)
        if duplicates:
            counters["duplicate_base_task_inputs"] += duplicates
            issues.append(
                issue(
                    "duplicate_base_task_inputs",
                    task_kind=task_kind,
                    observed=duplicates,
                )
            )
    return {
        "task_count": len(tasks),
        "matched_tasks": matched_tasks,
        "unique_acquired_inputs": len(covered),
        "_covered": covered,
        "counters": counters,
        "issues": issues,
    }


def _merge_counters(target: Counter[str], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] += int(value)


def _check_link(
    *,
    value: Mapping[str, Any] | None,
    field: str,
    expected: str | None,
    code_counter: str,
    task_kind: str,
    counters: Counter[str],
    issues: list[dict[str, object]],
) -> None:
    if value is not None and (expected is None or value.get(field) != expected):
        counters[code_counter] += 1
        issues.append(
            issue("receipt_sha256_link_mismatch", task_kind=task_kind, field=field)
        )


def audit_run(
    *,
    contract_path: Path,
    transform_tasks_path: Path,
    transform_manifest_path: Path,
    base_tasks_path: Path,
    base_manifest_path: Path,
    workers: int = 1,
    expected_files: int = EXPECTED_ACQUIRED_FILES,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    contract_path = contract_path.resolve(strict=True)
    contract = read_object(contract_path)
    if (
        contract.get("schema_version") != RUN_CONTRACT_SCHEMA
        or contract.get("status") != "passed"
    ):
        raise ValueError("run contract is not a passed agent1_v5_run_contract_v1")
    run_root = contract_path.parent.resolve(strict=True)
    contract_sha256 = sha256_file(contract_path)
    counters: Counter[str] = Counter({key: 0 for key in MISMATCH_COUNTERS})
    issues: list[dict[str, object]] = []
    inputs: dict[str, object] = {
        "run_contract": {
            "bytes": contract_path.stat().st_size,
            "sha256": contract_sha256,
        }
    }
    try:
        declared_run_root = Path(str(contract.get("run_root", ""))).resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        declared_run_root = Path("/")
    if declared_run_root != run_root:
        counters["run_root_mismatches"] += 1
        issues.append(issue("contract_run_root_mismatch"))

    acquisition_binding = contract.get("acquisition_receipt")
    acquisition: dict[str, Any] | None = None
    acquisition_path: Path | None = None
    if not isinstance(acquisition_binding, Mapping):
        counters["acquisition_schema_mismatches"] += 1
        issues.append(issue("contract_acquisition_binding_missing"))
        inputs["acquisition_receipt"] = {"available": False, "binding_valid": False}
    else:
        raw_path = Path(str(acquisition_binding.get("path", "")))
        acquisition_path = (
            run_root / raw_path if not raw_path.is_absolute() else raw_path
        ).resolve(strict=False)
        if not acquisition_path.is_file() or acquisition_path.is_symlink():
            counters["run_input_missing"] += 1
            issues.append(issue("acquisition_receipt_missing"))
            inputs["acquisition_receipt"] = {"available": False, "binding_valid": False}
        else:
            actual = _file_metadata(acquisition_path)
            size_valid = False
            try:
                size_valid = int(acquisition_binding.get("bytes", -1)) == int(
                    actual["bytes"]
                )
            except (TypeError, ValueError):
                pass
            hash_valid = acquisition_binding.get("sha256") == actual["sha256"]
            if not size_valid:
                counters["acquisition_receipt_size_mismatches"] += 1
                issues.append(issue("acquisition_receipt_size_mismatch"))
            if not hash_valid:
                counters["acquisition_receipt_sha256_mismatches"] += 1
                issues.append(issue("acquisition_receipt_sha256_mismatch"))
            inputs["acquisition_receipt"] = {
                "available": True,
                **actual,
                "bound_bytes": acquisition_binding.get("bytes"),
                "bound_sha256": acquisition_binding.get("sha256"),
                "binding_valid": size_valid and hash_valid,
            }
            try:
                acquisition = read_object(acquisition_path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                counters["acquisition_schema_mismatches"] += 1
                issues.append(issue("acquisition_receipt_not_json_object"))

    file_results: list[dict[str, Any]] = []
    acquired_by_path: dict[str, dict[str, Any]] = {}
    per_source: dict[str, dict[str, int]] = {}
    if acquisition is not None:
        if acquisition.get("schema_version") != ACQUISITION_SCHEMA:
            counters["acquisition_schema_mismatches"] += 1
            issues.append(
                issue(
                    "acquisition_schema_mismatch",
                    observed_schema=acquisition.get("schema_version", "missing"),
                )
            )
        if acquisition.get("status") != "passed":
            counters["acquisition_status_mismatches"] += 1
            issues.append(
                issue(
                    "acquisition_status_mismatch",
                    observed_status=acquisition.get("status", "missing"),
                )
            )
        sources_value = acquisition.get("sources")
        sources: Sequence[Any] = (
            sources_value if isinstance(sources_value, list) else []
        )
        expected_sources = [str(value) for value in contract.get("source_ids", [])] + [
            "nanochat_base"
        ]
        observed_sources = [
            str(value.get("source_id", ""))
            for value in sources
            if isinstance(value, Mapping)
        ]
        if len(observed_sources) != len(set(observed_sources)) or set(
            observed_sources
        ) != set(expected_sources):
            counters["source_roster_mismatches"] += 1
            issues.append(issue("acquisition_source_roster_mismatch"))

        jobs: list[tuple[str, object, Mapping[str, Any], Mapping[str, Any]]] = []
        for source_value in sources:
            if not isinstance(source_value, Mapping):
                counters["source_roster_mismatches"] += 1
                issues.append(issue("acquisition_source_not_object"))
                continue
            source_id = str(source_value.get("source_id", ""))
            files_value = source_value.get("files")
            files: Sequence[Any] = files_value if isinstance(files_value, list) else []
            if not isinstance(files_value, list):
                counters["acquisition_count_mismatches"] += 1
                issues.append(
                    issue("acquisition_source_files_not_list", source_id=source_id)
                )
            valid_rows = [row for row in files if isinstance(row, Mapping)]
            if len(valid_rows) != len(files):
                counters["acquisition_count_mismatches"] += len(files) - len(valid_rows)
                issues.append(
                    issue("acquisition_file_row_not_object", source_id=source_id)
                )
            expected_source_bytes = 0
            for row in valid_rows:
                try:
                    size = int(row.get("size", 0))
                    if size >= 0:
                        expected_source_bytes += size
                except (TypeError, ValueError):
                    pass
                enriched = dict(row)
                enriched["repo_id"] = source_value.get("repo_id")
                enriched["revision"] = source_value.get("revision")
                jobs.append(
                    (
                        source_id,
                        source_value.get("local_root", ""),
                        enriched,
                        source_value,
                    )
                )
            try:
                declared_source_count = int(source_value.get("selected_file_count", -1))
            except (TypeError, ValueError):
                declared_source_count = -1
            if declared_source_count != len(valid_rows):
                counters["source_count_mismatches"] += 1
                issues.append(issue("source_file_count_mismatch", source_id=source_id))
            try:
                declared_source_bytes = int(source_value.get("selected_bytes", -1))
            except (TypeError, ValueError):
                declared_source_bytes = -1
            if declared_source_bytes != expected_source_bytes:
                counters["source_byte_count_mismatches"] += 1
                issues.append(issue("source_byte_count_mismatch", source_id=source_id))

        if workers == 1:
            file_results = [
                audit_acquired_file(source_id, root, row)
                for source_id, root, row, _ in jobs
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                file_results = list(
                    pool.map(
                        lambda job: audit_acquired_file(job[0], job[1], job[2]),
                        jobs,
                    )
                )

        for result, (_, _, row, source) in zip(file_results, jobs):
            _merge_counters(counters, result["counters"])
            issues.extend(result["issues"])
            path_key = result["_path_key"]
            if path_key is not None:
                if path_key in acquired_by_path:
                    counters["duplicate_acquired_paths"] += 1
                    issues.append(
                        issue(
                            "duplicate_acquired_path",
                            source_id=result["source_id"],
                            artifact_path=result["artifact_path"],
                        )
                    )
                else:
                    acquired_by_path[path_key] = {
                        "source_id": result["source_id"],
                        "artifact_path": result["artifact_path"],
                        "artifact_path_raw": str(row.get("path", "")),
                        "expected_hash": result["expected_hash"],
                        "hash_kind": result["hash_kind"],
                        "repo_id": source.get("repo_id"),
                        "revision": source.get("revision"),
                    }

        if len(file_results) != expected_files:
            counters["expected_file_count_mismatches"] += 1
            issues.append(
                issue(
                    "expected_acquired_file_count_mismatch",
                    expected=expected_files,
                    observed=len(file_results),
                )
            )
        receipt_bytes = sum(int(value["receipt_bytes"]) for value in file_results)
        staging = acquisition.get("staging")
        if isinstance(staging, Mapping):
            try:
                if int(staging.get("files", -1)) != len(file_results):
                    counters["acquisition_count_mismatches"] += 1
                    issues.append(issue("acquisition_staging_file_count_mismatch"))
            except (TypeError, ValueError):
                counters["acquisition_count_mismatches"] += 1
                issues.append(issue("acquisition_staging_file_count_mismatch"))
            try:
                if int(staging.get("bytes", -1)) != receipt_bytes:
                    counters["acquisition_byte_count_mismatches"] += 1
                    issues.append(issue("acquisition_staging_byte_count_mismatch"))
            except (TypeError, ValueError):
                counters["acquisition_byte_count_mismatches"] += 1
                issues.append(issue("acquisition_staging_byte_count_mismatch"))

        source_stats: dict[str, Counter[str]] = defaultdict(Counter)
        for source_id in observed_sources:
            source_stats[safe_label(source_id)]
        for result in file_results:
            key = safe_label(result["source_id"])
            stats = source_stats[key]
            stats["files"] += 1
            stats["receipt_bytes"] += int(result["receipt_bytes"])
            stats["hashed_bytes"] += int(result["hashed_bytes"])
            stats["hashed_files"] += int(result["hashed"])
            stats["verified_files"] += int(result["verified"])
            stats["hash_mismatches"] += int(
                result["counters"].get("hash_mismatches", 0)
            )
            stats["stat_mismatches"] += sum(
                int(result["counters"].get(f"{field}_mismatches", 0))
                for field in ("size", *STAT_FIELDS)
            )
            stats["path_mismatches"] += int(
                result["counters"].get("path_containment_mismatches", 0)
            )
        per_source_fields = (
            "files",
            "receipt_bytes",
            "hashed_files",
            "hashed_bytes",
            "verified_files",
            "hash_mismatches",
            "stat_mismatches",
            "path_mismatches",
        )
        per_source = {
            key: {field: int(value[field]) for field in per_source_fields}
            for key, value in sorted(source_stats.items())
        }

    transform_tasks, transform_tasks_sha = _load_run_input(
        name="transform_tasks",
        path=transform_tasks_path,
        run_root=run_root,
        expected_schema=TRANSFORM_TASKS_SCHEMA,
        counters=counters,
        issues=issues,
        inputs=inputs,
    )
    transform_manifest, transform_manifest_sha = _load_run_input(
        name="transform_manifest",
        path=transform_manifest_path,
        run_root=run_root,
        expected_schema=TRANSFORM_MANIFEST_SCHEMA,
        counters=counters,
        issues=issues,
        inputs=inputs,
    )
    base_tasks, base_tasks_sha = _load_run_input(
        name="base_tasks",
        path=base_tasks_path,
        run_root=run_root,
        expected_schema=BASE_TASKS_SCHEMA,
        counters=counters,
        issues=issues,
        inputs=inputs,
    )
    base_manifest, base_manifest_sha = _load_run_input(
        name="base_manifest",
        path=base_manifest_path,
        run_root=run_root,
        expected_schema=BASE_MANIFEST_SCHEMA,
        counters=counters,
        issues=issues,
        inputs=inputs,
    )
    del transform_manifest_sha, base_manifest_sha

    for name, value in (
        ("transform_tasks", transform_tasks),
        ("transform_manifest", transform_manifest),
        ("base_tasks", base_tasks),
        ("base_manifest", base_manifest),
    ):
        _check_link(
            value=value,
            field="run_contract_sha256",
            expected=contract_sha256,
            code_counter="run_contract_link_mismatches",
            task_kind=name,
            counters=counters,
            issues=issues,
        )
    _check_link(
        value=transform_manifest,
        field="task_manifest_sha256",
        expected=transform_tasks_sha,
        code_counter="task_manifest_link_mismatches",
        task_kind="transform_manifest",
        counters=counters,
        issues=issues,
    )
    _check_link(
        value=base_manifest,
        field="base_plan_sha256",
        expected=base_tasks_sha,
        code_counter="task_manifest_link_mismatches",
        task_kind="base_manifest",
        counters=counters,
        issues=issues,
    )

    task_summary: dict[str, object] = {}
    covered_by_kind: dict[str, set[str]] = {
        "transform_tasks": set(),
        "base_tasks": set(),
    }
    for name, value in (
        ("transform_tasks", transform_tasks),
        ("base_tasks", base_tasks),
    ):
        if value is None:
            task_summary[name] = {"available": False}
            continue
        result = audit_task_closure(
            value, task_kind=name, acquired_by_path=acquired_by_path
        )
        _merge_counters(counters, result["counters"])
        issues.extend(result["issues"])
        covered_by_kind[name] = result["_covered"]
        task_summary[name] = {
            "available": True,
            "task_count": result["task_count"],
            "matched_tasks": result["matched_tasks"],
            "unique_acquired_inputs": result["unique_acquired_inputs"],
        }

    for path_key, acquired in acquired_by_path.items():
        artifact = str(acquired.get("artifact_path_raw", ""))
        if not artifact.casefold().endswith(".parquet"):
            continue
        kind = (
            "base_tasks"
            if acquired.get("source_id") == "nanochat_base"
            else "transform_tasks"
        )
        if path_key not in covered_by_kind[kind]:
            counters["acquired_parquet_missing_from_tasks"] += 1
            issues.append(
                issue(
                    "acquired_parquet_not_covered_by_tasks",
                    task_kind=kind,
                    source_id=acquired.get("source_id"),
                    artifact_path=artifact,
                )
            )

    for manifest_name, manifest_value, task_value in (
        ("transform_manifest", transform_manifest, transform_tasks),
        ("base_manifest", base_manifest, base_tasks),
    ):
        if manifest_value is None or task_value is None:
            continue
        try:
            manifest_count = int(manifest_value.get("task_count", -1))
            task_count = int(task_value.get("task_count", -2))
        except (TypeError, ValueError):
            manifest_count, task_count = -1, -2
        if manifest_count != task_count:
            counters["task_count_mismatches"] += 1
            issues.append(
                issue("merged_manifest_task_count_mismatch", task_kind=manifest_name)
            )

    receipt_bytes = sum(int(value["receipt_bytes"]) for value in file_results)
    hashed_bytes = sum(int(value["hashed_bytes"]) for value in file_results)
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "status": "blocked" if any(counters.values()) else "passed",
        "created_at": utc_now(),
        "pretraining_ready": False,
        "scope": "acquisition_integrity_and_task_input_lineage_only",
        "script": {
            "sha256": sha256_file(Path(__file__).resolve()),
            "hash_workers": workers,
        },
        "run_id": safe_label(contract.get("run_id", "missing")),
        "inputs": inputs,
        "expectations": {
            "acquired_files": expected_files,
            "source_ids": len(contract.get("source_ids", [])) + 1,
        },
        "counts": {
            "sources": len(per_source),
            "acquired_files": len(file_results),
            "receipt_bytes": receipt_bytes,
            "hashed_files": sum(int(value["hashed"]) for value in file_results),
            "hashed_bytes": hashed_bytes,
            "verified_files": sum(int(value["verified"]) for value in file_results),
            "blocking_issues": len(issues),
        },
        "per_source": per_source,
        "task_closure": task_summary,
        "mismatch_counters": {key: int(counters[key]) for key in MISMATCH_COUNTERS},
        "blocking_issues": issues,
    }
    assert_sanitized(receipt)
    return receipt


def assert_sanitized(value: object) -> None:
    """Fail closed if any output string is an absolute POSIX/Windows path."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_sanitized(key)
            assert_sanitized(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_sanitized(child)
    elif isinstance(value, str):
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("audit receipt contains an absolute path")


def write_json_immutable(path: Path, value: Mapping[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError("immutable audit receipt already exists")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--transform-tasks", type=Path)
    parser.add_argument("--transform-manifest", type=Path)
    parser.add_argument("--base-tasks", type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument(
        "--expected-files", type=positive_int, default=EXPECTED_ACQUIRED_FILES
    )
    args = parser.parse_args(argv)
    run_root = args.contract.resolve().parent
    receipt = audit_run(
        contract_path=args.contract,
        transform_tasks_path=args.transform_tasks or run_root / "transform_tasks.json",
        transform_manifest_path=args.transform_manifest
        or run_root / "transform_manifest.json",
        base_tasks_path=args.base_tasks or run_root / "base_tasks.json",
        base_manifest_path=args.base_manifest or run_root / "base_manifest.json",
        workers=args.workers,
        expected_files=args.expected_files,
    )
    write_json_immutable(args.output, receipt)
    print(
        canonical_json(
            {
                "ok": receipt["status"] == "passed",
                "status": receipt["status"],
                "files": receipt["counts"]["acquired_files"],
                "verified_files": receipt["counts"]["verified_files"],
                "blocking_issues": receipt["counts"]["blocking_issues"],
            }
        )
    )
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

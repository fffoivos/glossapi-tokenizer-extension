#!/usr/bin/env python3
"""Shared fail-closed helpers for the row-preserving HF v2 anonymization."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "glossapi_hf_v2_anonymized_release_config_v1"
CONTRACT_SCHEMA = "glossapi_hf_v2_anonymization_contract_v1"
TASK_SCHEMA = "glossapi_hf_v2_anonymized_shard_receipt_v1"
FINAL_SCHEMA = "glossapi_hf_v2_anonymized_release_manifest_v1"
PUBLICATION_SCHEMA = "glossapi_hf_v2_anonymized_publication_v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(dict(value)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_receipt(path: Path, *, rows: int | None = None, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    value: dict[str, Any] = {
        "path": path.relative_to(relative_to.resolve()).as_posix() if relative_to else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = int(rows)
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported release config")
    categories = config.get("source_categories")
    expected = config.get("expected_source_rows")
    if not isinstance(categories, list) or not isinstance(expected, dict):
        raise ValueError("source taxonomy is missing")
    mapping: dict[str, str] = {}
    category_ids: set[str] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("category must be an object")
        category_id = str(category.get("id", ""))
        if not category_id or category_id in category_ids:
            raise ValueError(f"invalid or duplicate category: {category_id!r}")
        category_ids.add(category_id)
        sources = category.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"category {category_id!r} has no sources")
        for source in sources:
            source = str(source)
            if source in mapping:
                raise ValueError(f"source appears in multiple categories: {source}")
            mapping[source] = category_id
    if set(mapping) != set(expected):
        raise ValueError(
            "taxonomy/source receipt mismatch: "
            f"missing={sorted(set(expected) - set(mapping))}, extra={sorted(set(mapping) - set(expected))}"
        )
    if sum(int(value) for value in expected.values()) != int(config["input"]["rows"]):
        raise ValueError("expected source rows do not close to the input release")
    if int(config["deduplication"]["retained_rows"]) != int(config["input"]["rows"]):
        raise ValueError("dedup retained rows differ from the release")
    return mapping


def code_inventory(code_root: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    code_root = code_root.resolve()
    result = []
    for relative in sorted(set(relative_paths)):
        path = code_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        result.append(file_receipt(path, relative_to=code_root))
    return result


def verify_code_inventory(code_root: Path, rows: list[dict[str, Any]]) -> None:
    code_root = code_root.resolve()
    for row in rows:
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe code receipt path: {relative}")
        path = code_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row.get("bytes", -1)) or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"executing code bundle drift: {relative}")


def load_contract(path: Path, *, executing_code_root: Path) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA or contract.get("status") != "frozen":
        raise ValueError("run contract is not frozen")
    code_root = Path(str(contract.get("code_root", ""))).resolve()
    if code_root != executing_code_root.resolve():
        raise ValueError("executing code root differs from the frozen contract")
    verify_code_inventory(code_root, list(contract.get("code_inventory", [])))
    config_path = code_root / str(contract["config"]["path"])
    if sha256_file(config_path) != contract["config"]["sha256"]:
        raise ValueError("release config drift")
    config = read_json(config_path)
    validate_config(config)
    if canonical_sha256(config) != contract["config"]["canonical_sha256"]:
        raise ValueError("release config canonical hash drift")
    return contract


def update_text_digest(digest: Any, text: str) -> None:
    payload = text.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


#!/usr/bin/env python3
"""Bind an audit launch to one immutable Phase-04 acquisition receipt.

Acquisition already performs the expensive LFS payload hashing.  This launch
gate deliberately validates only the small receipt/config plus the exact
resolved Parquet path set and current file sizes.  It therefore catches path,
revision, selection and obvious staging drift without re-reading the corpus at
the start of every audit.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
from pathlib import Path
from typing import Sequence


HEX = set("0123456789abcdef")


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expand_inputs(values: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.parquet")))
        elif any(character in value for character in "*?["):
            found.extend(Path(item) for item in sorted(glob.glob(value, recursive=True)))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"audit input did not resolve: {value}")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        canonical = path.resolve()
        if canonical.suffix.lower() != ".parquet":
            continue
        if canonical not in seen:
            seen.add(canonical)
            resolved.append(canonical)
    if not resolved:
        raise ValueError("no Parquet audit inputs resolved")
    return resolved


def route_for_source(config: dict, source_id: str) -> tuple[dict, str, list[str], list[str]]:
    for route in config.get("embedded_structural_routes", []):
        if route.get("source_id") == source_id:
            return (
                route,
                str(route["acquisition_source_id"]),
                list(route["acquisition_include_globs"]),
                [],
            )

    registry: dict[str, dict] = {
        "nanochat_base": config["base"],
        "apertus_overlap_overlay": config["apertus_overlap_overlay"],
        "modern_greek_148k_tokenizer": config["tokenizer"],
        **{str(row["source_id"]): row for row in config.get("sources", [])},
    }
    route = registry.get(source_id)
    if route is None:
        raise ValueError(f"{source_id!r} is not a tracked acquisition or embedded route")
    return route, source_id, list(route.get("include_globs", [])), list(route.get("exclude_globs", []))


def matches(path: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    included = bool(includes) and any(fnmatch.fnmatchcase(path, pattern) for pattern in includes)
    excluded = any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes)
    return included and not excluded


def valid_content_identifier(kind: object, value: object) -> bool:
    if not isinstance(value, str) or any(character not in HEX for character in value):
        return False
    if kind == "lfs_sha256":
        return len(value) == 64
    if kind == "git_blob_id":
        return len(value) in {40, 64}
    return False


def validate_input(
    *,
    receipt_path: Path,
    sources_path: Path,
    source_id: str,
    input_values: Sequence[str],
    text_column: str,
    id_column: str,
    source_column: str,
) -> dict:
    receipt = load_object(receipt_path)
    config = load_object(sources_path)
    errors: list[str] = []

    if receipt.get("schema_version") != "full_cpt_acquisition_receipt_v1":
        errors.append("input receipt has an unsupported schema")
    if receipt.get("status") != "passed":
        errors.append("input receipt is not passed")
    current_config_sha256 = sha256_file(sources_path)
    if receipt.get("sources_config_sha256") != current_config_sha256:
        errors.append("input receipt does not match the current sources.json")

    route, acquisition_source_id, include_globs, exclude_globs = route_for_source(config, source_id)
    if text_column not in route.get("text_columns", []):
        errors.append(
            f"{source_id}: text column {text_column!r} is not tracked in {route.get('text_columns', [])}"
        )
    if id_column not in route.get("id_columns", []):
        errors.append(f"{source_id}: id column {id_column!r} is not tracked in {route.get('id_columns', [])}")
    expected_source_column = route.get("source_column")
    if expected_source_column is not None and source_column != expected_source_column:
        errors.append(
            f"{source_id}: source column {source_column!r} differs from tracked {expected_source_column!r}"
        )

    receipt_sources = [
        row for row in receipt.get("sources", []) if isinstance(row, dict) and row.get("source_id") == acquisition_source_id
    ]
    if len(receipt_sources) != 1:
        errors.append(
            f"receipt must contain exactly one {acquisition_source_id!r} acquisition entry; "
            f"found {len(receipt_sources)}"
        )
        receipt_source: dict = {}
    else:
        receipt_source = receipt_sources[0]

    acquisition_registry = {
        "nanochat_base": config["base"],
        "apertus_overlap_overlay": config["apertus_overlap_overlay"],
        "modern_greek_148k_tokenizer": config["tokenizer"],
        **{str(row["source_id"]): row for row in config.get("sources", [])},
    }
    tracked_acquisition = acquisition_registry.get(acquisition_source_id, {})
    for field in ("repo_id", "revision"):
        if receipt_source.get(field) != tracked_acquisition.get(field):
            errors.append(f"{acquisition_source_id}: receipt {field} does not match sources.json")

    root_value = receipt_source.get("local_root")
    local_root = Path(root_value).resolve() if isinstance(root_value, str) and root_value else None
    selected: dict[Path, dict] = {}
    all_receipt_files = receipt_source.get("files", [])
    if not isinstance(all_receipt_files, list):
        errors.append(f"{acquisition_source_id}: receipt files must be a list")
        all_receipt_files = []
    for index, row in enumerate(all_receipt_files):
        if not isinstance(row, dict):
            errors.append(f"{acquisition_source_id}: receipt file {index} is not an object")
            continue
        relative = row.get("path")
        local_value = row.get("local_path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{acquisition_source_id}: receipt file {index} has no relative path")
            continue
        if not matches(relative, include_globs, exclude_globs):
            continue
        if not relative.lower().endswith(".parquet"):
            continue
        if local_root is None or not isinstance(local_value, str) or not local_value:
            errors.append(f"{acquisition_source_id}:{relative}: missing local path/root")
            continue
        relative_path = Path(relative)
        expected_local_path = (local_root / relative_path).resolve()
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"{acquisition_source_id}:{relative}: unsafe relative path")
            continue
        if Path(local_value).resolve() != expected_local_path:
            errors.append(f"{acquisition_source_id}:{relative}: receipt local path is inconsistent")
            continue
        try:
            expected_local_path.relative_to(local_root)
        except ValueError:
            errors.append(f"{acquisition_source_id}:{relative}: local path escapes acquisition root")
            continue
        if not valid_content_identifier(row.get("hash_kind"), row.get("expected_hash")):
            errors.append(f"{acquisition_source_id}:{relative}: invalid immutable content identifier")
        if expected_local_path in selected:
            errors.append(f"{acquisition_source_id}:{relative}: duplicate resolved receipt path")
        selected[expected_local_path] = row

    if not selected:
        errors.append(
            f"{source_id}: no receipt Parquets matched acquisition globs {include_globs!r}"
        )

    declared_count = receipt_source.get("selected_file_count")
    declared_bytes = receipt_source.get("selected_bytes")
    try:
        all_file_bytes = sum(int(row["size"]) for row in all_receipt_files if isinstance(row, dict))
    except (KeyError, TypeError, ValueError):
        errors.append(f"{acquisition_source_id}: receipt contains an invalid file size")
        all_file_bytes = -1
    if declared_count != len(all_receipt_files):
        errors.append(f"{acquisition_source_id}: receipt selected_file_count is inconsistent")
    if declared_bytes != all_file_bytes:
        errors.append(f"{acquisition_source_id}: receipt selected_bytes is inconsistent")

    actual_inputs = expand_inputs(input_values)
    actual_set = set(actual_inputs)
    expected_set = set(selected)
    if actual_set != expected_set:
        missing = sorted(str(path) for path in expected_set - actual_set)
        unexpected = sorted(str(path) for path in actual_set - expected_set)
        errors.append(
            f"resolved input path set differs from receipt route; missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )

    total_bytes = 0
    for path in actual_inputs:
        if not path.is_file():
            errors.append(f"receipt-bound input is missing: {path}")
            continue
        expected = selected.get(path)
        if expected is None:
            continue
        current_stat = path.stat()
        actual_size = current_stat.st_size
        try:
            expected_size = int(expected["size"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{path}: receipt has no valid size")
            continue
        if actual_size != expected_size:
            errors.append(f"{path}: size drifted after acquisition (expected {expected_size}, got {actual_size})")
        stat_bindings = {
            "device": current_stat.st_dev,
            "inode": current_stat.st_ino,
            "mtime_ns": current_stat.st_mtime_ns,
            "ctime_ns": current_stat.st_ctime_ns,
        }
        for field, actual in stat_bindings.items():
            if expected.get(field) != actual:
                errors.append(f"{path}: {field} drifted after acquisition")
        total_bytes += actual_size

    if errors:
        raise ValueError("input receipt validation failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": "full_cpt_input_receipt_check_v1",
        "ok": True,
        "source": source_id,
        "acquisition_source_id": acquisition_source_id,
        "revision": receipt_source["revision"],
        "text_column": text_column,
        "id_column": id_column,
        "source_column": source_column,
        "input_receipt": str(receipt_path.resolve()),
        "input_receipt_sha256": sha256_file(receipt_path),
        "sources_config_sha256": current_config_sha256,
        "content_check": (
            "exact resolved path set, file sizes and acquisition-time stat identity; payload hashes were verified during "
            "acquisition and are not re-read at launch"
        ),
        "files": len(actual_inputs),
        "bytes": total_bytes,
        "paths": [str(path) for path in actual_inputs],
        "inputs": [{"path": str(path), "bytes": path.stat().st_size} for path in actual_inputs],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", action="append", required=True, help="Parquet path/dir/glob; repeatable")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--source-column", default="source_dataset")
    args = parser.parse_args()

    result = validate_input(
        receipt_path=args.receipt,
        sources_path=args.sources,
        source_id=args.source,
        input_values=args.input,
        text_column=args.text_column,
        id_column=args.id_column,
        source_column=args.source_column,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

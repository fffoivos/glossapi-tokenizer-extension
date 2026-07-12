#!/usr/bin/env python3
"""Merge verified Hugging Face and MDC acquisitions into one downstream receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from acquire_mdc_sources import (  # noqa: E402
    validate_payload,
    validate_source_receipt,
)


HEX = frozenset("0123456789abcdef")
MDC_PAYLOAD_VALIDATION_SCHEMA = "full_cpt_mdc_payload_validation_v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_receipt(
    *,
    sources_path: Path,
    hf_path: Path,
    mdc_path: Path,
    destination_root: Path,
    expected_code_commit: str,
) -> dict[str, Any]:
    config = load_object(sources_path)
    hf = load_object(hf_path)
    mdc = load_object(mdc_path)
    errors: list[str] = []
    if (
        len(expected_code_commit) != 40
        or set(expected_code_commit) - HEX
    ):
        raise ValueError("expected code commit must be a full lowercase Git commit")
    config_hash = sha256_file(sources_path)
    registry = {
        "nanochat_base": dict(config.get("base", {})),
        "apertus_overlap_overlay": dict(config.get("apertus_overlap_overlay", {})),
        "modern_greek_148k_tokenizer": dict(config.get("tokenizer", {})),
        **{
            str(row["source_id"]): dict(row)
            for row in config.get("sources", [])
            if isinstance(row, dict) and row.get("source_id")
        },
    }
    expected = set(registry)
    if hf.get("schema_version") != "full_cpt_acquisition_receipt_v1" or hf.get("status") != "passed":
        errors.append("Hugging Face acquisition receipt is not passed")
    if mdc.get("schema_version") != "full_cpt_mdc_acquisition_receipt_v1" or mdc.get("status") != "passed":
        errors.append("MDC acquisition receipt is not passed")
    for label, receipt in (("HF", hf), ("MDC", mdc)):
        if receipt.get("sources_config_sha256") != config_hash:
            errors.append(f"{label} acquisition receipt is not bound to current sources.json")
    for label, receipt in (("HF", hf), ("MDC", mdc)):
        if receipt.get("code_commit") != expected_code_commit:
            errors.append(
                f"{label} acquisition code commit differs from the explicitly submitted commit"
            )

    component_ids: dict[str, set[str]] = {"HF": set(), "MDC": set()}
    rows: dict[str, dict[str, Any]] = {}
    for label, receipt in (("HF", hf), ("MDC", mdc)):
        for row in receipt.get("sources", []):
            source_id = str(row.get("source_id", ""))
            if not source_id:
                errors.append(f"{label} receipt contains a source without source_id")
            elif source_id in rows:
                errors.append(f"source {source_id!r} appears in multiple acquisition receipts")
            else:
                component_ids[label].add(source_id)
                rows[source_id] = dict(row)
    external_ids = {
        str(row["source_id"])
        for row in config.get("sources", [])
        if isinstance(row, dict)
        and row.get("acquisition_kind") == "mozilla_data_collective"
    }
    expected_hf_ids = expected - external_ids
    if component_ids["HF"] != expected_hf_ids:
        errors.append(
            "HF acquisition identities differ from the registry: "
            f"missing={sorted(expected_hf_ids - component_ids['HF'])}, "
            f"unexpected={sorted(component_ids['HF'] - expected_hf_ids)}"
        )
    if component_ids["MDC"] != external_ids:
        errors.append(
            "MDC acquisition identities differ from the registry: "
            f"missing={sorted(external_ids - component_ids['MDC'])}, "
            f"unexpected={sorted(component_ids['MDC'] - external_ids)}"
        )
    if set(rows) != expected:
        errors.append(
            "combined acquisition identities differ from the registry: "
            f"missing={sorted(expected - set(rows))}, unexpected={sorted(set(rows) - expected)}"
        )
    destination_root = destination_root.resolve()
    raw_mdc_destination = Path(str(mdc.get("destination", "")))
    mdc_destination = raw_mdc_destination.resolve()
    try:
        mdc_destination.relative_to(destination_root)
    except ValueError:
        errors.append("MDC acquisition destination is outside the selected destination root")
    if not mdc_destination.is_dir() or raw_mdc_destination.is_symlink():
        errors.append("MDC acquisition destination is absent or linked")
    for source_id, row in rows.items():
        tracked = registry.get(source_id, {})
        mdc_payload_paths: list[Path] = []
        mdc_payload_files_valid = True
        for field in ("repo_id", "revision"):
            if row.get(field) != tracked.get(field):
                errors.append(f"{source_id}: acquisition {field} differs from sources.json")
        if source_id in external_ids:
            source_root = mdc_destination / source_id / str(tracked.get("revision", ""))
            payload_root = source_root / "payload"
            if Path(str(row.get("local_root", ""))).resolve() != payload_root.resolve():
                errors.append(f"{source_id}: MDC payload root differs from its acquisition root")
            if row.get("mdc_dataset_id") != tracked.get("mdc_dataset_id"):
                errors.append(f"{source_id}: acquisition MDC dataset ID differs from sources.json")
            if row.get("source_config_sha256") != canonical_object_sha256(tracked):
                errors.append(f"{source_id}: acquisition source configuration drift")
            archive = row.get("archive")
            pinned_archive_sha = str(tracked.get("mdc_expected_sha256", ""))
            if (
                not isinstance(archive, dict)
                or archive.get("sha256") != pinned_archive_sha
                or archive.get("registry_sha256") != pinned_archive_sha
                or archive.get("metadata_sha256") != pinned_archive_sha
            ):
                errors.append(f"{source_id}: MDC archive receipt differs from registry SHA-256")
            else:
                raw_archive_path = Path(str(archive.get("local_path", "")))
                archive_path = raw_archive_path.resolve()
                expected_archive_path = (
                    source_root
                    / "archive"
                    / str(tracked.get("mdc_expected_filename", ""))
                ).resolve()
                try:
                    archive_path.relative_to(destination_root)
                except ValueError:
                    errors.append(
                        f"{source_id}: MDC archive is outside destination root"
                    )
                else:
                    if archive_path != expected_archive_path:
                        errors.append(
                            f"{source_id}: MDC archive path differs from its acquisition root"
                        )
                    elif (
                        not archive_path.is_file()
                        or raw_archive_path.is_symlink()
                        or archive_path.name != tracked.get("mdc_expected_filename")
                        or archive_path.stat().st_size
                        != int(tracked.get("mdc_expected_bytes", -1))
                        or archive_path.stat().st_size
                        != int(archive.get("bytes", -1))
                    ):
                        errors.append(
                            f"{source_id}: pinned MDC archive is missing or size-drifted"
                        )
                    elif sha256_file(archive_path) != pinned_archive_sha:
                        errors.append(
                            f"{source_id}: pinned MDC archive SHA-256 drift"
                        )
            audit = row.get("payload_validation")
            tracked_format = str(tracked.get("mdc_format", "")).upper()
            if not isinstance(audit, dict):
                errors.append(f"{source_id}: MDC payload validation receipt is absent")
            else:
                if (
                    audit.get("schema_version") != MDC_PAYLOAD_VALIDATION_SCHEMA
                    or audit.get("status") != "passed"
                    or audit.get("format") != tracked_format
                    or audit.get("source_config_sha256")
                    != canonical_object_sha256(tracked)
                ):
                    errors.append(f"{source_id}: MDC payload validation binding is invalid")
                if tracked_format != "PARQUET":
                    errors.append(
                        f"{source_id}: unsupported MDC payload format {tracked_format!r}"
                    )
                elif int(audit.get("total_rows", 0)) < 1:
                    errors.append(f"{source_id}: MDC payload validation has zero rows")
                audit_files = audit.get("files", [])
                if not isinstance(audit_files, list) or int(
                    audit.get("selected_file_count", -1)
                ) != len(audit_files):
                    errors.append(
                        f"{source_id}: MDC payload validation file inventory is invalid"
                    )
                else:
                    configured_text = sorted(
                        {
                            str(value)
                            for value in (
                                list(tracked.get("text_columns", []))
                                + list(tracked.get("alternate_text_columns", []))
                            )
                            if str(value)
                        }
                    )
                    configured_ids = sorted(
                        {
                            str(value)
                            for value in tracked.get("id_columns", [])
                            if str(value)
                        }
                    )
                    if audit.get("candidate_text_columns") != configured_text or audit.get(
                        "candidate_id_columns"
                    ) != configured_ids:
                        errors.append(
                            f"{source_id}: MDC payload validation column contract drift"
                        )
                    audited_rows = 0
                    for file in audit_files:
                        if not isinstance(file, dict):
                            errors.append(
                                f"{source_id}: MDC payload validation has an invalid file row"
                            )
                            continue
                        rows_count = int(file.get("rows", 0))
                        audited_rows += rows_count
                        if rows_count < 1 or not file.get("present_text_columns"):
                            errors.append(
                                f"{source_id}: MDC payload validation has an empty/invalid Parquet shard"
                            )
                        if configured_ids and not file.get("present_id_columns"):
                            errors.append(
                                f"{source_id}: MDC payload validation shard has no configured identifier"
                            )
                    if audited_rows != int(audit.get("total_rows", -1)):
                        errors.append(
                            f"{source_id}: MDC payload validation row-count binding drift"
                        )
                    audited_paths = {
                        str(Path(str(file.get("local_path", ""))).resolve())
                        for file in audit_files
                        if isinstance(file, dict)
                    }
                    acquired_paths = {
                        str(Path(str(file.get("local_path", ""))).resolve())
                        for file in row.get("files", [])
                        if isinstance(file, dict)
                    }
                    if audited_paths != acquired_paths:
                        errors.append(
                            f"{source_id}: MDC payload validation inventory differs from acquisition"
                        )
        for file in row.get("files", []):
            raw_path = Path(str(file.get("local_path", "")))
            path = raw_path.resolve()
            try:
                path.relative_to(destination_root)
            except ValueError:
                errors.append(f"{source_id}: acquired file is outside destination root: {path}")
                mdc_payload_files_valid = False
                continue
            if (
                not path.is_file()
                or raw_path.is_symlink()
                or path.stat().st_size != int(file.get("size", -1))
            ):
                errors.append(f"{source_id}: acquired file is missing or size-drifted: {path}")
                mdc_payload_files_valid = False
                continue
            stat = path.stat()
            for field, actual in {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }.items():
                if int(file.get(field, -1)) != actual:
                    errors.append(f"{source_id}: acquired file {field} drifted: {path}")
                    mdc_payload_files_valid = False
            if source_id in external_ids:
                try:
                    path.relative_to(
                        mdc_destination
                        / source_id
                        / str(tracked.get("revision", ""))
                        / "payload"
                    )
                except ValueError:
                    errors.append(
                        f"{source_id}: MDC payload file is outside its acquisition payload root: {path}"
                    )
                    mdc_payload_files_valid = False
                digest = str(file.get("expected_hash", ""))
                if file.get("hash_kind") != "sha256" or len(digest) != 64 or set(digest) - HEX:
                    errors.append(f"{source_id}: MDC payload lacks a valid SHA-256 binding: {path}")
                    mdc_payload_files_valid = False
                elif sha256_file(path) != digest:
                    errors.append(f"{source_id}: MDC payload SHA-256 drift: {path}")
                    mdc_payload_files_valid = False
                else:
                    mdc_payload_paths.append(path)
        if source_id in external_ids and mdc_payload_files_valid:
            source_receipt_path = (
                mdc_destination
                / source_id
                / str(tracked.get("revision", ""))
                / "source_receipt.json"
            )
            try:
                fresh_source_receipt = validate_source_receipt(
                    source_receipt_path, tracked
                )
            except (OSError, TypeError, ValueError) as error:
                errors.append(f"{source_id}: source receipt revalidation failed: {error}")
            else:
                if fresh_source_receipt != row:
                    errors.append(
                        f"{source_id}: top-level MDC row differs from immutable source receipt"
                    )
            try:
                fresh_payload_validation = validate_payload(mdc_payload_paths, tracked)
            except (OSError, TypeError, ValueError) as error:
                errors.append(
                    f"{source_id}: fresh MDC payload validation failed: {error}"
                )
            else:
                if fresh_payload_validation != row.get("payload_validation"):
                    errors.append(
                        f"{source_id}: embedded MDC payload validation differs from fresh recomputation"
                    )
    if errors:
        raise ValueError("acquisition receipt merge failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": "full_cpt_acquisition_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": expected_code_commit,
        "destination": str(destination_root),
        "sources_config": str(sources_path.resolve()),
        "sources_config_sha256": config_hash,
        "component_receipts": [
            {"kind": "huggingface", "path": str(hf_path.resolve()), "sha256": sha256_file(hf_path)},
            {"kind": "mozilla_data_collective", "path": str(mdc_path.resolve()), "sha256": sha256_file(mdc_path)},
        ],
        "content_verification": (
            "HF LFS/blob and schema verification plus MDC archive/extracted-file "
            "SHA-256 and format-specific payload schema verification"
        ),
        "sources": [rows[source_id] for source_id in sorted(rows)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--hf-receipt", type=Path, required=True)
    parser.add_argument("--mdc-receipt", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        sources_path=args.sources,
        hf_path=args.hf_receipt,
        mdc_path=args.mdc_receipt,
        destination_root=args.destination_root,
        expected_code_commit=args.expected_code_commit,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps({"ok": True, "sources": len(receipt["sources"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

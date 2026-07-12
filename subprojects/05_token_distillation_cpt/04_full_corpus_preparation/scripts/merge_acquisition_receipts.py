#!/usr/bin/env python3
"""Merge verified Hugging Face and MDC acquisitions into one downstream receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HEX = frozenset("0123456789abcdef")
MDC_PAYLOAD_VALIDATION_SCHEMA = "full_cpt_mdc_payload_validation_v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
) -> dict[str, Any]:
    config = load_object(sources_path)
    hf = load_object(hf_path)
    mdc = load_object(mdc_path)
    errors: list[str] = []
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
    if hf.get("code_commit") != mdc.get("code_commit"):
        errors.append("HF and MDC acquisition code commits differ")

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
    for source_id, row in rows.items():
        tracked = registry.get(source_id, {})
        for field in ("repo_id", "revision"):
            if row.get(field) != tracked.get(field):
                errors.append(f"{source_id}: acquisition {field} differs from sources.json")
        if source_id in external_ids:
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
            path = Path(str(file.get("local_path", ""))).resolve()
            try:
                path.relative_to(destination_root)
            except ValueError:
                errors.append(f"{source_id}: acquired file is outside destination root: {path}")
                continue
            if not path.is_file() or path.stat().st_size != int(file.get("size", -1)):
                errors.append(f"{source_id}: acquired file is missing or size-drifted: {path}")
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
            if source_id in external_ids:
                digest = str(file.get("expected_hash", ""))
                if file.get("hash_kind") != "sha256" or len(digest) != 64 or set(digest) - HEX:
                    errors.append(f"{source_id}: MDC payload lacks a valid SHA-256 binding: {path}")
    if errors:
        raise ValueError("acquisition receipt merge failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": "full_cpt_acquisition_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": hf["code_commit"],
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        sources_path=args.sources,
        hf_path=args.hf_receipt,
        mdc_path=args.mdc_receipt,
        destination_root=args.destination_root,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps({"ok": True, "sources": len(receipt["sources"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

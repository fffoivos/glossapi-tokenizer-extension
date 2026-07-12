#!/usr/bin/env python3
"""Derive a SPAN rehydration source receipt from a passed Phase-04 acquisition."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mdc_safe_extract import tree_manifest
from .span_rehydration import (
    RehydrationError,
    _open_jsonl_text,
    canonical_json_sha256,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
PHASE04_DIR = HERE.parents[3] / "04_full_corpus_preparation"
DEFAULT_SOURCES = PHASE04_DIR / "configs" / "sources.json"
DEFAULT_MANIFEST = EVAL_DIR / "units" / "SPAN_manifest.jsonl"
DEFAULT_ANNOTATIONS = EVAL_DIR / "annotations_span" / "all.json"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_LOGICAL_SOURCES = {"greek_phd", "openarchives", "kallipos"}
MDC_GREEK_PHD_DATASET_ID = "cmkwvpu7s0032mo07jpk20pj1"


@dataclass(frozen=True)
class Route:
    logical_source: str
    acquisition_source_id: str
    path_patterns: tuple[str, ...]
    format: str
    fields: Mapping[str, Any]
    historical_source_relation: str
    document_id_alignment: str


def _load_object(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehydrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise RehydrationError(f"{path}: expected schema_version {schema}")
    return value


def _unique_rows(rows: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RehydrationError(f"{name}: expected a list")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("source_id"):
            raise RehydrationError(f"{name}:{index}: invalid source row")
        source_id = str(row["source_id"])
        if source_id in result:
            raise RehydrationError(f"{name}: duplicate source_id {source_id!r}")
        result[source_id] = row
    return result


def _config_sources(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {
        "nanochat_base": config.get("base", {}),
        "apertus_overlap_overlay": config.get("apertus_overlap_overlay", {}),
        "modern_greek_148k_tokenizer": config.get("tokenizer", {}),
    }
    for row in config.get("sources", []):
        if isinstance(row, dict) and row.get("source_id"):
            result[str(row["source_id"])] = row
    return result


def _under(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _greek_manifest_expectations(
    manifest_path: Path, annotations_path: Path
) -> dict[str, Any]:
    units: list[str] = []
    documents: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RehydrationError(
                    f"{manifest_path}:{line_number}: invalid JSON"
                ) from exc
            if isinstance(row, dict) and row.get("source") == "greek_phd":
                unit_id = row.get("unit_id")
                doc_id = row.get("doc_id")
                if not isinstance(unit_id, str) or not isinstance(doc_id, str):
                    raise RehydrationError(
                        f"{manifest_path}:{line_number}: invalid Greek-PhD identity"
                    )
                units.append(unit_id)
                documents.add(doc_id)
    if not units or len(units) != len(set(units)):
        raise RehydrationError(f"{manifest_path}: invalid Greek-PhD unit inventory")
    try:
        value = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehydrationError(f"cannot read {annotations_path}: {exc}") from exc
    rows = value.get("annotations")
    if not isinstance(rows, list):
        raise RehydrationError(f"{annotations_path}: annotations must be a list")
    by_unit: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("unit_id"), str):
            raise RehydrationError(f"{annotations_path}:{index}: invalid annotation")
        unit_id = str(row["unit_id"])
        if unit_id in by_unit:
            raise RehydrationError(f"{annotations_path}: duplicate unit {unit_id}")
        by_unit[unit_id] = row
    unit_set = set(units)
    positive_spans = 0
    for unit_id in sorted(unit_set & set(by_unit)):
        spans = by_unit[unit_id].get("spans")
        if not isinstance(spans, list):
            raise RehydrationError(f"{annotations_path}: {unit_id} spans are malformed")
        positive_spans += len(spans)
    return {
        "target_documents": len(documents),
        "manifest_units": len(units),
        "positive_spans": positive_spans,
        "missing_annotation_units": sorted(unit_set - set(by_unit)),
    }


def _read_safe_extraction_manifest(
    manifest_path: Path, extracted_root: Path
) -> dict[str, tuple[Path, str, int]]:
    value = _load_object(manifest_path, "mdc_safe_extraction_manifest_v1")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise RehydrationError(f"{manifest_path}: empty extracted inventory")
    result: dict[str, tuple[Path, str, int]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RehydrationError(f"{manifest_path}: malformed file row {index}")
        relative = Path(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        size = row.get("bytes")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or not HEX64_RE.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RehydrationError(f"{manifest_path}: unsafe file row {index}")
        key = relative.as_posix()
        if key in result:
            raise RehydrationError(f"{manifest_path}: duplicate extracted path {key}")
        resolved = (extracted_root / relative).resolve()
        if not _under(extracted_root, resolved) or not resolved.is_file():
            raise RehydrationError(
                f"{manifest_path}: extracted file is missing or unsafe: {key}"
            )
        result[key] = (resolved, digest, size)
    if (
        int(value.get("file_count", -1)) != len(result)
        or int(value.get("total_file_bytes", -1))
        != sum(size for _, _, size in result.values())
    ):
        raise RehydrationError(f"{manifest_path}: aggregate inventory drift")
    return result


def _mdc_raw_forensic_source(
    args: argparse.Namespace,
    manifest_path: Path,
    annotations_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not args.allow_quarantined_mdc_comparison_only:
        raise RehydrationError(
            "mdc_raw_forensic requires --allow-quarantined-mdc-comparison-only"
        )
    expected_observed = str(args.mdc_expected_observed_sha256 or "")
    if not HEX64_RE.fullmatch(expected_observed):
        raise RehydrationError(
            "mdc_raw_forensic requires --mdc-expected-observed-sha256"
        )
    if not args.mdc_quarantine_receipt or not args.mdc_span_audit_receipt:
        raise RehydrationError(
            "mdc_raw_forensic requires quarantine and span-audit receipts"
        )
    quarantine_path = Path(args.mdc_quarantine_receipt).resolve()
    audit_path = Path(args.mdc_span_audit_receipt).resolve()
    quarantine = _load_object(
        quarantine_path, "mdc_quarantined_object_receipt_v2"
    )
    if quarantine.get("status") != "quarantined_publisher_checksum_mismatch":
        raise RehydrationError(f"{quarantine_path}: object is not quarantined")
    archive = quarantine.get("archive")
    extracted = quarantine.get("extracted")
    safe_extraction = quarantine.get("safe_extraction")
    if (
        not isinstance(archive, dict)
        or not isinstance(extracted, dict)
        or not isinstance(safe_extraction, dict)
    ):
        raise RehydrationError(f"{quarantine_path}: incomplete quarantine receipt")
    archive_path = Path(str(archive.get("path", ""))).resolve()
    if not archive_path.is_file() or archive_path.parent != quarantine_path.parent:
        raise RehydrationError(f"{quarantine_path}: unsafe or missing archive path")
    if archive.get("observed_sha256") != expected_observed:
        raise RehydrationError(f"{quarantine_path}: observed SHA-256 differs from explicit pin")
    publisher_sha = str(archive.get("publisher_declared_sha256", ""))
    if (
        not HEX64_RE.fullmatch(publisher_sha)
        or publisher_sha == expected_observed
        or archive.get("gzip_and_tar_integrity") != "passed"
    ):
        raise RehydrationError(f"{quarantine_path}: invalid quarantine hash/integrity state")
    if sha256_file(archive_path) != expected_observed:
        raise RehydrationError(f"{archive_path}: observed archive SHA-256 drift")
    if int(archive.get("bytes", -1)) != archive_path.stat().st_size:
        raise RehydrationError(f"{quarantine_path}: archive byte count drift")

    extracted_manifest = Path(str(extracted.get("sha256_manifest_path", ""))).resolve()
    extraction_receipt_path = Path(
        str(safe_extraction.get("receipt_path", ""))
    ).resolve()
    if (
        extracted_manifest.parent != quarantine_path.parent
        or not extracted_manifest.is_file()
        or sha256_file(extracted_manifest) != extracted.get("sha256_manifest_sha256")
        or extraction_receipt_path.parent != quarantine_path.parent
        or not extraction_receipt_path.is_file()
        or sha256_file(extraction_receipt_path)
        != safe_extraction.get("receipt_sha256")
        or safe_extraction.get("status") != "passed_fresh_archive_tree_matches"
    ):
        raise RehydrationError(f"{quarantine_path}: safe extraction binding drift")
    extraction_receipt = _load_object(
        extraction_receipt_path, "mdc_safe_extraction_receipt_v1"
    )
    extraction_archive = extraction_receipt.get("archive")
    extraction = extraction_receipt.get("extraction")
    extraction_tool = extraction_receipt.get("tool")
    extracted_root = Path(str(extracted.get("path", ""))).resolve()
    if (
        extraction_receipt.get("status") != "passed_fresh_archive_tree_matches"
        or not isinstance(extraction_archive, dict)
        or Path(str(extraction_archive.get("path", ""))).resolve() != archive_path
        or extraction_archive.get("sha256") != expected_observed
        or not isinstance(extraction, dict)
        or Path(str(extraction.get("root", ""))).resolve() != extracted_root
        or Path(str(extraction.get("manifest_path", ""))).resolve()
        != extracted_manifest
        or extraction.get("manifest_sha256") != sha256_file(extracted_manifest)
        or not isinstance(extraction_tool, dict)
        or extraction_tool.get("sha256")
        != sha256_file(HERE / "mdc_safe_extract.py")
    ):
        raise RehydrationError(f"{extraction_receipt_path}: extraction provenance drift")
    if not extracted_root.is_dir() or extracted_root.parent != quarantine_path.parent:
        raise RehydrationError(f"{quarantine_path}: extracted root is absent or unsafe")
    expected_tree = _load_object(
        extracted_manifest, "mdc_safe_extraction_manifest_v1"
    )
    if tree_manifest(extracted_root) != expected_tree:
        raise RehydrationError(f"{quarantine_path}: extracted tree differs from safe manifest")
    inventory = _read_safe_extraction_manifest(extracted_manifest, extracted_root)
    if int(extracted.get("file_count", -1)) != len(inventory):
        raise RehydrationError(f"{quarantine_path}: extracted file count drift")
    prefix = "phd-theses-corpus/contents/"
    selected = {
        relative: item
        for relative, item in inventory.items()
        if relative.startswith(prefix) and relative.endswith(".jsonl.zst")
    }
    if not selected:
        raise RehydrationError(f"{quarantine_path}: no MDC content shards selected")
    artifacts: list[dict[str, Any]] = []
    for relative, (path, expected_sha, expected_bytes) in sorted(selected.items()):
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha or path.stat().st_size != expected_bytes:
            raise RehydrationError(f"{path}: extracted shard SHA-256 drift")
        artifacts.append(
            {
                "path": str(path),
                "repository_path": relative,
                "sha256": actual_sha,
                "bytes": path.stat().st_size,
                "acquisition_hash_kind": "quarantined_extracted_sha256",
            }
        )

    audit = _load_object(audit_path, "mdc_greek_phd_span_coordinate_audit_v2")
    if (
        audit.get("status")
        not in {"passed", "comparison_only_with_silver_anomalies"}
        or audit.get("snapshot_equivalence_to_historical_span_inputs") != "unverified"
        or audit.get("research_evidence_scope") != "LLM_silver_comparison_only"
        or audit.get("production_eligible") is not False
    ):
        raise RehydrationError(f"{audit_path}: audit is not comparison-only eligible")
    audit_archive = audit.get("archive")
    audit_inputs = audit.get("inputs")
    if not isinstance(audit_archive, dict) or not isinstance(audit_inputs, dict):
        raise RehydrationError(f"{audit_path}: incomplete archive/input bindings")
    if (
        Path(str(audit_archive.get("path", ""))).resolve() != archive_path
        or int(audit_archive.get("bytes", -1)) != archive_path.stat().st_size
        or audit_archive.get("observed_sha256") != expected_observed
        or audit_archive.get("publisher_declared_sha256") != publisher_sha
        or audit_archive.get("publisher_checksum_matches") is not False
    ):
        raise RehydrationError(f"{audit_path}: archive binding differs from quarantine")
    audit_quarantine = audit_inputs.get("quarantine_receipt")
    audit_extraction = audit_inputs.get("safe_extraction_receipt")
    audit_tool = audit_inputs.get("tool")
    expected_audit_tool = HERE / "mdc_span_audit.py"
    if (
        not isinstance(audit_tool, dict)
        or audit_tool.get("sha256") != sha256_file(expected_audit_tool)
    ):
        raise RehydrationError(f"{audit_path}: audit tool hash differs from this checkout")
    if (
        not isinstance(audit_quarantine, dict)
        or Path(str(audit_quarantine.get("path", ""))).resolve() != quarantine_path
        or audit_quarantine.get("sha256") != sha256_file(quarantine_path)
    ):
        raise RehydrationError(f"{audit_path}: quarantine receipt binding differs")
    if (
        not isinstance(audit_extraction, dict)
        or Path(str(audit_extraction.get("path", ""))).resolve()
        != extraction_receipt_path
        or audit_extraction.get("sha256") != sha256_file(extraction_receipt_path)
        or audit_extraction.get("status") != "passed_fresh_archive_tree_matches"
    ):
        raise RehydrationError(f"{audit_path}: safe extraction binding differs")
    contents_root = (extracted_root / "phd-theses-corpus" / "contents").resolve()
    if (
        Path(str(audit_inputs.get("contents_root", ""))).resolve() != contents_root
        or int(audit_inputs.get("shard_count", -1)) != len(artifacts)
    ):
        raise RehydrationError(f"{audit_path}: selected shard inventory differs")
    audit_inventory_sha = canonical_json_sha256(
        [
            (str(Path(row["path"]).relative_to(contents_root)), row["bytes"])
            for row in artifacts
        ]
    )
    if audit_inputs.get("shard_inventory_sha256") != audit_inventory_sha:
        raise RehydrationError(f"{audit_path}: shard path/size inventory hash differs")
    for key, path in (("manifest", manifest_path), ("annotations", annotations_path)):
        item = audit_inputs.get(key)
        if (
            not isinstance(item, dict)
            or Path(str(item.get("path", ""))).resolve() != path
            or item.get("sha256") != sha256_file(path)
        ):
            raise RehydrationError(f"{audit_path}: {key} binding differs")
    source_integrity = audit.get("source_coordinate_integrity")
    failure_counts = (
        source_integrity.get("failure_counts")
        if isinstance(source_integrity, dict)
        else None
    )
    if (
        not isinstance(source_integrity, dict)
        or source_integrity.get("status") != "passed"
        or not isinstance(failure_counts, dict)
        or any(failure_counts.values())
    ):
        raise RehydrationError(f"{audit_path}: source-coordinate integrity did not pass")
    source_details = audit.get("source_details")
    projection_details = audit.get("projection_details")
    if (
        not isinstance(source_details, dict)
        or audit.get("source_details_sha256") != canonical_json_sha256(source_details)
        or not isinstance(projection_details, dict)
        or audit.get("projection_details_sha256")
        != canonical_json_sha256(projection_details)
    ):
        raise RehydrationError(f"{audit_path}: diagnostic details hash drift")
    projection = audit.get("historical_document_union_projection")
    projection_counts = projection.get("counts") if isinstance(projection, dict) else None
    if not isinstance(projection_counts, dict):
        raise RehydrationError(f"{audit_path}: projection counts are absent")
    expected = _greek_manifest_expectations(manifest_path, annotations_path)
    audit_counts = audit.get("counts")
    if not isinstance(audit_counts, dict):
        raise RehydrationError(f"{audit_path}: audit counts are absent")
    for field, expected_value in (
        ("target_documents", expected["target_documents"]),
        ("found_documents", expected["target_documents"]),
        ("manifest_units", expected["manifest_units"]),
        ("positive_spans_checked", expected["positive_spans"]),
        ("missing_annotation_units", len(expected["missing_annotation_units"])),
    ):
        if int(audit_counts.get(field, -1)) != expected_value:
            raise RehydrationError(f"{audit_path}: {field} differs from tracked inputs")
    if projection_details.get("missing_annotation_units") != expected["missing_annotation_units"]:
        raise RehydrationError(f"{audit_path}: missing-annotation ledger differs")
    selection_contract = audit_inputs.get("selection_contract")
    id_field_counts = (
        selection_contract.get("selected_document_id_field_counts")
        if isinstance(selection_contract, dict)
        else None
    )
    text_field_counts = (
        selection_contract.get("selected_text_field_counts")
        if isinstance(selection_contract, dict)
        else None
    )
    if (
        not isinstance(selection_contract, dict)
        or selection_contract.get("document_id_field") != "doc_id"
        or selection_contract.get("text_precedence")
        != ["text", "document", "content"]
        or id_field_counts != {"doc_id": expected["target_documents"]}
        or not isinstance(text_field_counts, dict)
        or set(text_field_counts) - {"text", "document", "content"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in text_field_counts.values()
        )
        or sum(text_field_counts.values()) != expected["target_documents"]
    ):
        raise RehydrationError(f"{audit_path}: JSONL field-selection contract differs")
    exact = int(projection_counts.get("exact_nonempty_spans", -1))
    adjusted = int(projection_counts.get("adjusted_nonempty_spans", -1))
    zero = int(projection_counts.get("zero_effective_spans", -1))
    if min(exact, adjusted, zero) < 0 or exact + adjusted + zero != expected["positive_spans"]:
        raise RehydrationError(f"{audit_path}: projection partition is inconsistent")
    expected_projection_status = (
        "comparison_only_with_silver_anomalies"
        if adjusted or zero
        else "exact_on_present_document_union"
    )
    expected_audit_status = (
        "comparison_only_with_silver_anomalies" if adjusted or zero else "passed"
    )
    if (
        projection.get("status") != expected_projection_status
        or audit.get("status") != expected_audit_status
        or not HEX64_RE.fullmatch(str(audit.get("selected_documents_text_sha256", "")))
    ):
        raise RehydrationError(f"{audit_path}: projection/audit status is inconsistent")
    for detail_key, count_key in (
        ("adjusted_nonempty_spans", "adjusted_nonempty_spans"),
        ("zero_effective_spans", "zero_effective_spans"),
        ("unit_window_escapes", "unit_window_escape_spans"),
    ):
        rows = projection_details.get(detail_key)
        if not isinstance(rows, list) or len(rows) != int(projection_counts.get(count_key, -1)):
            raise RehydrationError(f"{audit_path}: {detail_key} ledger/count mismatch")

    source = {
        "repo_type": "archive",
        "repo_id": f"mozilla-data-collective/{MDC_GREEK_PHD_DATASET_ID}",
        "revision": expected_observed,
        "format": "jsonl_documents",
        "fields": {
            "document_id": "doc_id",
            "text_precedence": ["text", "document", "content"],
        },
        "acquisition_source_id": "mdc_raw_forensic",
        "selection_globs": ["phd-theses-corpus/contents/*.jsonl.zst"],
        "historical_source_relation": (
            "current MDC Greek-PhD v1 raw layout with exact tracked coordinate coverage; "
            "publisher checksum mismatch keeps the object quarantined"
        ),
        "label_text_equivalence": "unverified_without_independent_historical_snapshot_digest",
        "document_id_alignment": "exact_tracked_hash_domain_coordinate_audited",
        "quarantine_receipt_sha256": sha256_file(quarantine_path),
        "safe_extraction_receipt_sha256": sha256_file(extraction_receipt_path),
        "span_coordinate_audit_sha256": sha256_file(audit_path),
        "selected_documents_text_sha256": audit["selected_documents_text_sha256"],
        "selected_document_id_field_counts": id_field_counts,
        "selected_text_field_counts": text_field_counts,
        "publisher_checksum_status": "mismatch_quarantined",
        "artifacts": artifacts,
    }
    report = {
        "format": "jsonl_documents",
        "archive_sha256": expected_observed,
        "publisher_declared_sha256": publisher_sha,
        "publisher_checksum_matches": False,
        "extracted_manifest_sha256": sha256_file(extracted_manifest),
        "safe_extraction_receipt_sha256": sha256_file(extraction_receipt_path),
        "selected_documents_text_sha256": audit["selected_documents_text_sha256"],
        "selected_document_id_field_counts": id_field_counts,
        "selected_text_field_counts": text_field_counts,
        "selected_shard_count": len(artifacts),
        "selected_shards_inventory_sha256": canonical_json_sha256(
            [
                (row["repository_path"], row["bytes"], row["sha256"])
                for row in artifacts
            ]
        ),
        "audit_status": audit["status"],
        "source_coordinate_integrity": source_integrity["status"],
        "projection_counts": projection_counts,
    }
    derivation = {
        "quarantine_receipt": str(quarantine_path),
        "quarantine_receipt_sha256": sha256_file(quarantine_path),
        "safe_extraction_receipt": str(extraction_receipt_path),
        "safe_extraction_receipt_sha256": sha256_file(extraction_receipt_path),
        "span_coordinate_audit": str(audit_path),
        "span_coordinate_audit_sha256": sha256_file(audit_path),
        "archive": str(archive_path),
        "archive_sha256": expected_observed,
        "extracted_sha256_manifest": str(extracted_manifest),
        "extracted_sha256_manifest_sha256": sha256_file(extracted_manifest),
    }
    return source, report, derivation


def _validate_phase04_bindings(
    acquisition_path: Path,
    lock_path: Path,
    sources_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    acquisition = _load_object(acquisition_path, "full_cpt_acquisition_receipt_v1")
    lock = _load_object(lock_path, "full_cpt_sources_lock_v1")
    sources = _load_object(sources_path, "full_cpt_sources_v1")
    if acquisition.get("status") != "passed":
        raise RehydrationError(f"{acquisition_path}: acquisition status is not passed")
    lock_sha = sha256_file(lock_path)
    sources_sha = sha256_file(sources_path)
    if acquisition.get("source_lock_sha256") != lock_sha:
        raise RehydrationError(
            "acquisition receipt is not bound to the supplied source lock"
        )
    if acquisition.get("sources_config_sha256") != sources_sha:
        raise RehydrationError(
            "acquisition receipt is not bound to the supplied sources.json"
        )
    if lock.get("sources_config_sha256") != sources_sha:
        raise RehydrationError("source lock is not bound to the supplied sources.json")
    recorded_lock = Path(str(acquisition.get("source_lock", ""))).resolve()
    if recorded_lock != lock_path.resolve():
        raise RehydrationError(
            f"acquisition receipt names a different source lock: {recorded_lock}"
        )
    locked = _unique_rows(lock.get("sources"), "source lock")
    acquired = _unique_rows(acquisition.get("sources"), "acquisition receipt")
    if set(locked) != set(acquired):
        raise RehydrationError("acquisition and lock source inventories differ")
    configured = _config_sources(sources)
    for source_id, lock_row in locked.items():
        receipt_row = acquired[source_id]
        config_row = configured.get(source_id)
        if config_row is None:
            raise RehydrationError(f"{source_id}: absent from sources.json")
        for field, configured_value in (
            ("repo_id", config_row.get("repo_id")),
            ("repo_type", config_row.get("repo_type", "dataset")),
            ("revision", config_row.get("revision")),
        ):
            if (
                lock_row.get(field) != configured_value
                or receipt_row.get(field) != configured_value
            ):
                raise RehydrationError(
                    f"{source_id}: {field} differs between config, lock, and acquisition receipt"
                )
        locked_files = {
            str(row.get("path")): row
            for row in lock_row.get("selected_files", [])
            if isinstance(row, dict) and row.get("path")
        }
        receipt_files = {
            str(row.get("path")): row
            for row in receipt_row.get("files", [])
            if isinstance(row, dict) and row.get("path")
        }
        if len(locked_files) != len(lock_row.get("selected_files", [])):
            raise RehydrationError(
                f"{source_id}: lock contains duplicate/invalid files"
            )
        if len(receipt_files) != len(receipt_row.get("files", [])):
            raise RehydrationError(
                f"{source_id}: acquisition contains duplicate/invalid files"
            )
        if set(locked_files) != set(receipt_files):
            raise RehydrationError(
                f"{source_id}: acquisition file inventory differs from lock"
            )
        local_root = Path(str(receipt_row.get("local_root", ""))).resolve()
        for relative, locked_file in locked_files.items():
            receipt_file = receipt_files[relative]
            lfs_sha = locked_file.get("lfs_sha256")
            expected_hash = lfs_sha or locked_file.get("blob_id")
            expected_kind = "lfs_sha256" if lfs_sha else "git_blob_id"
            if not isinstance(expected_hash, str) or not expected_hash:
                raise RehydrationError(
                    f"{source_id}:{relative}: lock has no content identity"
                )
            if (
                receipt_file.get("hash_kind") != expected_kind
                or receipt_file.get("expected_hash") != expected_hash
            ):
                raise RehydrationError(
                    f"{source_id}:{relative}: acquisition hash does not match the LFS lock"
                )
            local_path = Path(str(receipt_file.get("local_path", ""))).resolve()
            expected_local = (local_root / relative).resolve()
            if local_path != expected_local or not _under(local_root, local_path):
                raise RehydrationError(
                    f"{source_id}:{relative}: unsafe/inconsistent local path"
                )
            if not local_path.is_file():
                raise RehydrationError(
                    f"{source_id}:{relative}: acquired file is missing"
                )
            stat = local_path.stat()
            if stat.st_size != int(locked_file.get("size", -1)):
                raise RehydrationError(f"{source_id}:{relative}: acquired size drift")
            for field, actual in (
                ("device", stat.st_dev),
                ("inode", stat.st_ino),
                ("mtime_ns", stat.st_mtime_ns),
                ("ctime_ns", stat.st_ctime_ns),
            ):
                if int(receipt_file.get(field, -1)) != int(actual):
                    raise RehydrationError(
                        f"{source_id}:{relative}: acquired {field} drift"
                    )
    return acquisition, locked, acquired, configured


def _manifest_source_ids(path: Path, source: str) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise RehydrationError(
                    f"{path}:{line_number}: manifest row is not an object"
                )
            if row.get("source") == source:
                result.add(str(row.get("doc_id", "")))
    if not result or "" in result:
        raise RehydrationError(f"{path}: no valid {source} document identities")
    return result


def _routes(args: argparse.Namespace, config: Mapping[str, Any]) -> list[Route]:
    if args.greek_phd_route == "nanochat_base":
        if args.greek_phd_document_id_column or args.greek_phd_text_column:
            raise RehydrationError(
                "Greek PhD Nanochat route has pinned source_doc_id/text fields; do not override them"
            )
        greek = Route(
            logical_source="greek_phd",
            acquisition_source_id="nanochat_base",
            path_patterns=(
                "data/greek_phd.part-00000.parquet",
                "data/greek_phd.part-00001.parquet",
            ),
            format="parquet_documents",
            fields={
                "document_id": "source_doc_id",
                "text_precedence": ["text"],
                "row_filter": {"column": "source_dataset", "equals": "greek_phd"},
            },
            historical_source_relation=(
                "Nanochat processed representation of Greek PhD; closer identifier domain than v2, "
                "but not the historical raw Mozilla JSONL snapshot"
            ),
            document_id_alignment="hash_domain_compatible_unverified",
        )
    elif args.greek_phd_route == "greek_phd_v2":
        configured = _config_sources(config)["greek_phd_v2"]
        id_column = args.greek_phd_document_id_column
        if not id_column:
            raise RehydrationError(
                "greek_phd_v2 does not declare the historical hash doc_id; pass an explicit "
                "--greek-phd-document-id-column and --allow-unverified-greek-phd-id-domain"
            )
        text_precedence = args.greek_phd_text_column or [
            *configured.get("text_columns", []),
            *configured.get("alternate_text_columns", []),
        ]
        if not args.allow_unverified_greek_phd_id_domain:
            raise RehydrationError(
                "greek_phd_v2 identifier alignment is unverified; explicit "
                "--allow-unverified-greek-phd-id-domain is required"
            )
        greek = Route(
            logical_source="greek_phd",
            acquisition_source_id="greek_phd_v2",
            path_patterns=("Greek PhD Theses Corpus v2.0.parquet",),
            format="parquet_documents",
            fields={"document_id": id_column, "text_precedence": text_precedence},
            historical_source_relation=(
                "newer Greek PhD extraction; historical labels used raw Mozilla JSONL and this "
                "route must not be treated as equivalent"
            ),
            document_id_alignment="unverified_nonhistorical_identifier_requires_mapping",
        )
    else:
        if args.greek_phd_document_id_column or args.greek_phd_text_column:
            raise RehydrationError(
                "mdc_raw_forensic retains historical JSONL fields; do not override them"
            )
        greek = Route(
            logical_source="greek_phd",
            acquisition_source_id="mdc_raw_forensic",
            path_patterns=("phd-theses-corpus/contents/*.jsonl.zst",),
            format="jsonl_documents",
            fields={
                "document_id": "doc_id",
                "text_precedence": ["text", "document", "content"],
            },
            historical_source_relation="quarantined current MDC raw comparison route",
            document_id_alignment="validated_by_forensic_coordinate_audit",
        )
    if args.kallipos_route == "kallipos_sections":
        kallipos = Route(
            logical_source="kallipos",
            acquisition_source_id="kallipos_sections",
            path_patterns=("Dataset_Kallipos.parquet",),
            format="parquet_sections",
            fields={"filename": "filename", "order": "id", "section": "section"},
            historical_source_relation=(
                "same repository artifact family used by the historical section-grouping builder; "
                "revision equivalence remains unverified"
            ),
            document_id_alignment="filename_domain_compatible_unverified",
        )
    else:
        kallipos = Route(
            logical_source="kallipos",
            acquisition_source_id="nanochat_base",
            path_patterns=("data/Apothetirio_Kallipos.parquet",),
            format="parquet_documents",
            fields={
                "document_id": "source_doc_id",
                "text_precedence": ["text"],
                "row_filter": {
                    "column": "source_dataset",
                    "equals": "Apothetirio_Kallipos",
                },
            },
            historical_source_relation=(
                "Nanochat processed document representation; historical labels used grouped raw "
                "Kallipos sections"
            ),
            document_id_alignment="paper_id_domain_compatible_unverified",
        )
    if args.openarchives_route == "nanochat_base":
        openarchives = Route(
            logical_source="openarchives",
            acquisition_source_id="nanochat_base",
            path_patterns=(
                "data/openarchives.gr.parquet",
                "data/openarchives.gr.part-00000.parquet",
                "data/openarchives.gr.part-00001.parquet",
                "data/openarchives.gr.part-00002.parquet",
                "data/openarchives.gr.part-00003.parquet",
            ),
            format="parquet_documents",
            fields={
                "document_id": "source_doc_id",
                "text_precedence": ["text"],
                "row_filter": {
                    "column": "source_dataset",
                    "equals": "openarchives.gr",
                },
            },
            historical_source_relation=(
                "OpenArchives representation retained inside the pinned Nanochat base compiled "
                "before the current replacement/resegmentation candidate; exact historical "
                "SPAN snapshot equivalence remains unverified"
            ),
            document_id_alignment="hash_domain_compatible_unverified",
        )
    else:
        openarchives = Route(
            logical_source="openarchives",
            acquisition_source_id="openarchives_current",
            path_patterns=("data/openarchives/**/*.jsonl.zst",),
            format="jsonl_documents",
            fields={
                "document_id": "doc_id",
                "text_precedence": ["text", "document", "content"],
            },
            historical_source_relation=(
                "current replacement/resegmentation OpenArchives JSONL family; it is an explicit "
                "comparison route and not presumed text-equivalent to the historical SPAN source"
            ),
            document_id_alignment="doc_id_domain_compatible_unverified",
        )
    return [greek, openarchives, kallipos]


def _select_files(
    route: Route,
    lock_row: Mapping[str, Any],
    receipt_row: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    locked = {str(row["path"]): row for row in lock_row["selected_files"]}
    acquired = {str(row["path"]): row for row in receipt_row["files"]}
    selected_paths = sorted(
        path
        for path in locked
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in route.path_patterns)
    )
    if not selected_paths:
        raise RehydrationError(
            f"{route.logical_source}: route selected no {route.acquisition_source_id} artifacts"
        )
    exact_path_family = route.logical_source in {"greek_phd", "kallipos"} or (
        route.logical_source == "openarchives"
        and route.acquisition_source_id == "nanochat_base"
    )
    if exact_path_family and len(selected_paths) != len(route.path_patterns):
        raise RehydrationError(
            f"{route.logical_source}: expected {len(route.path_patterns)} exact artifacts, "
            f"selected {selected_paths}"
        )
    return [(locked[path], acquired[path]) for path in selected_paths]


def _inspect_parquet_route(route: Route, paths: Sequence[Path]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RehydrationError(
            "source-receipt building requires the pinned pyarrow runtime"
        ) from exc
    reports: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    if route.format == "parquet_sections":
        required = set(route.fields.values())
    else:
        required = {str(route.fields["document_id"])}
        required.update(str(item) for item in route.fields["text_precedence"])
        if route.fields.get("row_filter"):
            required.add(str(route.fields["row_filter"]["column"]))
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = sorted(required - columns)
        if missing:
            raise RehydrationError(
                f"{path}: route-required Parquet columns are absent: {missing}"
            )
        if route.format == "parquet_documents":
            id_column = str(route.fields["document_id"])
            for batch in parquet.iter_batches(
                batch_size=32, columns=[id_column], use_threads=False
            ):
                sample_ids.extend(
                    str(value)
                    for value in batch.column(0).to_pylist()
                    if value is not None
                )
                break
        reports.append(
            {
                "path": str(path.resolve()),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "columns": sorted(columns),
            }
        )
    return {
        "format": route.format,
        "files": reports,
        "sample_document_ids": sample_ids[:64],
    }


def _inspect_jsonl_route(route: Route, paths: Sequence[Path]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    id_field = str(route.fields["document_id"])
    text_fields = list(map(str, route.fields["text_precedence"]))
    for path in paths:
        first: dict[str, Any] | None = None
        with _open_jsonl_text(path) as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RehydrationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise RehydrationError(
                        f"{path}:{line_number}: row must be an object"
                    )
                first = value
                break
        if first is None:
            raise RehydrationError(f"{path}: empty JSONL artifact")
        if not isinstance(first.get(id_field), str) or not any(
            isinstance(first.get(field), str) for field in text_fields
        ):
            raise RehydrationError(
                f"{path}: first row lacks string {id_field!r} or all text precedence fields"
            )
        reports.append(
            {
                "path": str(path.resolve()),
                "first_row_fields": sorted(first),
                "first_document_id_sha256": hashlib.sha256(
                    str(first[id_field]).encode("utf-8")
                ).hexdigest(),
            }
        )
    return {"format": route.format, "files": reports}


def build_span_source_receipt(args: argparse.Namespace) -> dict[str, Any]:
    acquisition_path = Path(args.acquisition_receipt).resolve()
    lock_path = Path(args.source_lock).resolve()
    sources_path = Path(args.sources_config).resolve()
    manifest_path = Path(args.manifest).resolve()
    annotations_path = Path(args.annotations).resolve()
    output = Path(args.output).resolve()
    raw_route_options = (
        args.mdc_quarantine_receipt,
        args.mdc_span_audit_receipt,
        args.mdc_expected_observed_sha256,
        args.allow_quarantined_mdc_comparison_only,
    )
    if args.greek_phd_route != "mdc_raw_forensic" and any(raw_route_options):
        raise RehydrationError(
            "MDC quarantine options are valid only with greek_phd_route=mdc_raw_forensic"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    acquisition, locked, acquired, config_rows = _validate_phase04_bindings(
        acquisition_path, lock_path, sources_path
    )
    config = _load_object(sources_path, "full_cpt_sources_v1")
    routes = _routes(args, config)
    if {route.logical_source for route in routes} != REQUIRED_LOGICAL_SOURCES:
        raise RehydrationError(
            "builder did not resolve exactly the three SPAN logical sources"
        )
    logical_sources: dict[str, Any] = {}
    schema_reports: dict[str, Any] = {}
    external_derivations: dict[str, Any] = {}
    for route in routes:
        source_id = route.acquisition_source_id
        if source_id == "mdc_raw_forensic":
            source, report, external_derivation = _mdc_raw_forensic_source(
                args, manifest_path, annotations_path
            )
            logical_sources[route.logical_source] = source
            schema_reports[route.logical_source] = report
            external_derivations[route.logical_source] = external_derivation
            continue
        if (
            source_id not in locked
            or source_id not in acquired
            or source_id not in config_rows
        ):
            raise RehydrationError(
                f"{route.logical_source}: acquisition lacks required source {source_id!r}"
            )
        selected = _select_files(route, locked[source_id], acquired[source_id])
        paths = [
            Path(str(receipt_file["local_path"])).resolve()
            for _, receipt_file in selected
        ]
        report = (
            _inspect_jsonl_route(route, paths)
            if route.format == "jsonl_documents"
            else _inspect_parquet_route(route, paths)
        )
        if route.logical_source == "greek_phd" and route.format == "parquet_documents":
            manifest_ids = _manifest_source_ids(manifest_path, "greek_phd")
            if not all(HEX64_RE.fullmatch(item) for item in manifest_ids):
                raise RehydrationError(
                    "tracked Greek PhD manifest IDs are not the expected hash domain"
                )
            samples = report.get("sample_document_ids", [])
            sample_hash_compatible = bool(samples) and all(
                HEX64_RE.fullmatch(item) for item in samples
            )
            if (
                route.acquisition_source_id == "nanochat_base"
                and not sample_hash_compatible
            ):
                raise RehydrationError(
                    "Nanochat Greek PhD source_doc_id samples are not hash-domain compatible"
                )
            report["manifest_document_id_count"] = len(manifest_ids)
            report["manifest_document_ids_sha256"] = hashlib.sha256(
                json.dumps(sorted(manifest_ids), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            report["sample_id_domain_hash_compatible"] = sample_hash_compatible
        source_row = acquired[source_id]
        artifacts = []
        for locked_file, receipt_file in selected:
            lfs_sha = locked_file.get("lfs_sha256")
            if not isinstance(lfs_sha, str) or not HEX64_RE.fullmatch(lfs_sha):
                raise RehydrationError(
                    f"{source_id}:{locked_file.get('path')}: selected SPAN artifact lacks LFS SHA-256"
                )
            artifacts.append(
                {
                    "path": str(Path(str(receipt_file["local_path"])).resolve()),
                    "repository_path": str(locked_file["path"]),
                    "sha256": lfs_sha,
                    "bytes": int(locked_file["size"]),
                    "acquisition_hash_kind": "lfs_sha256",
                }
            )
        logical_sources[route.logical_source] = {
            "repo_type": str(source_row["repo_type"]),
            "repo_id": str(source_row["repo_id"]),
            "revision": str(source_row["revision"]),
            "format": route.format,
            "fields": dict(route.fields),
            "acquisition_source_id": source_id,
            "selection_globs": list(route.path_patterns),
            "historical_source_relation": route.historical_source_relation,
            "label_text_equivalence": "unverified_without_expected_snapshot_artifact_sha256",
            "document_id_alignment": route.document_id_alignment,
            "artifacts": artifacts,
        }
        schema_reports[route.logical_source] = report
    receipt = {
        "schema_version": "span-source-artifacts-v1",
        "snapshot_equivalence_status": "rehydrated_unverified_snapshot",
        "labels_read_created_or_inferred": False,
        "derivation": {
            "schema_version": "span-source-artifact-derivation-v1",
            "acquisition_receipt": str(acquisition_path),
            "acquisition_receipt_sha256": sha256_file(acquisition_path),
            "source_lock": str(lock_path),
            "source_lock_sha256": sha256_file(lock_path),
            "sources_config": str(sources_path),
            "sources_config_sha256": sha256_file(sources_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "annotations": str(annotations_path),
            "annotations_sha256": sha256_file(annotations_path),
            "acquisition_code_commit": acquisition.get("code_commit"),
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "execution_code_commit": os.environ.get("PHASE04_EXPECTED_COMMIT"),
            },
            "route_choices": {
                "greek_phd": args.greek_phd_route,
                "openarchives": args.openarchives_route,
                "kallipos": args.kallipos_route,
            },
            "schema_reports": schema_reports,
            "external_forensic_inputs": external_derivations,
        },
        "sources": logical_sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--sources-config", default=str(DEFAULT_SOURCES))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument(
        "--greek-phd-route",
        required=True,
        choices=("nanochat_base", "greek_phd_v2", "mdc_raw_forensic"),
        help="explicitly choose a Greek-PhD comparison route; no replacement is preferred implicitly",
    )
    parser.add_argument(
        "--kallipos-route",
        required=True,
        choices=("kallipos_sections", "nanochat_base"),
    )
    parser.add_argument(
        "--openarchives-route",
        required=True,
        choices=("nanochat_base", "openarchives_current"),
        help=(
            "explicitly choose the retained Nanochat representation or the current "
            "replacement/resegmentation comparison source"
        ),
    )
    parser.add_argument("--greek-phd-document-id-column")
    parser.add_argument("--greek-phd-text-column", action="append")
    parser.add_argument("--allow-unverified-greek-phd-id-domain", action="store_true")
    parser.add_argument("--mdc-quarantine-receipt")
    parser.add_argument("--mdc-span-audit-receipt")
    parser.add_argument("--mdc-expected-observed-sha256")
    parser.add_argument(
        "--allow-quarantined-mdc-comparison-only", action="store_true"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build_span_source_receipt(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "sources": sorted(receipt["sources"]),
                "snapshot_equivalence_status": receipt["snapshot_equivalence_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit a quarantined MDC Greek-PhD snapshot against tracked SPAN coordinates.

This is a provenance/coordinate check only.  It never creates labels and never
upgrades an MDC object to the historical annotation snapshot.  Source-coordinate
integrity and the historical LLM-silver document-union projection are reported as
separate claims so annotation anomalies cannot be mistaken for source drift.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, TextIO

try:
    from .span_rehydration import canonical_json_sha256, load_manifest, sha256_file
except ImportError:  # Support an immutable copied script with the eval directory on PYTHONPATH.
    from sequence_models.span_rehydration import (  # type: ignore[no-redef]
        canonical_json_sha256,
        load_manifest,
        sha256_file,
    )


TEXT_FIELDS = ("text", "document", "content")


class AuditError(ValueError):
    """The supplied raw snapshot cannot reproduce the tracked coordinates."""


@contextlib.contextmanager
def _open_jsonl(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as handle:
            yield handle
        return
    if not path.name.endswith(".jsonl.zst"):
        raise AuditError(f"unsupported content shard: {path}")
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise AuditError("zstandard is required for .jsonl.zst audit inputs") from exc
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
            with io.TextIOWrapper(decompressed, encoding="utf-8") as text:
                yield text


def _pick_string_with_field(
    row: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[str, str] | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            return field, value
    return None


def _load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("annotations") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise AuditError(f"{path}: expected an annotation list")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("unit_id"), str):
            raise AuditError(f"{path}: malformed annotation row {index}")
        unit_id = str(row["unit_id"])
        if unit_id in result:
            raise AuditError(f"{path}: duplicate annotation unit {unit_id}")
        result[unit_id] = row
    return result


def _load_quarantine_receipt(
    path: Path,
    archive_path: Path,
    publisher_declared_sha256: str,
    observed_archive_sha256: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read quarantine receipt {path}: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "mdc_quarantined_object_receipt_v2"
        or value.get("status") != "quarantined_publisher_checksum_mismatch"
    ):
        raise AuditError(f"{path}: not a quarantined MDC object receipt")
    archive = value.get("archive")
    if not isinstance(archive, dict):
        raise AuditError(f"{path}: quarantine receipt has no archive binding")
    recorded_path = Path(str(archive.get("path", ""))).resolve()
    if recorded_path != archive_path:
        raise AuditError(f"{path}: quarantine receipt names a different archive")
    if archive.get("observed_sha256") != observed_archive_sha256:
        raise AuditError(f"{path}: observed archive SHA-256 does not match")
    if archive.get("publisher_declared_sha256") != publisher_declared_sha256:
        raise AuditError(f"{path}: publisher SHA-256 does not match")
    if archive.get("gzip_and_tar_integrity") != "passed":
        raise AuditError(f"{path}: gzip/tar integrity was not recorded as passed")
    if observed_archive_sha256 == publisher_declared_sha256:
        raise AuditError(f"{path}: receipt claims a checksum mismatch but hashes are equal")
    if int(archive.get("bytes", -1)) != archive_path.stat().st_size:
        raise AuditError(f"{path}: archive byte count does not match")
    safe = value.get("safe_extraction")
    extracted = value.get("extracted")
    if not isinstance(safe, dict) or not isinstance(extracted, dict):
        raise AuditError(f"{path}: safe extraction binding is absent")
    extraction_receipt_path = Path(str(safe.get("receipt_path", ""))).resolve()
    if (
        not extraction_receipt_path.is_file()
        or sha256_file(extraction_receipt_path) != safe.get("receipt_sha256")
        or safe.get("status") != "passed_fresh_archive_tree_matches"
    ):
        raise AuditError(f"{path}: safe extraction receipt hash/status differs")
    try:
        extraction_receipt = json.loads(
            extraction_receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(
            f"cannot read safe extraction receipt {extraction_receipt_path}: {exc}"
        ) from exc
    if (
        not isinstance(extraction_receipt, dict)
        or extraction_receipt.get("schema_version")
        != "mdc_safe_extraction_receipt_v1"
        or extraction_receipt.get("status") != "passed_fresh_archive_tree_matches"
    ):
        raise AuditError(f"{extraction_receipt_path}: invalid safe extraction receipt")
    extraction_archive = extraction_receipt.get("archive")
    extraction = extraction_receipt.get("extraction")
    tool = extraction_receipt.get("tool")
    expected_tool = Path(__file__).with_name("mdc_safe_extract.py")
    if (
        not isinstance(extraction_archive, dict)
        or extraction_archive.get("sha256") != observed_archive_sha256
        or Path(str(extraction_archive.get("path", ""))).resolve() != archive_path
        or not isinstance(extraction, dict)
        or Path(str(extraction.get("root", ""))).resolve()
        != Path(str(extracted.get("path", ""))).resolve()
        or extraction.get("manifest_sha256")
        != extracted.get("sha256_manifest_sha256")
        or int(extraction.get("file_count", -1)) != int(extracted.get("file_count", -2))
        or not isinstance(tool, dict)
        or tool.get("sha256") != sha256_file(expected_tool)
    ):
        raise AuditError(f"{path}: safe extraction provenance differs")
    value["_validated_extraction_receipt_path"] = str(extraction_receipt_path)
    value["_validated_extraction_receipt_sha256"] = sha256_file(
        extraction_receipt_path
    )
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        if path.exists():
            expected = temporary.read_bytes()
            if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
                raise AuditError(f"immutable audit output differs: {path}")
            temporary.unlink()
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    annotations_path = Path(args.annotations).resolve()
    contents_root = Path(args.contents_root).resolve()
    archive_path = Path(args.archive).resolve()
    quarantine_receipt_path = Path(args.quarantine_receipt).resolve()
    if not archive_path.is_file():
        raise AuditError(f"archive is not a regular file: {archive_path}")
    observed_archive_sha256 = sha256_file(archive_path)
    quarantine_receipt = _load_quarantine_receipt(
        quarantine_receipt_path,
        archive_path,
        args.publisher_declared_sha256,
        observed_archive_sha256,
    )
    shards = sorted(contents_root.glob(args.shard_glob))
    if not shards:
        raise AuditError(f"no content shards match {contents_root / args.shard_glob}")

    manifest_rows = [row for row in load_manifest(manifest_path) if row.source == "greek_phd"]
    manifests_by_doc: dict[str, list[Any]] = {}
    for row in manifest_rows:
        manifests_by_doc.setdefault(row.doc_id, []).append(row)
    annotations = _load_annotations(annotations_path)
    required = set(manifests_by_doc)
    found: dict[str, str] = {}
    duplicate_documents: list[str] = []
    rows_scanned = 0
    parse_errors = 0
    selected_id_field_counts: collections.Counter[str] = collections.Counter()
    selected_text_field_counts: collections.Counter[str] = collections.Counter()

    for shard in shards:
        with _open_jsonl(shard) as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                rows_scanned += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if not isinstance(row, dict):
                    parse_errors += 1
                    continue
                doc_id = row.get("doc_id")
                if not isinstance(doc_id, str):
                    continue
                if doc_id not in required:
                    continue
                selected_text = _pick_string_with_field(row, TEXT_FIELDS)
                if selected_text is None:
                    raise AuditError(f"{shard}: target document {doc_id} has no text field")
                if doc_id in found:
                    duplicate_documents.append(doc_id)
                    continue
                text_field, text = selected_text
                found[doc_id] = text
                selected_id_field_counts["doc_id"] += 1
                selected_text_field_counts[text_field] += 1

    missing_documents = sorted(required - set(found))
    tail_length_mismatches: list[dict[str, Any]] = []
    window_overflows: list[dict[str, Any]] = []
    boundary_absences: list[dict[str, Any]] = []
    unit_window_escapes: list[dict[str, Any]] = []
    adjusted_nonempty_spans: list[dict[str, Any]] = []
    zero_effective_spans: list[dict[str, Any]] = []
    selected_text_receipts: list[tuple[str, str]] = []
    positive_spans_checked = 0
    exact_nonempty_spans = 0
    annotated_unit_ids: set[str] = set()

    for doc_id in sorted(found):
        text = found[doc_id]
        lines = text.split("\n")
        selected_text_receipts.append(
            (doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
        )
        units = manifests_by_doc[doc_id]
        present_document_union = {
            index
            for unit in units
            for index in range(max(0, unit.win_lo), min(len(lines), unit.win_hi))
            if lines[index].strip()
        }
        tails = [unit for unit in units if unit.window == "tail"]
        if len(tails) != 1 or tails[0].win_hi != len(lines):
            tail_length_mismatches.append(
                {
                    "doc_id": doc_id,
                    "tail_unit_ids": [unit.unit_id for unit in tails],
                    "historical_tail_win_hi": tails[0].win_hi if len(tails) == 1 else None,
                    "raw_physical_line_count": len(lines),
                }
            )
        for unit in units:
            if unit.win_hi > len(lines):
                window_overflows.append(
                    {
                        "doc_id": doc_id,
                        "unit_id": unit.unit_id,
                        "win_hi": unit.win_hi,
                        "raw_physical_line_count": len(lines),
                    }
                )
            annotation = annotations.get(unit.unit_id)
            if annotation is None:
                continue
            annotated_unit_ids.add(unit.unit_id)
            spans = annotation.get("spans", [])
            if not isinstance(spans, list):
                raise AuditError(f"annotation {unit.unit_id} has non-list spans")
            has_bib = annotation.get("has_bib")
            if not isinstance(has_bib, bool):
                raise AuditError(f"annotation {unit.unit_id} has non-boolean has_bib")
            if has_bib != bool(spans):
                raise AuditError(
                    f"annotation {unit.unit_id} has_bib is inconsistent with its spans"
                )
            for span_index, span in enumerate(spans):
                if not isinstance(span, dict):
                    raise AuditError(f"annotation {unit.unit_id} span {span_index} is malformed")
                start = span.get("start_line")
                end = span.get("end_line")
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end < start
                ):
                    raise AuditError(f"annotation {unit.unit_id} span {span_index} lacks integers")
                positive_spans_checked += 1
                absent = [
                    index
                    for index in (start, end)
                    if index < 0 or index >= len(lines) or not lines[index].strip()
                ]
                present_in_raw_span = [
                    index
                    for index in range(max(0, start), min(len(lines), end + 1))
                    if lines[index].strip()
                ]
                effective = sorted(
                    index for index in present_document_union if start <= index <= end
                )
                unit_window_contains = unit.win_lo <= start <= end < unit.win_hi
                diagnostic = {
                    "doc_id": doc_id,
                    "unit_id": unit.unit_id,
                    "span_index": span_index,
                    "start_line": start,
                    "end_line": end,
                    "unit_win_lo": unit.win_lo,
                    "unit_win_hi": unit.win_hi,
                    "unit_window_contains_declared_span": unit_window_contains,
                    "declared_span_within_raw_document": 0 <= start <= end < len(lines),
                    "absent_boundary_indices": absent,
                    "present_line_count_within_raw_span": len(present_in_raw_span),
                    "effective_present_line_count": len(effective),
                    "effective_first_present_line": effective[0] if effective else None,
                    "effective_last_present_line": effective[-1] if effective else None,
                }
                if absent:
                    boundary_absences.append(
                        {
                            **diagnostic,
                            # Compatibility aliases retained for the first forensic receipt.
                            "present_line_count_within_span": len(present_in_raw_span),
                            "first_present_line_within_span": (
                                present_in_raw_span[0] if present_in_raw_span else None
                            ),
                            "last_present_line_within_span": (
                                present_in_raw_span[-1] if present_in_raw_span else None
                            ),
                        }
                    )
                if not unit_window_contains:
                    unit_window_escapes.append(diagnostic)
                if not effective:
                    zero_effective_spans.append({**diagnostic, "outcome": "zero_effective"})
                elif absent or not unit_window_contains:
                    adjusted_nonempty_spans.append(
                        {**diagnostic, "outcome": "projected_nonempty"}
                    )
                else:
                    exact_nonempty_spans += 1

    missing_annotation_units = sorted(
        unit.unit_id
        for units in manifests_by_doc.values()
        for unit in units
        if unit.unit_id not in annotated_unit_ids
    )
    source_failures = any(
        (missing_documents, duplicate_documents, parse_errors, tail_length_mismatches, window_overflows)
    )
    source_coordinate_integrity = {
        "status": "failed" if source_failures else "passed",
        "checks": [
            "all tracked Greek-PhD document IDs found exactly once",
            "tail win_hi equals raw physical-line count",
            "all tracked windows fit within raw physical-line coordinates",
        ],
        "failure_counts": {
            "missing_documents": len(missing_documents),
            "duplicate_documents": len(set(duplicate_documents)),
            "parse_errors": parse_errors,
            "tail_length_mismatches": len(tail_length_mismatches),
            "window_overflows": len(window_overflows),
        },
    }
    projection_has_anomalies = bool(adjusted_nonempty_spans or zero_effective_spans)
    projection_status = (
        "comparison_only_with_silver_anomalies"
        if projection_has_anomalies
        else "exact_on_present_document_union"
    )
    historical_projection = {
        "status": projection_status,
        "semantics": (
            "merge present nonblank lines from all sampled windows by document, then label the "
            "intersection with each inclusive declared span"
        ),
        "zero_effective_policy": (
            "retain sampled lines as O, matching historical span_seq_data comparison semantics"
        ),
        "counts": {
            "declared_positive_spans": positive_spans_checked,
            "exact_nonempty_spans": exact_nonempty_spans,
            "adjusted_nonempty_spans": len(adjusted_nonempty_spans),
            "zero_effective_spans": len(zero_effective_spans),
            "unit_window_escape_spans": len(unit_window_escapes),
            "raw_boundary_absence_spans": len(boundary_absences),
            "missing_annotation_units": len(missing_annotation_units),
        },
    }
    status = (
        "failed"
        if source_failures
        else (
            "comparison_only_with_silver_anomalies"
            if projection_has_anomalies
            else "passed"
        )
    )
    source_details = {
        "missing_documents": missing_documents,
        "duplicate_documents": sorted(set(duplicate_documents)),
        "tail_length_mismatches": tail_length_mismatches,
        "window_overflows": window_overflows,
    }
    projection_details = {
        "missing_annotation_units": missing_annotation_units,
        "annotation_boundary_absences": boundary_absences,
        "unit_window_escapes": unit_window_escapes,
        "adjusted_nonempty_spans": adjusted_nonempty_spans,
        "zero_effective_spans": zero_effective_spans,
    }
    receipt = {
        "schema_version": "mdc_greek_phd_span_coordinate_audit_v2",
        "status": status,
        "snapshot_equivalence_to_historical_span_inputs": "unverified",
        "research_evidence_scope": "LLM_silver_comparison_only",
        "production_eligible": False,
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "observed_sha256": observed_archive_sha256,
            "publisher_declared_sha256": args.publisher_declared_sha256,
            "publisher_checksum_matches": (
                observed_archive_sha256 == args.publisher_declared_sha256
            ),
        },
        "inputs": {
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "quarantine_receipt": {
                "path": str(quarantine_receipt_path),
                "sha256": sha256_file(quarantine_receipt_path),
                "schema_version": quarantine_receipt["schema_version"],
            },
            "safe_extraction_receipt": {
                "path": quarantine_receipt["_validated_extraction_receipt_path"],
                "sha256": quarantine_receipt[
                    "_validated_extraction_receipt_sha256"
                ],
                "schema_version": "mdc_safe_extraction_receipt_v1",
                "status": "passed_fresh_archive_tree_matches",
            },
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "annotations": {
                "path": str(annotations_path),
                "sha256": sha256_file(annotations_path),
            },
            "contents_root": str(contents_root),
            "shard_count": len(shards),
            "shard_inventory_sha256": canonical_json_sha256(
                [(str(path.relative_to(contents_root)), path.stat().st_size) for path in shards]
            ),
            "selection_contract": {
                "document_id_field": "doc_id",
                "text_precedence": list(TEXT_FIELDS),
                "selected_document_id_field_counts": dict(
                    sorted(selected_id_field_counts.items())
                ),
                "selected_text_field_counts": dict(
                    sorted(selected_text_field_counts.items())
                ),
            },
        },
        "counts": {
            "rows_scanned": rows_scanned,
            "parse_errors": parse_errors,
            "target_documents": len(required),
            "found_documents": len(found),
            "manifest_units": len(manifest_rows),
            "positive_spans_checked": positive_spans_checked,
            "missing_documents": len(missing_documents),
            "duplicate_documents": len(set(duplicate_documents)),
            "tail_length_mismatches": len(tail_length_mismatches),
            "window_overflows": len(window_overflows),
            "annotation_boundary_absences": len(boundary_absences),
            "adjusted_nonempty_spans": len(adjusted_nonempty_spans),
            "zero_effective_spans": len(zero_effective_spans),
            "unit_window_escape_spans": len(unit_window_escapes),
            "missing_annotation_units": len(missing_annotation_units),
        },
        "source_coordinate_integrity": source_coordinate_integrity,
        "historical_document_union_projection": historical_projection,
        "selected_documents_text_sha256": canonical_json_sha256(selected_text_receipts),
        "source_details_sha256": canonical_json_sha256(source_details),
        "projection_details_sha256": canonical_json_sha256(projection_details),
        "source_details": source_details,
        "projection_details": projection_details,
    }
    _atomic_json(Path(args.output).resolve(), receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--quarantine-receipt", required=True)
    parser.add_argument("--publisher-declared-sha256", required=True)
    parser.add_argument("--contents-root", required=True)
    parser.add_argument("--shard-glob", default="*.jsonl.zst")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = audit(args)
    print(json.dumps({"output": args.output, "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["source_coordinate_integrity"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

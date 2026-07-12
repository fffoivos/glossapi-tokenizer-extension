#!/usr/bin/env python3
"""Deterministically rehydrate missing SPAN unit payloads without creating labels.

The tracked manifest is the unit authority.  Source artifacts only supply the text
that used to be present in ``units/SPAN/batch_*.json``.  Every source artifact is
hash checked before it is read, and a manifest-bound layout describes historical
batch phase boundaries (the extension intentionally began a new partial batch).
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
BATCH_NAME_RE = re.compile(r"^batch_(\d{4})\.json$")
EXPECTED_UNIT_KEYS = ("unit_id", "source", "window", "text_numbered")


class RehydrationError(ValueError):
    """The supplied artifacts cannot defensibly reproduce the tracked units."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehydrationError(f"cannot read JSON input {path}: {exc}") from exc


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RehydrationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise RehydrationError(f"{path}:{line_number}: row must be a JSON object")
            yield value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        # This matches json.dump(..., ensure_ascii=False) in build_span_units*.py.
        text = json.dumps(value, ensure_ascii=False)
    return text.encode("utf-8")


def _require_sha256(value: Any, context: str) -> str:
    result = str(value)
    if not SHA256_RE.fullmatch(result):
        raise RehydrationError(f"{context}: expected a lowercase 64-hex SHA-256")
    return result


def _require_pinned_revision(value: Any, context: str) -> str:
    result = str(value)
    if not PINNED_REVISION_RE.fullmatch(result):
        raise RehydrationError(
            f"{context}: revision must be an immutable 40-64 character hexadecimal commit/fingerprint"
        )
    return result


@dataclass(frozen=True)
class ManifestUnit:
    unit_id: str
    doc_id: str
    source: str
    window: str
    win_lo: int
    win_hi: int


@dataclass(frozen=True)
class Artifact:
    local_path: Path
    declared_path: str
    repository_path: str
    expected_sha256: str


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    format: str
    artifacts: tuple[Artifact, ...]
    fields: Mapping[str, Any]
    provenance: Mapping[str, Any]


def load_manifest(path: str | Path) -> list[ManifestUnit]:
    source = Path(path)
    rows: list[ManifestUnit] = []
    seen: set[str] = set()
    for row_number, row in enumerate(_iter_jsonl(source), 1):
        required = {"unit_id", "doc_id", "source", "window", "win_lo", "win_hi"}
        missing = sorted(required - set(row))
        if missing:
            raise RehydrationError(f"{source}:{row_number}: missing keys {missing}")
        unit_id = str(row["unit_id"])
        if not unit_id or unit_id in seen:
            raise RehydrationError(f"{source}:{row_number}: empty/duplicate unit_id {unit_id!r}")
        seen.add(unit_id)
        if (
            isinstance(row["win_lo"], bool)
            or isinstance(row["win_hi"], bool)
            or not isinstance(row["win_lo"], int)
            or not isinstance(row["win_hi"], int)
        ):
            raise RehydrationError(f"{source}:{row_number}: window bounds must be integers")
        lo, hi = row["win_lo"], row["win_hi"]
        if lo < 0 or hi <= lo:
            raise RehydrationError(f"{source}:{row_number}: invalid window [{lo}, {hi})")
        rows.append(
            ManifestUnit(
                unit_id=unit_id,
                doc_id=str(row["doc_id"]),
                source=str(row["source"]),
                window=str(row["window"]),
                win_lo=lo,
                win_hi=hi,
            )
        )
        if not rows[-1].doc_id or not rows[-1].source or not rows[-1].window:
            raise RehydrationError(f"{source}:{row_number}: empty document/source/window identity")
    if not rows:
        raise RehydrationError(f"{source}: empty SPAN manifest")
    return rows


def load_batch_names(path: str | Path) -> list[str]:
    source = Path(path)
    value = _load_json(source)
    if not isinstance(value, list) or not value:
        raise RehydrationError(f"{source}: batchpaths must be a non-empty JSON list")
    names = [Path(str(item)).name for item in value]
    if len(names) != len(set(names)):
        raise RehydrationError(f"{source}: duplicate batch names")
    expected = [f"batch_{index:04d}.json" for index in range(len(names))]
    if names != expected:
        raise RehydrationError(
            f"{source}: expected contiguous names {expected[0]}..{expected[-1]}, got "
            f"{names[:2]}..{names[-2:]}"
        )
    return names


def _resolve_input(base: Path, value: Any, context: str) -> tuple[Path, str]:
    declared = str(value)
    if not declared:
        raise RehydrationError(f"{context}: empty path")
    candidate = Path(declared).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RehydrationError(f"{context}: artifact is not a regular file: {candidate}")
    return candidate, declared


def _path_is_under(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _load_bound_receipt(
    base: Path,
    declared_path: Any,
    expected_sha256: Any,
    schema_version: str,
    context: str,
) -> tuple[Path, dict[str, Any]]:
    path, _ = _resolve_input(base, declared_path, context)
    expected = _require_sha256(expected_sha256, f"{context} SHA-256")
    if sha256_file(path) != expected:
        raise RehydrationError(f"{context}: receipt SHA-256 drift")
    value = _load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != schema_version:
        raise RehydrationError(f"{context}: expected schema_version {schema_version}")
    return path, value


def _validate_mdc_raw_builder_derivation(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    raw_source: Mapping[str, Any],
) -> None:
    derivation = receipt.get("derivation")
    if (
        not isinstance(derivation, dict)
        or derivation.get("schema_version") != "span-source-artifact-derivation-v1"
    ):
        raise RehydrationError(
            "mdc_raw_forensic source requires builder derivation; hand-authored receipt rejected"
        )
    builder = derivation.get("builder")
    current_builder = Path(__file__).with_name("build_span_source_receipt.py")
    if (
        not isinstance(builder, dict)
        or builder.get("sha256") != sha256_file(current_builder)
    ):
        raise RehydrationError("mdc_raw_forensic builder SHA-256 differs from this checkout")
    routes = derivation.get("route_choices")
    external = derivation.get("external_forensic_inputs")
    raw_external = external.get("greek_phd") if isinstance(external, dict) else None
    if (
        not isinstance(routes, dict)
        or routes.get("greek_phd") != "mdc_raw_forensic"
        or not isinstance(raw_external, dict)
    ):
        raise RehydrationError("mdc_raw_forensic route derivation is absent")
    quarantine_path, quarantine = _load_bound_receipt(
        receipt_path.parent,
        raw_external.get("quarantine_receipt"),
        raw_external.get("quarantine_receipt_sha256"),
        "mdc_quarantined_object_receipt_v2",
        "MDC quarantine receipt",
    )
    extraction_path, extraction = _load_bound_receipt(
        receipt_path.parent,
        raw_external.get("safe_extraction_receipt"),
        raw_external.get("safe_extraction_receipt_sha256"),
        "mdc_safe_extraction_receipt_v1",
        "MDC safe extraction receipt",
    )
    audit_path, audit = _load_bound_receipt(
        receipt_path.parent,
        raw_external.get("span_coordinate_audit"),
        raw_external.get("span_coordinate_audit_sha256"),
        "mdc_greek_phd_span_coordinate_audit_v2",
        "MDC SPAN audit receipt",
    )
    for field, actual in (
        ("quarantine_receipt_sha256", sha256_file(quarantine_path)),
        ("safe_extraction_receipt_sha256", sha256_file(extraction_path)),
        ("span_coordinate_audit_sha256", sha256_file(audit_path)),
    ):
        if raw_source.get(field) != actual:
            raise RehydrationError(f"mdc_raw_forensic source {field} differs from derivation")
    if (
        quarantine.get("status") != "quarantined_publisher_checksum_mismatch"
        or extraction.get("status") != "passed_fresh_archive_tree_matches"
        or audit.get("status")
        not in {"passed", "comparison_only_with_silver_anomalies"}
        or audit.get("production_eligible") is not False
        or audit.get("research_evidence_scope") != "LLM_silver_comparison_only"
    ):
        raise RehydrationError("mdc_raw_forensic receipt status is not comparison-only safe")
    source_integrity = audit.get("source_coordinate_integrity")
    if (
        not isinstance(source_integrity, dict)
        or source_integrity.get("status") != "passed"
        or not isinstance(source_integrity.get("failure_counts"), dict)
        or any(source_integrity["failure_counts"].values())
    ):
        raise RehydrationError("mdc_raw_forensic source-coordinate audit did not pass")
    safe = quarantine.get("safe_extraction")
    audit_inputs = audit.get("inputs")
    audit_quarantine = (
        audit_inputs.get("quarantine_receipt") if isinstance(audit_inputs, dict) else None
    )
    audit_extraction = (
        audit_inputs.get("safe_extraction_receipt")
        if isinstance(audit_inputs, dict)
        else None
    )
    if (
        not isinstance(safe, dict)
        or safe.get("receipt_sha256") != sha256_file(extraction_path)
        or not isinstance(audit_quarantine, dict)
        or audit_quarantine.get("sha256") != sha256_file(quarantine_path)
        or not isinstance(audit_extraction, dict)
        or audit_extraction.get("sha256") != sha256_file(extraction_path)
    ):
        raise RehydrationError("mdc_raw_forensic quarantine/audit/extraction chain differs")
    extraction_tool = extraction.get("tool")
    audit_tool = audit_inputs.get("tool") if isinstance(audit_inputs, dict) else None
    if (
        not isinstance(extraction_tool, dict)
        or extraction_tool.get("sha256")
        != sha256_file(Path(__file__).with_name("mdc_safe_extract.py"))
        or not isinstance(audit_tool, dict)
        or audit_tool.get("sha256")
        != sha256_file(Path(__file__).with_name("mdc_span_audit.py"))
    ):
        raise RehydrationError("mdc_raw_forensic forensic tool SHA-256 differs")
    selected_digest = _require_sha256(
        audit.get("selected_documents_text_sha256"),
        "MDC selected document text digest",
    )
    if raw_source.get("selected_documents_text_sha256") != selected_digest:
        raise RehydrationError("mdc_raw_forensic selected document text digest differs")
    selection = audit_inputs.get("selection_contract")
    if (
        not isinstance(selection, dict)
        or raw_source.get("selected_document_id_field_counts")
        != selection.get("selected_document_id_field_counts")
        or raw_source.get("selected_text_field_counts")
        != selection.get("selected_text_field_counts")
        or selection.get("document_id_field") != "doc_id"
        or selection.get("text_precedence") != ["text", "document", "content"]
    ):
        raise RehydrationError("mdc_raw_forensic selected field counts differ")
    extraction_payload = extraction.get("extraction")
    if not isinstance(extraction_payload, dict):
        raise RehydrationError("mdc_raw_forensic safe extraction manifest is malformed")
    extraction_manifest_path, _ = _resolve_input(
        receipt_path.parent,
        raw_external.get("extracted_sha256_manifest"),
        "MDC extraction manifest",
    )
    receipt_manifest_path, _ = _resolve_input(
        extraction_path.parent,
        extraction_payload.get("manifest_path"),
        "MDC extraction-receipt manifest",
    )
    declared_manifest_sha = _require_sha256(
        raw_external.get("extracted_sha256_manifest_sha256"),
        "MDC extraction manifest SHA-256",
    )
    receipt_manifest_sha = _require_sha256(
        extraction_payload.get("manifest_sha256"),
        "MDC extraction-receipt manifest SHA-256",
    )
    if (
        extraction_manifest_path != receipt_manifest_path
        or declared_manifest_sha != receipt_manifest_sha
        or sha256_file(extraction_manifest_path) != receipt_manifest_sha
    ):
        raise RehydrationError(
            "mdc_raw_forensic extraction manifest path/hash differs from "
            "the safe extraction receipt"
        )
    manifest_value = _load_json(extraction_manifest_path)
    manifest_rows = (
        manifest_value.get("files") if isinstance(manifest_value, dict) else None
    )
    if (
        not isinstance(manifest_value, dict)
        or manifest_value.get("schema_version")
        != "mdc_safe_extraction_manifest_v1"
        or not isinstance(manifest_rows, list)
        or manifest_value.get("file_count") != extraction_payload.get("file_count")
        or manifest_value.get("directory_count")
        != extraction_payload.get("directory_count")
        or manifest_value.get("total_file_bytes")
        != extraction_payload.get("total_file_bytes")
        or canonical_json_sha256(manifest_value)
        != extraction_payload.get("inventory_sha256")
    ):
        raise RehydrationError("mdc_raw_forensic safe extraction manifest is malformed")
    extraction_root = Path(str(extraction_payload.get("root", ""))).resolve()
    expected_artifacts: list[dict[str, Any]] = []
    prefix = "phd-theses-corpus/contents/"
    for index, row in enumerate(manifest_rows):
        if not isinstance(row, dict):
            raise RehydrationError(
                f"mdc_raw_forensic extraction manifest row {index} is malformed"
            )
        repository_path = str(row.get("path", ""))
        if not (
            repository_path.startswith(prefix)
            and repository_path.endswith(".jsonl.zst")
        ):
            continue
        relative = Path(repository_path)
        expected_sha = _require_sha256(
            row.get("sha256", ""),
            f"mdc_raw_forensic extraction manifest row {index}",
        )
        expected_bytes = row.get("bytes")
        declared_artifact = extraction_root / relative
        resolved = declared_artifact.resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _path_is_under(extraction_root, resolved)
            or declared_artifact.is_symlink()
            or not resolved.is_file()
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise RehydrationError(
                f"mdc_raw_forensic extraction manifest row {index} is unsafe"
            )
        expected_artifacts.append(
            {
                "path": str(resolved),
                "repository_path": repository_path,
                "sha256": expected_sha,
                "bytes": expected_bytes,
                "acquisition_hash_kind": "quarantined_extracted_sha256",
            }
        )
    expected_artifacts.sort(key=lambda row: str(row["repository_path"]))
    if not expected_artifacts or raw_source.get("artifacts") != expected_artifacts:
        raise RehydrationError(
            "mdc_raw_forensic source artifacts differ from the safe extraction manifest"
        )
    if (
        raw_source.get("repo_type") != "archive"
        or raw_source.get("repo_id")
        != "mozilla-data-collective/cmkwvpu7s0032mo07jpk20pj1"
        or raw_source.get("selection_globs")
        != ["phd-theses-corpus/contents/*.jsonl.zst"]
        or raw_source.get("fields")
        != {
            "document_id": "doc_id",
            "text_precedence": ["text", "document", "content"],
        }
        or raw_source.get("publisher_checksum_status") != "mismatch_quarantined"
    ):
        raise RehydrationError("mdc_raw_forensic source identity contract differs")
    archive = quarantine.get("archive")
    extraction_archive = extraction.get("archive")
    if (
        not isinstance(archive, dict)
        or not isinstance(extraction_archive, dict)
        or archive.get("observed_sha256") != raw_source.get("revision")
        or archive.get("publisher_declared_sha256") == raw_source.get("revision")
        or extraction_archive.get("sha256") != raw_source.get("revision")
        or raw_external.get("archive_sha256") != raw_source.get("revision")
        or raw_source.get("publisher_checksum_status") != "mismatch_quarantined"
    ):
        raise RehydrationError("mdc_raw_forensic archive identity chain differs")


def load_source_specs(
    path: str | Path, expected_sources: set[str]
) -> tuple[dict[str, SourceSpec], list[dict[str, Any]]]:
    receipt_path = Path(path).resolve()
    value = _load_json(receipt_path)
    if not isinstance(value, dict) or value.get("schema_version") != "span-source-artifacts-v1":
        raise RehydrationError(f"{receipt_path}: expected schema_version span-source-artifacts-v1")
    sources = value.get("sources")
    observed_sources = set(sources) if isinstance(sources, dict) else set()
    if not isinstance(sources, dict) or observed_sources != expected_sources:
        raise RehydrationError(
            f"{receipt_path}: source inventory mismatch: "
            f"missing={sorted(expected_sources-observed_sources)}, "
            f"extra={sorted(observed_sources-expected_sources)}"
        )
    greek_raw = sources.get("greek_phd")
    if isinstance(greek_raw, dict) and (
        greek_raw.get("format") == "jsonl_documents"
        or greek_raw.get("acquisition_source_id") == "mdc_raw_forensic"
        or greek_raw.get("repo_type") == "archive"
    ):
        if (
            greek_raw.get("format") != "jsonl_documents"
            or greek_raw.get("repo_type") != "archive"
            or greek_raw.get("acquisition_source_id") != "mdc_raw_forensic"
        ):
            raise RehydrationError(
                "Greek PhD JSONL/archive source must use exact mdc_raw_forensic provenance"
            )
        _validate_mdc_raw_builder_derivation(receipt_path, value, greek_raw)
    result: dict[str, SourceSpec] = {}
    all_paths: set[Path] = set()
    all_repository_artifacts: set[tuple[str, str, str]] = set()
    artifact_receipts: list[dict[str, Any]] = []
    for name in sorted(expected_sources):
        raw = sources[name]
        if not isinstance(raw, dict):
            raise RehydrationError(f"{receipt_path}: source {name} must be an object")
        repo_id = str(raw.get("repo_id", ""))
        repo_type = str(raw.get("repo_type", ""))
        if not repo_id or repo_type not in {"dataset", "git", "archive"}:
            raise RehydrationError(
                f"{receipt_path}: source {name} needs repo_id and repo_type=dataset|git|archive"
            )
        revision = _require_pinned_revision(raw.get("revision", ""), f"source {name}")
        source_format = str(raw.get("format", ""))
        allowed_formats = {
            "greek_phd": {"jsonl_documents", "parquet_documents"},
            "openarchives": {"jsonl_documents", "parquet_documents"},
            "kallipos": {"parquet_sections", "parquet_documents"},
        }.get(name, set())
        if source_format not in allowed_formats:
            raise RehydrationError(
                f"source {name}: format {source_format!r} is not one of {sorted(allowed_formats)}"
            )
        fields = raw.get("fields")
        if not isinstance(fields, dict):
            raise RehydrationError(f"source {name}: fields must be an object")
        if source_format in {"jsonl_documents", "parquet_documents"}:
            document_id = fields.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise RehydrationError(f"source {name}: document_id must be a non-empty column name")
            text_fields = fields.get("text_precedence")
            if not isinstance(text_fields, list) or not text_fields or not all(
                isinstance(item, str) and item for item in text_fields
            ) or len(text_fields) != len(set(text_fields)):
                raise RehydrationError(
                    f"source {name}: text_precedence must be a non-empty list of unique columns"
                )
            if source_format == "jsonl_documents" and (
                document_id != "doc_id"
                or text_fields != ["text", "document", "content"]
            ):
                raise RehydrationError(
                    f"source {name}: JSONL fields must retain historical doc_id and "
                    "['text', 'document', 'content'] precedence"
                )
            row_filter = fields.get("row_filter")
            if row_filter is not None and (
                not isinstance(row_filter, dict)
                or set(row_filter) != {"column", "equals"}
                or not isinstance(row_filter.get("column"), str)
                or not row_filter.get("column")
            ):
                raise RehydrationError(
                    f"source {name}: row_filter must contain exactly column and equals"
                )
        else:
            required_fields = {"filename": "filename", "order": "id", "section": "section"}
            if fields != required_fields:
                raise RehydrationError(
                    f"source {name}: fields must exactly match {required_fields}"
                )
        raw_artifacts = raw.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise RehydrationError(f"source {name}: artifacts must be a non-empty list")
        artifacts: list[Artifact] = []
        for index, item in enumerate(raw_artifacts):
            if not isinstance(item, dict):
                raise RehydrationError(f"source {name} artifact {index}: expected object")
            local, declared = _resolve_input(
                receipt_path.parent, item.get("path", ""), f"source {name} artifact {index}"
            )
            if local in all_paths:
                raise RehydrationError(f"artifact path appears more than once: {local}")
            all_paths.add(local)
            repository_path = str(item.get("repository_path", ""))
            if not repository_path or repository_path.startswith("/") or ".." in Path(repository_path).parts:
                raise RehydrationError(
                    f"source {name} artifact {index}: repository_path must be a relative pinned path"
                )
            repository_identity = (repo_id, revision, repository_path)
            if repository_identity in all_repository_artifacts:
                raise RehydrationError(
                    f"source {name} artifact {index}: duplicate repository artifact {repository_path}"
                )
            all_repository_artifacts.add(repository_identity)
            expected_sha = _require_sha256(
                item.get("sha256", ""), f"source {name} artifact {index}"
            )
            suffix_ok = (
                local.name.endswith((".jsonl", ".jsonl.zst"))
                if source_format == "jsonl_documents"
                else local.name.endswith(".parquet")
            )
            if not suffix_ok:
                raise RehydrationError(
                    f"source {name} artifact {index}: file extension does not match {source_format}"
                )
            actual_sha = sha256_file(local)
            if actual_sha != expected_sha:
                raise RehydrationError(
                    f"source {name} artifact {index}: SHA-256 mismatch: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            artifacts.append(Artifact(local, declared, repository_path, expected_sha))
            artifact_receipts.append(
                {
                    "source": name,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "repository_path": repository_path,
                    "declared_local_path": declared,
                    "resolved_local_path": str(local),
                    "bytes": local.stat().st_size,
                    "sha256": actual_sha,
                }
            )
        result[name] = SourceSpec(
            name=name,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            format=source_format,
            artifacts=tuple(artifacts),
            fields=fields,
            provenance={
                "acquisition_source_id": raw.get("acquisition_source_id"),
                "selection_globs": raw.get("selection_globs"),
                "historical_source_relation": raw.get("historical_source_relation"),
                "label_text_equivalence": raw.get("label_text_equivalence"),
                "document_id_alignment": raw.get("document_id_alignment"),
                "quarantine_receipt_sha256": raw.get("quarantine_receipt_sha256"),
                "safe_extraction_receipt_sha256": raw.get(
                    "safe_extraction_receipt_sha256"
                ),
                "span_coordinate_audit_sha256": raw.get("span_coordinate_audit_sha256"),
                "selected_documents_text_sha256": raw.get(
                    "selected_documents_text_sha256"
                ),
                "selected_document_id_field_counts": raw.get(
                    "selected_document_id_field_counts"
                ),
                "selected_text_field_counts": raw.get("selected_text_field_counts"),
                "publisher_checksum_status": raw.get("publisher_checksum_status"),
            },
        )
    return result, artifact_receipts


def _load_layout(
    path: str | Path,
    manifest_path: Path,
    batchpaths_path: Path,
    rows: Sequence[ManifestUnit],
    batch_names: Sequence[str],
) -> tuple[list[list[ManifestUnit]], dict[str, Any]]:
    layout_path = Path(path).resolve()
    value = _load_json(layout_path)
    if not isinstance(value, dict) or value.get("schema_version") != "span-rehydration-layout-v1":
        raise RehydrationError(f"{layout_path}: expected schema_version span-rehydration-layout-v1")
    manifest_sha = sha256_file(manifest_path)
    batchpaths_sha = sha256_file(batchpaths_path)
    if value.get("manifest_sha256") != manifest_sha:
        raise RehydrationError(f"{layout_path}: manifest SHA-256 does not match {manifest_path}")
    if value.get("batchpaths_sha256") != batchpaths_sha:
        raise RehydrationError(f"{layout_path}: batchpaths SHA-256 does not match {batchpaths_path}")
    batch_size = value.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RehydrationError(f"{layout_path}: batch_size must be a positive integer")
    builders = value.get("builders")
    if not isinstance(builders, list):
        raise RehydrationError(f"{layout_path}: builders must be a list")
    builder_receipts: list[dict[str, Any]] = []
    builder_names: set[str] = set()
    for index, builder in enumerate(builders):
        if not isinstance(builder, dict):
            raise RehydrationError(f"{layout_path}: builder {index} must be an object")
        name = str(builder.get("name", ""))
        if not name or name in builder_names:
            raise RehydrationError(f"{layout_path}: empty/duplicate builder name {name!r}")
        builder_names.add(name)
        builder_path, declared = _resolve_input(
            layout_path.parent, builder.get("path", ""), f"layout builder {name}"
        )
        expected_sha = _require_sha256(builder.get("sha256", ""), f"layout builder {name}")
        actual_sha = sha256_file(builder_path)
        if actual_sha != expected_sha:
            raise RehydrationError(
                f"layout builder {name}: expected SHA-256 {expected_sha}, got {actual_sha}"
            )
        builder_receipts.append(
            {
                "name": name,
                "declared_path": declared,
                "resolved_path": str(builder_path),
                "bytes": builder_path.stat().st_size,
                "sha256": actual_sha,
            }
        )
    phases = value.get("phases")
    if not isinstance(phases, list) or not phases:
        raise RehydrationError(f"{layout_path}: phases must be a non-empty list")
    unit_index = {row.unit_id: index for index, row in enumerate(rows)}
    starts: list[int] = []
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise RehydrationError(f"{layout_path}: phase {index} must be an object")
        start_id = str(phase.get("start_unit_id", ""))
        if start_id not in unit_index:
            raise RehydrationError(f"{layout_path}: phase {index} start unit {start_id!r} is absent")
        if str(phase.get("builder", "")) not in builder_names and builder_names:
            raise RehydrationError(f"{layout_path}: phase {index} references an unknown builder")
        starts.append(unit_index[start_id])
    if starts[0] != 0 or starts != sorted(set(starts)):
        raise RehydrationError(f"{layout_path}: phases must be unique, ordered, and begin at unit 0")
    batches: list[list[ManifestUnit]] = []
    phase_receipts: list[dict[str, Any]] = []
    for phase_index, (phase, lo) in enumerate(zip(phases, starts)):
        hi = starts[phase_index + 1] if phase_index + 1 < len(starts) else len(rows)
        phase_rows = list(rows[lo:hi])
        expected_count = phase.get("expected_unit_count")
        if expected_count != len(phase_rows):
            raise RehydrationError(
                f"{layout_path}: phase {phase_index} expected {expected_count} units, "
                f"manifest supplies {len(phase_rows)}"
            )
        unit_digest = canonical_json_sha256([row.unit_id for row in phase_rows])
        if phase.get("expected_unit_ids_sha256") != unit_digest:
            raise RehydrationError(f"{layout_path}: phase {phase_index} unit identity digest drift")
        phase_batches = [
            phase_rows[start : start + batch_size]
            for start in range(0, len(phase_rows), batch_size)
        ]
        if phase.get("expected_batch_count") != len(phase_batches):
            raise RehydrationError(
                f"{layout_path}: phase {phase_index} expected {phase.get('expected_batch_count')} "
                f"batches, got {len(phase_batches)}"
            )
        batches.extend(phase_batches)
        phase_receipts.append(
            {
                "name": str(phase.get("name", f"phase-{phase_index}")),
                "builder": str(phase.get("builder", "")),
                "start_unit_id": phase_rows[0].unit_id,
                "end_unit_id": phase_rows[-1].unit_id,
                "unit_count": len(phase_rows),
                "unit_ids_sha256": unit_digest,
                "batch_count": len(phase_batches),
            }
        )
    if len(batches) != len(batch_names):
        raise RehydrationError(
            f"layout creates {len(batches)} batches but batchpaths names {len(batch_names)}"
        )
    flattened = [row.unit_id for batch in batches for row in batch]
    if flattened != [row.unit_id for row in rows]:
        raise RehydrationError("layout dropped, duplicated, or reordered manifest units")
    return batches, {
        "path": str(layout_path),
        "bytes": layout_path.stat().st_size,
        "sha256": sha256_file(layout_path),
        "batch_size": batch_size,
        "builders": builder_receipts,
        "phases": phase_receipts,
    }


@contextlib.contextmanager
def _open_jsonl_text(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as handle:
            yield handle
        return
    if not path.name.endswith(".jsonl.zst"):
        raise RehydrationError(f"unsupported JSONL artifact extension: {path}")
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        executable = shutil.which("zstd")
        if not executable:
            raise RehydrationError(
                "reading .jsonl.zst requires the pinned zstandard package or a zstd executable"
            )
        process = subprocess.Popen(
            [executable, "-q", "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        text = io.TextIOWrapper(process.stdout, encoding="utf-8")
        consumer_failed = False
        try:
            yield text
        except BaseException:
            consumer_failed = True
            raise
        finally:
            text.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            if process.stderr:
                process.stderr.close()
            returncode = process.wait()
            if returncode and not consumer_failed:
                raise RehydrationError(f"zstd failed for {path} ({returncode}): {stderr.strip()}")
        return
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
            with io.TextIOWrapper(decompressed, encoding="utf-8") as text:
                yield text


def _pick_text(row: Mapping[str, Any], fields: Sequence[str], context: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            return value
    raise RehydrationError(f"{context}: none of the pinned text fields {list(fields)} is a string")


def _unit_payloads_for_document(
    source: str,
    doc_id: str,
    text: str,
    manifest_units: Sequence[ManifestUnit],
) -> dict[str, dict[str, Any]]:
    lines = text.split("\n")
    result: dict[str, dict[str, Any]] = {}
    for meta in manifest_units:
        if meta.source != source or meta.doc_id != doc_id:
            raise RehydrationError(f"internal manifest/document identity mismatch for {meta.unit_id}")
        if meta.win_hi > len(lines):
            raise RehydrationError(
                f"{source}/{doc_id}/{meta.unit_id}: win_hi={meta.win_hi} exceeds "
                f"rehydrated physical line count {len(lines)}"
            )
        segment = [
            (index, lines[index])
            for index in range(meta.win_lo, meta.win_hi)
            if lines[index].strip()
        ]
        if not segment:
            raise RehydrationError(f"{source}/{doc_id}/{meta.unit_id}: window has no nonempty lines")
        numbered = "\n".join(f"L{index:05d}: {line}" for index, line in segment)
        result[meta.unit_id] = {
            "unit_id": meta.unit_id,
            "source": meta.source,
            "window": meta.window,
            "text_numbered": numbered,
        }
    return result


def _scan_jsonl_source(
    spec: SourceSpec,
    manifests_by_doc: Mapping[str, Sequence[ManifestUnit]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    required = set(manifests_by_doc)
    found: set[str] = set()
    units: dict[str, dict[str, Any]] = {}
    document_receipts: list[tuple[str, str]] = []
    rows_scanned = 0
    selected_by_artifact: dict[str, int] = {}
    selected_id_field_counts: collections.Counter[str] = collections.Counter()
    selected_text_field_counts: collections.Counter[str] = collections.Counter()
    document_id_field = str(spec.fields["document_id"])
    text_fields = tuple(str(item) for item in spec.fields["text_precedence"])
    for artifact in spec.artifacts:
        selected = 0
        with _open_jsonl_text(artifact.local_path) as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                rows_scanned += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RehydrationError(
                        f"{artifact.local_path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RehydrationError(
                        f"{artifact.local_path}:{line_number}: row must be a JSON object"
                    )
                doc_id = row.get(document_id_field)
                if not isinstance(doc_id, str) or doc_id not in required:
                    continue
                if doc_id in found:
                    raise RehydrationError(
                        f"{spec.name}/{doc_id}: requested document occurs more than once in supplied artifacts"
                    )
                text = _pick_text(
                    row, text_fields, f"{artifact.local_path}:{line_number}:{doc_id}"
                )
                text_field = next(
                    field for field in text_fields if isinstance(row.get(field), str)
                )
                found.add(doc_id)
                selected += 1
                selected_id_field_counts[document_id_field] += 1
                selected_text_field_counts[text_field] += 1
                document_receipts.append((doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest()))
                for unit_id, unit in _unit_payloads_for_document(
                    spec.name, doc_id, text, manifests_by_doc[doc_id]
                ).items():
                    if unit_id in units:
                        raise RehydrationError(f"duplicate reconstructed unit {unit_id}")
                    units[unit_id] = unit
        selected_by_artifact[artifact.repository_path] = selected
    missing = sorted(required - found)
    extra = sorted(found - required)
    if missing or extra:
        raise RehydrationError(
            f"{spec.name}: selected document inventory mismatch: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    selected_text_sha256 = canonical_json_sha256(sorted(document_receipts))
    if spec.provenance.get("acquisition_source_id") == "mdc_raw_forensic":
        if (
            selected_text_sha256
            != spec.provenance.get("selected_documents_text_sha256")
        ):
            raise RehydrationError(
                "greek_phd: selected document text digest differs from the forensic audit"
            )
        if (
            dict(sorted(selected_id_field_counts.items()))
            != spec.provenance.get("selected_document_id_field_counts")
            or dict(sorted(selected_text_field_counts.items()))
            != spec.provenance.get("selected_text_field_counts")
        ):
            raise RehydrationError(
                "greek_phd: selected ID/text field counts differ from the forensic audit"
            )
    return units, {
        "source": spec.name,
        "required_document_count": len(required),
        "required_document_ids_sha256": canonical_json_sha256(sorted(required)),
        "selected_document_count": len(found),
        "selected_documents_text_sha256": selected_text_sha256,
        "selected_document_id_field_counts": dict(
            sorted(selected_id_field_counts.items())
        ),
        "selected_text_field_counts": dict(sorted(selected_text_field_counts.items())),
        "rows_scanned": rows_scanned,
        "selected_documents_by_artifact": selected_by_artifact,
        "ignored_non_target_rows": rows_scanned - len(found),
    }


def _row_passes_filter(row: Mapping[str, Any], row_filter: Any) -> bool:
    if row_filter is None:
        return True
    return row.get(str(row_filter["column"])) == row_filter["equals"]


def _scan_parquet_document_source(
    spec: SourceSpec,
    manifests_by_doc: Mapping[str, Sequence[ManifestUnit]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RehydrationError(
            "document-Parquet rehydration requires the pinned pyarrow runtime"
        ) from exc
    required = set(manifests_by_doc)
    found: set[str] = set()
    units: dict[str, dict[str, Any]] = {}
    document_receipts: list[tuple[str, str]] = []
    rows_scanned = 0
    selected_by_artifact: dict[str, int] = {}
    document_id_field = str(spec.fields["document_id"])
    text_fields = tuple(str(item) for item in spec.fields["text_precedence"])
    row_filter = spec.fields.get("row_filter")
    for artifact in spec.artifacts:
        parquet = pq.ParquetFile(artifact.local_path)
        columns = set(parquet.schema_arrow.names)
        if document_id_field not in columns:
            raise RehydrationError(
                f"{artifact.local_path}: missing document-id column {document_id_field!r}"
            )
        present_text = [field for field in text_fields if field in columns]
        if not present_text:
            raise RehydrationError(
                f"{artifact.local_path}: none of the text precedence columns are present: "
                f"{list(text_fields)}"
            )
        selected_columns = [document_id_field, *present_text]
        if row_filter is not None:
            filter_column = str(row_filter["column"])
            if filter_column not in columns:
                raise RehydrationError(
                    f"{artifact.local_path}: missing row-filter column {filter_column!r}"
                )
            if filter_column not in selected_columns:
                selected_columns.append(filter_column)
        selected = 0
        for batch in parquet.iter_batches(
            batch_size=8192, columns=selected_columns, use_threads=False
        ):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                rows_scanned += 1
                row = {column: column_values[index] for column, column_values in values.items()}
                if not _row_passes_filter(row, row_filter):
                    continue
                doc_id = row.get(document_id_field)
                if not isinstance(doc_id, str) or doc_id not in required:
                    continue
                if doc_id in found:
                    raise RehydrationError(
                        f"{spec.name}/{doc_id}: requested document occurs more than once in supplied Parquet"
                    )
                text = _pick_text(
                    row,
                    text_fields,
                    f"{artifact.local_path}:row={rows_scanned}:{doc_id}",
                )
                found.add(doc_id)
                selected += 1
                document_receipts.append(
                    (doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest())
                )
                for unit_id, unit in _unit_payloads_for_document(
                    spec.name, doc_id, text, manifests_by_doc[doc_id]
                ).items():
                    if unit_id in units:
                        raise RehydrationError(f"duplicate reconstructed unit {unit_id}")
                    units[unit_id] = unit
        selected_by_artifact[artifact.repository_path] = selected
    missing = sorted(required - found)
    if missing:
        raise RehydrationError(
            f"{spec.name}: selected document inventory mismatch: missing={missing[:20]}, extra=[]"
        )
    return units, {
        "source": spec.name,
        "format": "parquet_documents",
        "document_id_column": document_id_field,
        "text_precedence": list(text_fields),
        "row_filter": row_filter,
        "required_document_count": len(required),
        "required_document_ids_sha256": canonical_json_sha256(sorted(required)),
        "selected_document_count": len(found),
        "selected_documents_text_sha256": canonical_json_sha256(sorted(document_receipts)),
        "rows_scanned": rows_scanned,
        "selected_documents_by_artifact": selected_by_artifact,
        "ignored_non_target_rows": rows_scanned - len(found),
    }


def _id_sort_category(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        raise RehydrationError(f"Kallipos section id must be a non-null string or number, got {value!r}")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RehydrationError(f"Kallipos section id must be finite, got {value!r}")
        return "number"
    if isinstance(value, str):
        return "string"
    raise RehydrationError(f"unsupported Kallipos section id type {type(value).__name__}")


def assemble_kallipos_document(rows: Iterable[tuple[Any, Any]], doc_id: str) -> str:
    """Order one Kallipos work by ``id`` and join sections exactly as the builder did."""
    materialized = list(rows)
    if not materialized:
        raise RehydrationError(f"kallipos/{doc_id}: no section rows")
    categories = {_id_sort_category(order) for order, _ in materialized}
    if len(categories) != 1:
        raise RehydrationError(f"kallipos/{doc_id}: mixed section id types are ambiguous")
    order_values = [order for order, _ in materialized]
    if len(order_values) != len(set(order_values)):
        raise RehydrationError(f"kallipos/{doc_id}: duplicate section id")
    materialized.sort(key=lambda item: item[0])
    sections: list[str] = []
    for _, section in materialized:
        if section is None:
            sections.append("")
        elif isinstance(section, str):
            sections.append(section)
        else:
            raise RehydrationError(
                f"kallipos/{doc_id}: section must be a string or null, got {type(section).__name__}"
            )
    return "\n\n".join(sections)


def _scan_kallipos_source(
    spec: SourceSpec,
    manifests_by_doc: Mapping[str, Sequence[ManifestUnit]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RehydrationError(
            "Kallipos rehydration requires the pinned pyarrow runtime"
        ) from exc
    required = set(manifests_by_doc)
    sections: dict[str, list[tuple[Any, Any]]] = {doc_id: [] for doc_id in required}
    filename_identities: dict[str, tuple[str, str]] = {}
    rows_scanned = 0
    selected_rows_by_artifact: dict[str, int] = {}
    for artifact in spec.artifacts:
        parquet = pq.ParquetFile(artifact.local_path)
        required_columns = ["filename", "id", "section"]
        missing_columns = sorted(set(required_columns) - set(parquet.schema_arrow.names))
        if missing_columns:
            raise RehydrationError(
                f"{artifact.local_path}: missing Kallipos columns {missing_columns}"
            )
        selected = 0
        for batch in parquet.iter_batches(batch_size=65536, columns=required_columns):
            values = batch.to_pydict()
            for filename, order, section in zip(
                values["filename"], values["id"], values["section"]
            ):
                rows_scanned += 1
                if filename is None:
                    continue
                doc_id = str(filename)
                if doc_id in required:
                    identity = (type(filename).__name__, repr(filename))
                    previous_identity = filename_identities.setdefault(doc_id, identity)
                    if previous_identity != identity:
                        raise RehydrationError(
                            f"kallipos/{doc_id}: distinct filename values collapse to the same string"
                        )
                    sections[doc_id].append((order, section))
                    selected += 1
        selected_rows_by_artifact[artifact.repository_path] = selected
    missing = sorted(doc_id for doc_id, rows in sections.items() if not rows)
    if missing:
        raise RehydrationError(f"kallipos: missing requested documents {missing[:20]}")
    units: dict[str, dict[str, Any]] = {}
    document_receipts: list[tuple[str, str]] = []
    for doc_id in sorted(required):
        text = assemble_kallipos_document(sections[doc_id], doc_id)
        document_receipts.append((doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest()))
        units.update(
            _unit_payloads_for_document(spec.name, doc_id, text, manifests_by_doc[doc_id])
        )
    return units, {
        "source": spec.name,
        "required_document_count": len(required),
        "required_document_ids_sha256": canonical_json_sha256(sorted(required)),
        "selected_document_count": len(required),
        "selected_documents_text_sha256": canonical_json_sha256(document_receipts),
        "rows_scanned": rows_scanned,
        "selected_section_rows_by_artifact": selected_rows_by_artifact,
        "ignored_non_target_rows": rows_scanned - sum(len(rows) for rows in sections.values()),
        "ordering": "filename grouped; section rows sorted by id; duplicate ids refused",
    }


def _validate_unit_shape(unit: Any, meta: ManifestUnit, context: str) -> None:
    if not isinstance(unit, dict) or tuple(unit) != EXPECTED_UNIT_KEYS:
        raise RehydrationError(
            f"{context}: expected ordered keys {EXPECTED_UNIT_KEYS}, got "
            f"{tuple(unit) if isinstance(unit, dict) else type(unit).__name__}"
        )
    expected = (meta.unit_id, meta.source, meta.window)
    observed = (unit["unit_id"], unit["source"], unit["window"])
    if observed != expected or not isinstance(unit["text_numbered"], str):
        raise RehydrationError(f"{context}: unit identity/shape does not match manifest")


def _batch_receipt(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise RehydrationError(f"{path}: batch root must be a list")
    ids: list[str] = []
    for index, unit in enumerate(value):
        if not isinstance(unit, dict) or tuple(unit) != EXPECTED_UNIT_KEYS:
            raise RehydrationError(f"{path}:{index}: invalid SPAN unit payload shape")
        unit_id = unit.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise RehydrationError(f"{path}:{index}: empty unit_id")
        ids.append(unit_id)
    if len(ids) != len(set(ids)):
        raise RehydrationError(f"{path}: duplicate unit ids")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "unit_count": len(ids),
        "unit_ids_sha256": canonical_json_sha256(ids),
    }


def snapshot_artifact_sha256(
    manifest_sha256: str, batchpaths_sha256: str, batches: Sequence[Mapping[str, Any]]
) -> str:
    return canonical_json_sha256(
        {
            "schema_version": "span-unit-snapshot-artifact-v1",
            "manifest_sha256": manifest_sha256,
            "batchpaths_sha256": batchpaths_sha256,
            "batches": list(batches),
        }
    )


def _read_snapshot(
    unit_dir: Path,
    manifest_path: Path,
    batchpaths_path: Path,
    layout_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    rows = load_manifest(manifest_path)
    names = load_batch_names(batchpaths_path)
    expected_batches, _ = _load_layout(
        layout_path, manifest_path, batchpaths_path, rows, names
    )
    actual_paths = sorted(unit_dir.glob("batch_*.json"))
    actual_names = [path.name for path in actual_paths]
    if actual_names != names:
        raise RehydrationError(
            f"{unit_dir}: batch inventory mismatch: missing={sorted(set(names)-set(actual_names))[:20]}, "
            f"extra={sorted(set(actual_names)-set(names))[:20]}"
        )
    manifest_by_id = {row.unit_id: row for row in rows}
    units: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for path, expected_rows in zip(actual_paths, expected_batches):
        value = _load_json(path)
        if not isinstance(value, list) or len(value) != len(expected_rows):
            raise RehydrationError(
                f"{path}: expected {len(expected_rows)} units, got "
                f"{len(value) if isinstance(value, list) else 'non-list'}"
            )
        expected_ids = [row.unit_id for row in expected_rows]
        observed_ids = [unit.get("unit_id") if isinstance(unit, dict) else None for unit in value]
        if observed_ids != expected_ids:
            raise RehydrationError(f"{path}: unit membership/order differs from layout")
        for unit in value:
            unit_id = str(unit["unit_id"])
            _validate_unit_shape(unit, manifest_by_id[unit_id], f"{path}:{unit_id}")
            if unit_id in units:
                raise RehydrationError(f"{path}: duplicate snapshot unit {unit_id}")
            units[unit_id] = unit
        receipts.append(_batch_receipt(path))
    if set(units) != set(manifest_by_id):
        raise RehydrationError(f"{unit_dir}: snapshot does not exactly cover the manifest")
    artifact_sha = snapshot_artifact_sha256(
        sha256_file(manifest_path), sha256_file(batchpaths_path), receipts
    )
    return receipts, units, artifact_sha


def compare_snapshots(
    generated_units: Mapping[str, Mapping[str, Any]],
    generated_receipts: Sequence[Mapping[str, Any]],
    generated_artifact_sha256: str,
    reference_dir: Path,
    manifest_path: Path,
    batchpaths_path: Path,
    layout_path: Path,
) -> dict[str, Any]:
    reference_receipts, reference_units, reference_sha = _read_snapshot(
        reference_dir, manifest_path, batchpaths_path, layout_path
    )
    generated_files = {row["name"]: row["sha256"] for row in generated_receipts}
    reference_files = {row["name"]: row["sha256"] for row in reference_receipts}
    mismatched_files = sorted(
        name for name in generated_files if generated_files[name] != reference_files[name]
    )
    mismatched_units = sorted(
        unit_id
        for unit_id in generated_units
        if generated_units[unit_id] != reference_units[unit_id]
    )
    return {
        "schema_version": "span-snapshot-comparison-v1",
        "labels_compared_or_inferred": False,
        "reference_directory": str(reference_dir.resolve()),
        "reference_snapshot_artifact_sha256": reference_sha,
        "generated_snapshot_artifact_sha256": generated_artifact_sha256,
        "artifact_sha256_equal": reference_sha == generated_artifact_sha256,
        "byte_identical_batch_count": len(generated_files) - len(mismatched_files),
        "mismatched_batch_count": len(mismatched_files),
        "mismatched_batch_names_sha256": canonical_json_sha256(mismatched_files),
        "mismatched_batch_names_first_20": mismatched_files[:20],
        "unit_payload_match_count": len(generated_units) - len(mismatched_units),
        "mismatched_unit_count": len(mismatched_units),
        "mismatched_unit_ids_sha256": canonical_json_sha256(mismatched_units),
        "mismatched_unit_ids_first_20": mismatched_units[:20],
    }


def rehydrate_span_units(
    *,
    manifest_path: str | Path,
    batchpaths_path: str | Path,
    layout_path: str | Path,
    source_receipt_path: str | Path,
    output_dir: str | Path,
    receipt_path: str | Path,
    expected_artifact_sha256: str | None = None,
    reference_unit_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    batchpaths_path = Path(batchpaths_path).resolve()
    layout_path = Path(layout_path).resolve()
    source_receipt_path = Path(source_receipt_path).resolve()
    output_dir = Path(output_dir).resolve()
    receipt_path = Path(receipt_path).resolve()
    if output_dir.exists() or receipt_path.exists():
        raise FileExistsError(
            f"refusing immutable output overwrite: output_dir={output_dir.exists()}, "
            f"receipt={receipt_path.exists()}"
        )
    if expected_artifact_sha256 is not None:
        expected_artifact_sha256 = _require_sha256(
            expected_artifact_sha256, "expected artifact SHA-256"
        )
    rows = load_manifest(manifest_path)
    batch_names = load_batch_names(batchpaths_path)
    batches, layout_receipt = _load_layout(
        layout_path, manifest_path, batchpaths_path, rows, batch_names
    )
    sources = {row.source for row in rows}
    specs, artifact_receipts = load_source_specs(source_receipt_path, sources)
    manifests_by_source: dict[str, dict[str, list[ManifestUnit]]] = {}
    for row in rows:
        manifests_by_source.setdefault(row.source, {}).setdefault(row.doc_id, []).append(row)
    units: dict[str, dict[str, Any]] = {}
    extraction_receipts: list[dict[str, Any]] = []
    for source in sorted(sources):
        spec = specs[source]
        if spec.format == "jsonl_documents":
            source_units, extraction = _scan_jsonl_source(
                spec, manifests_by_source[source]
            )
        elif spec.format == "parquet_documents":
            source_units, extraction = _scan_parquet_document_source(
                spec, manifests_by_source[source]
            )
        else:
            source_units, extraction = _scan_kallipos_source(
                spec, manifests_by_source[source]
            )
        overlap = sorted(set(units) & set(source_units))
        if overlap:
            raise RehydrationError(f"duplicate units across source scans: {overlap[:20]}")
        units.update(source_units)
        extraction_receipts.append(extraction)
    expected_unit_ids = [row.unit_id for row in rows]
    if set(units) != set(expected_unit_ids):
        raise RehydrationError(
            f"rehydrated unit inventory mismatch: missing={sorted(set(expected_unit_ids)-set(units))[:20]}, "
            f"extra={sorted(set(units)-set(expected_unit_ids))[:20]}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".partial", dir=output_dir.parent)
    )
    batch_receipts: list[dict[str, Any]] = []
    try:
        for name, batch in zip(batch_names, batches):
            payloads = [units[row.unit_id] for row in batch]
            for payload, meta in zip(payloads, batch):
                _validate_unit_shape(payload, meta, f"generated:{name}:{meta.unit_id}")
            path = temporary / name
            path.write_bytes(_json_bytes(payloads))
            batch_receipts.append(_batch_receipt(path))
        artifact_sha = snapshot_artifact_sha256(
            sha256_file(manifest_path), sha256_file(batchpaths_path), batch_receipts
        )
        if expected_artifact_sha256 is not None and artifact_sha != expected_artifact_sha256:
            raise RehydrationError(
                "rehydrated snapshot artifact SHA-256 mismatch: "
                f"expected {expected_artifact_sha256}, got {artifact_sha}"
            )
        comparison = None
        if reference_unit_dir is not None:
            comparison = compare_snapshots(
                units,
                batch_receipts,
                artifact_sha,
                Path(reference_unit_dir),
                manifest_path,
                batchpaths_path,
                layout_path,
            )
        equivalence_verified = expected_artifact_sha256 is not None
        status = (
            "verified_expected_artifact_sha256"
            if equivalence_verified
            else "rehydrated_unverified_snapshot"
        )
        receipt: dict[str, Any] = {
            "schema_version": "span-unit-rehydration-receipt-v1",
            "operation": "text_payload_rehydration_only",
            "execution": {"cpu_only": True, "accelerator": "none"},
            "labels_read_created_or_inferred": False,
            "snapshot_equivalence_status": status,
            "snapshot_equivalence_verified": equivalence_verified,
            "expected_snapshot_artifact_sha256": expected_artifact_sha256,
            "snapshot_artifact_sha256": artifact_sha,
            "snapshot_artifact_hash_definition": "canonical JSON span-unit-snapshot-artifact-v1 over manifest SHA, batchpaths SHA, and ordered per-batch byte/unit receipts",
            "research_fit_eligible": True,
            "research_evidence_scope": "LLM_silver_comparison_only",
            "promotion_eligible": False,
            "promotion_ineligibility_reasons": [
                "LLM_silver_not_human_gold",
                *([] if equivalence_verified else ["rehydrated_unverified_snapshot"]),
            ],
            "inputs": {
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                },
                "batchpaths": {
                    "path": str(batchpaths_path),
                    "bytes": batchpaths_path.stat().st_size,
                    "sha256": sha256_file(batchpaths_path),
                },
                "layout": layout_receipt,
                "source_artifact_receipt": {
                    "path": str(source_receipt_path),
                    "bytes": source_receipt_path.stat().st_size,
                    "sha256": sha256_file(source_receipt_path),
                },
                "source_artifacts": artifact_receipts,
                "source_route_provenance": {
                    name: dict(specs[name].provenance) for name in sorted(specs)
                },
                "tool": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "extraction": extraction_receipts,
            "outputs": {
                "directory": str(output_dir),
                "batch_count": len(batch_receipts),
                "unit_count": len(units),
                "unit_ids_sha256": canonical_json_sha256(expected_unit_ids),
                "batches": batch_receipts,
            },
        }
        if comparison is not None:
            receipt["comparison_diagnostic"] = comparison
        os.replace(temporary, output_dir)
        _atomic_write(receipt_path, _json_bytes(receipt, pretty=True))
        return receipt
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def inspect_span_snapshot(
    *,
    unit_dir: str | Path,
    manifest_path: str | Path,
    batchpaths_path: str | Path,
    layout_path: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    batchpaths = Path(batchpaths_path).resolve()
    receipts, units, artifact_sha = _read_snapshot(
        Path(unit_dir), manifest, batchpaths, Path(layout_path)
    )
    return {
        "schema_version": "span-unit-snapshot-inspection-v1",
        "labels_read_created_or_inferred": False,
        "snapshot_artifact_sha256": artifact_sha,
        "manifest_sha256": sha256_file(manifest),
        "batchpaths_sha256": sha256_file(batchpaths),
        "batch_count": len(receipts),
        "unit_count": len(units),
        "batches": receipts,
    }


def verify_rehydration_receipt(
    unit_dir: str | Path,
    receipt_path: str | Path,
    manifest_path: str | Path,
    batchpaths_path: str | Path,
    layout_path: str | Path,
) -> dict[str, Any]:
    receipt_path = Path(receipt_path).resolve()
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "span-unit-rehydration-receipt-v1":
        raise RehydrationError(f"{receipt_path}: not a SPAN rehydration receipt")
    if receipt.get("operation") != "text_payload_rehydration_only":
        raise RehydrationError(f"{receipt_path}: unexpected recovery operation")
    if receipt.get("labels_read_created_or_inferred") is not False:
        raise RehydrationError(f"{receipt_path}: receipt does not prove a label-free operation")
    inspection = inspect_span_snapshot(
        unit_dir=unit_dir,
        manifest_path=manifest_path,
        batchpaths_path=batchpaths_path,
        layout_path=layout_path,
    )
    if inspection["snapshot_artifact_sha256"] != receipt.get("snapshot_artifact_sha256"):
        raise RehydrationError(f"{receipt_path}: current unit directory has drifted from its receipt")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("batches") != inspection["batches"]:
        raise RehydrationError(f"{receipt_path}: per-batch output receipts do not match")
    if outputs.get("batch_count") != inspection["batch_count"] or outputs.get("unit_count") != inspection["unit_count"]:
        raise RehydrationError(f"{receipt_path}: output counts do not match")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise RehydrationError(f"{receipt_path}: missing input provenance")
    for key, observed in (
        ("manifest", inspection["manifest_sha256"]),
        ("batchpaths", inspection["batchpaths_sha256"]),
    ):
        item = inputs.get(key)
        if not isinstance(item, dict) or item.get("sha256") != observed:
            raise RehydrationError(f"{receipt_path}: {key} input provenance does not match")
    source_artifacts = inputs.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise RehydrationError(f"{receipt_path}: missing source artifact provenance")
    for index, artifact in enumerate(source_artifacts):
        if not isinstance(artifact, dict):
            raise RehydrationError(f"{receipt_path}: source artifact {index} is malformed")
        _require_sha256(artifact.get("sha256", ""), f"receipt source artifact {index}")
        _require_pinned_revision(
            artifact.get("revision", ""), f"receipt source artifact {index}"
        )
        if not artifact.get("repo_id") or not artifact.get("repository_path"):
            raise RehydrationError(f"{receipt_path}: source artifact {index} lacks repository identity")
    verified = bool(receipt.get("snapshot_equivalence_verified"))
    status = receipt.get("snapshot_equivalence_status")
    if verified != (status == "verified_expected_artifact_sha256"):
        raise RehydrationError(f"{receipt_path}: inconsistent snapshot verification status")
    if verified and receipt.get("expected_snapshot_artifact_sha256") != inspection["snapshot_artifact_sha256"]:
        raise RehydrationError(f"{receipt_path}: expected artifact SHA does not bind this snapshot")
    if not verified and receipt.get("expected_snapshot_artifact_sha256") is not None:
        raise RehydrationError(f"{receipt_path}: unverified receipt cannot claim an expected artifact SHA")
    if receipt.get("promotion_eligible") is not False:
        raise RehydrationError(f"{receipt_path}: rehydrated LLM silver must never be promotion eligible")
    if (
        receipt.get("research_fit_eligible") is not True
        or receipt.get("research_evidence_scope") != "LLM_silver_comparison_only"
    ):
        raise RehydrationError(
            f"{receipt_path}: receipt must authorize only LLM-silver research fitting/comparison"
        )
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "snapshot_artifact_sha256": inspection["snapshot_artifact_sha256"],
        "snapshot_equivalence_status": status,
        "snapshot_equivalence_verified": verified,
        "research_fit_eligible": True,
        "research_evidence_scope": "LLM_silver_comparison_only",
        "production_eligible": False,
    }

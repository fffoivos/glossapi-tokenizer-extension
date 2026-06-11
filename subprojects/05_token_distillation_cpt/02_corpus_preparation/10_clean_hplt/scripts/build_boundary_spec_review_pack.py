#!/usr/bin/env python3
"""Build a full-text boundary-spec review pack for unresolved HPLT actions.

This is non-destructive. It gathers reviewed changing-action rows that still
lack concrete char/token deltas, removes rows already covered by validated
shadow evidence, deduplicates to one row per source document, and materializes
full text for boundary review. The output is a review pack plus an annotation
template; it does not trim, split, drop, quarantine, or mutate source HPLT rows.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - cluster preflight catches this.
    pc = ds = None
    PYARROW_IMPORT_ERROR = exc
else:
    PYARROW_IMPORT_ERROR = None

try:
    from tokenizers import Tokenizer
except Exception as exc:  # pragma: no cover - optional.
    Tokenizer = None
    TOKENIZERS_IMPORT_ERROR = exc
else:
    TOKENIZERS_IMPORT_ERROR = None


SCHEMA_VERSION = "hplt-boundary-spec-review-pack-v1"
ANNOTATION_SCHEMA_VERSION = "hplt-boundary-spec-annotation-template-v1"
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_INPUT_PARQUET = "/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet"
DEFAULT_TOKENIZER_JSON = "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json"
DEFAULT_OUTPUT_PREFIX = "reports/boundary_spec_review_pack"
DEFAULT_SOURCE_DATASET = "HPLT/ell_Grek_ge8_no_mt_clean60"
TIMESTAMP_SUFFIX_RE = re.compile(r"\d{8}T\d{6}Z$")

CHANGING_ACTIONS = {
    "drop_doc",
    "normalize_or_trim_span",
    "quarantine",
    "split_doc",
    "trim_prefix",
    "trim_span",
    "trim_suffix",
}

ACTION_PRIORITY = {
    "split_doc": 100,
    "trim_suffix": 90,
    "trim_prefix": 85,
    "trim_span": 80,
    "normalize_or_trim_span": 70,
    "drop_doc": 60,
    "quarantine": 50,
}

FILE_PRIORITY_PATTERNS = [
    (re.compile(r"false_negative_global200_review_"), 100),
    (re.compile(r"false_negative_triage_review_annotations_"), 95),
    (re.compile(r"false_negative_representative_review_annotations_"), 90),
    (re.compile(r"destructive_span_policy_review_annotations_"), 85),
    (re.compile(r"quarantine_precision_review_annotations_"), 80),
    (re.compile(r"pilot_review_annotations_20260605T204701Z"), 70),
    (re.compile(r"pilot_review_annotations_20260605T202344Z"), 60),
    (re.compile(r"pilot_review_annotations_20260605T200254Z"), 50),
    (re.compile(r"pilot_review_annotations_20260605T194002Z"), 40),
    (re.compile(r"e001_control_char_review_annotations_"), 10),
]


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def annotation_files(reports_dir: Path) -> list[Path]:
    files = []
    latest_global_prevalence: Path | None = None
    for path in sorted(reports_dir.glob("*annotations*.jsonl")):
        name = path.name
        if name.startswith("boundary_"):
            continue
        if "annotation_template" in name:
            continue
        if name.startswith("global_prevalence_review_annotations_"):
            latest_global_prevalence = path
            continue
        files.append(path)
    if latest_global_prevalence is not None:
        files.append(latest_global_prevalence)
    global200 = reports_dir / "false_negative_global200_review_20260606T023000Z.jsonl"
    if global200.exists() and global200 not in files:
        files.append(global200)
    return sorted(files)


def true_error_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in (
        "review_true_error_type_ids",
        "true_error_type_ids",
        "review_false_negative_error_ids",
        "missed_true_error_type_ids",
    ):
        value = row.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).startswith("E"))
    return sorted(set(ids))


def row_source_doc_id(row: dict[str, Any]) -> str:
    return compact_text(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("policy_evaluation_sample_id"))


def row_action(row: dict[str, Any]) -> str:
    return compact_text(row.get("review_correct_action") or row.get("candidate_action") or row.get("action"))


def file_priority(path: str) -> int:
    for pattern, priority in FILE_PRIORITY_PATTERNS:
        if pattern.search(path):
            return priority
    return 0


def row_priority(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    action = row_action(row)
    label = compact_text(row.get("review_label"))
    label_priority = {
        "false_negative_found": 100,
        "true_positive": 90,
        "partial_true_positive": 80,
        "unclear": 20,
        "false_positive": 10,
        "clean": 0,
    }.get(label, 30)
    has_doc_text_path = 1 if row.get("doc_text_path") else 0
    return (
        file_priority(compact_text(row.get("_annotation_file"))),
        label_priority,
        ACTION_PRIORITY.get(action, 0),
        has_doc_text_path,
        compact_text(row.get("_annotation_file")),
    )


def latest_e001_shadow_overlay(reports_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summaries = sorted(reports_dir.glob("e001_shadow_overlay_*_summary.json"))
    if not summaries:
        return ({}, {}, [])
    summary_path = summaries[-1]
    summary = read_json(summary_path)
    base = summary_path.name.removesuffix("_summary.json")
    validation_path = reports_dir / f"{base}_manifest_validation.json"
    manifest_path = reports_dir / f"{base}_manifest.jsonl"
    validation = read_json(validation_path) if validation_path.exists() else {}
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    return (summary, validation, manifest_rows)


def add_resolution_keys(
    exact_keys: set[tuple[str, str, str]],
    source_error_keys: set[tuple[str, str]],
    source_action_keys: set[tuple[str, str]],
    row: dict[str, Any],
) -> None:
    source_doc_id = row_source_doc_id(row)
    action = row_action(row)
    for error_id in true_error_ids(row) or [str(item) for item in row.get("error_type_ids") or []]:
        exact_keys.add((source_doc_id, action, error_id))
        source_error_keys.add((source_doc_id, error_id))
    source_action_keys.add((source_doc_id, action))


def boundary_materialization_delta_evidence(reports_dir: Path) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    exact_keys: set[tuple[str, str, str]] = set()
    source_error_keys: set[tuple[str, str]] = set()
    source_action_keys: set[tuple[str, str]] = set()
    for manifest_path in sorted(reports_dir.glob("boundary_*materialization_*_decision_manifest.jsonl")):
        for row in read_jsonl(manifest_path):
            action = row_action(row)
            has_delta = row.get("chars_removed") is not None and row.get("tokens_removed") is not None
            has_holdout_delta = (
                action == "quarantine"
                and row.get("held_out_chars") is not None
                and row.get("held_out_tokens") is not None
            )
            if not has_delta and not has_holdout_delta:
                continue
            add_resolution_keys(exact_keys, source_error_keys, source_action_keys, row)
    return exact_keys, source_error_keys, source_action_keys


def resolved_delta_evidence(reports_dir: Path) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    summary, validation, manifest_rows = latest_e001_shadow_overlay(reports_dir)
    valid = (
        bool(summary)
        and bool(validation)
        and int(validation.get("violation_count") or 0) == 0
        and int(summary.get("materialized_rows") or 0) > 0
        and not summary.get("hash_mismatches")
        and not summary.get("missing_expected_source_doc_ids")
        and not summary.get("unchanged_after_source_doc_ids")
    )
    exact_keys, source_error_keys, source_action_keys = boundary_materialization_delta_evidence(reports_dir)
    if not valid:
        return exact_keys, source_error_keys, source_action_keys
    for row in manifest_rows:
        if row.get("chars_removed") is None or row.get("tokens_removed") is None:
            continue
        add_resolution_keys(exact_keys, source_error_keys, source_action_keys, row)
    return exact_keys, source_error_keys, source_action_keys


def resolved_delta_keys(reports_dir: Path) -> set[tuple[str, str, str]]:
    return resolved_delta_evidence(reports_dir)[0]


def load_tokenizer(path: str | None) -> Any:
    if not path:
        return None
    if TOKENIZERS_IMPORT_ERROR is not None:
        raise RuntimeError(f"tokenizers import failed: {TOKENIZERS_IMPORT_ERROR}")
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(f"tokenizer.json not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def count_tokens(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text).ids)


def collect_unresolved_rows(reports_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    resolved, resolved_source_error, resolved_source_action = resolved_delta_evidence(reports_dir)
    raw_rows: list[dict[str, Any]] = []
    resolved_rows = 0
    source_review_rows = 0
    for path in annotation_files(reports_dir):
        for row in read_jsonl(path):
            action = row_action(row)
            if action not in CHANGING_ACTIONS:
                continue
            if row.get("chars_removed") is not None and row.get("tokens_removed") is not None:
                continue
            source_doc_id = row_source_doc_id(row)
            error_ids = true_error_ids(row)
            resolved_by_exact_action = any((source_doc_id, action, error_id) in resolved for error_id in error_ids)
            resolved_by_final_action = any((source_doc_id, error_id) in resolved_source_error for error_id in error_ids)
            resolved_by_source_action = not error_ids and (source_doc_id, action) in resolved_source_action
            if resolved_by_exact_action or resolved_by_final_action or resolved_by_source_action:
                resolved_rows += 1
                continue
            source_review_rows += 1
            enriched = dict(row)
            enriched["_annotation_file"] = str(path)
            enriched["_reviewed_action"] = action
            enriched["_source_doc_id"] = source_doc_id
            enriched["_error_type_ids"] = error_ids
            raw_rows.append(enriched)

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        grouped[row["_source_doc_id"]].append(row)

    selected: list[dict[str, Any]] = []
    for source_doc_id, group in grouped.items():
        group_sorted = sorted(group, key=row_priority, reverse=True)
        best = dict(group_sorted[0])
        best["_source_review_rows_count"] = len(group)
        best["_source_annotation_files"] = sorted(set(compact_text(item.get("_annotation_file")) for item in group))
        best["_all_review_labels"] = sorted(set(compact_text(item.get("review_label")) for item in group if item.get("review_label")))
        best["_all_error_type_ids"] = sorted(set(error_id for item in group for error_id in item.get("_error_type_ids", [])))
        selected.append(best)

    selected.sort(
        key=lambda row: (
            -ACTION_PRIORITY.get(row_action(row), 0),
            -(row.get("chars_before") or 0),
            compact_text(row.get("host")),
            row_source_doc_id(row),
        )
    )
    return (selected, source_review_rows, resolved_rows)


def read_existing_doc_text(path: str | None) -> tuple[str | None, dict[str, Any]]:
    if not path:
        return None, {
            "text_source": None,
            "doc_text_path_exists": False,
            "doc_text_read_error": "missing doc_text_path",
        }
    if not os.path.exists(path):
        return None, {
            "text_source": "doc_text_path",
            "doc_text_path_exists": False,
            "doc_text_read_error": "path does not exist",
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except Exception as exc:
        return None, {
            "text_source": "doc_text_path",
            "doc_text_path_exists": True,
            "doc_text_read_error": str(exc),
        }
    return text, {
        "text_source": "doc_text_path",
        "doc_text_path_exists": True,
        "doc_text_read_error": None,
    }


def source_filter(schema_names: set[str], source_dataset: str | None, source_ids: set[str]) -> Any:
    filters: list[Any] = []
    if source_dataset and "source_dataset" in schema_names:
        filters.append(pc.field("source_dataset") == source_dataset)
    if "source_doc_id" in schema_names:
        filters.append(pc.field("source_doc_id").isin(list(source_ids)))
    elif "doc_key" in schema_names:
        filters.append(pc.field("doc_key").isin(list(source_ids)))
    if not filters:
        return None
    expr = filters[0]
    for extra in filters[1:]:
        expr = expr & extra
    return expr


def columns_for_schema(schema_names: set[str]) -> list[str]:
    wanted = [
        "source_doc_id",
        "source_dataset",
        "doc_key",
        "text",
        "title",
        "url",
        "host",
        "quality_bin",
        "source_metadata_json",
    ]
    return [name for name in wanted if name in schema_names]


def load_texts_from_parquet(input_parquet: str, source_ids: set[str], source_dataset: str | None, batch_size: int) -> dict[str, dict[str, Any]]:
    if not source_ids:
        return {}
    if PYARROW_IMPORT_ERROR is not None:
        raise RuntimeError(f"pyarrow import failed: {PYARROW_IMPORT_ERROR}")
    dataset = ds.dataset(input_parquet, format="parquet")
    schema_names = set(dataset.schema.names)
    columns = columns_for_schema(schema_names)
    if "text" not in columns:
        raise RuntimeError("Input parquet does not expose a text column")
    filter_expr = source_filter(schema_names, source_dataset, source_ids)
    loaded: dict[str, dict[str, Any]] = {}
    for batch in dataset.to_batches(columns=columns, filter=filter_expr, batch_size=batch_size):
        for row in batch.to_pylist():
            source_doc_id = compact_text(row.get("source_doc_id") or row.get("doc_key"))
            if source_doc_id in source_ids and source_doc_id not in loaded:
                loaded[source_doc_id] = row
    return loaded


def infer_boundary_requirements(action: str) -> list[str]:
    if action in {"trim_prefix", "trim_suffix", "trim_span", "normalize_or_trim_span"}:
        return ["span_ranges with exact character start/end offsets", "good_text_loss_estimate", "boundary_notes"]
    if action == "split_doc":
        return ["split_parts with exact character start/end offsets for each retained child", "dropped_span_ranges for separators or boilerplate", "boundary_notes"]
    if action in {"drop_doc", "quarantine"}:
        return ["reviewed whole-doc rationale", "good_text_loss_estimate", "boundary_notes"]
    return ["boundary_notes"]


def make_pack_row(
    row: dict[str, Any],
    text: str,
    text_meta: dict[str, Any],
    parquet_row: dict[str, Any] | None,
    tokenizer: Any,
    created_at_utc: str,
) -> dict[str, Any]:
    source_doc_id = row_source_doc_id(row)
    action = row_action(row)
    text_hash = sha256_text(text)
    expected_hash = compact_text(row.get("text_sha256_before"))
    if expected_hash and expected_hash == text_hash:
        hash_check_status = "match"
        hash_matches_before: bool | None = True
    elif expected_hash:
        hash_check_status = "mismatch"
        hash_matches_before = False
    else:
        hash_check_status = "no_expected_hash"
        hash_matches_before = None
    tokens_before = count_tokens(tokenizer, text)
    error_ids = row.get("_all_error_type_ids") or row.get("_error_type_ids") or true_error_ids(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_doc_id": source_doc_id,
        "parent_source_doc_id": source_doc_id,
        "derived_doc_id": None,
        "is_shadow_record": False,
        "source_dataset": compact_text((parquet_row or {}).get("source_dataset") or row.get("source_dataset") or DEFAULT_SOURCE_DATASET),
        "doc_key": compact_text((parquet_row or {}).get("doc_key") or row.get("doc_key") or source_doc_id),
        "url": (parquet_row or {}).get("url") or row.get("url"),
        "host": (parquet_row or {}).get("host") or row.get("host"),
        "quality_bin": (parquet_row or {}).get("quality_bin") or row.get("quality_bin"),
        "action": action,
        "candidate_action": row.get("candidate_action"),
        "review_correct_action": row.get("review_correct_action"),
        "review_label": row.get("review_label"),
        "review_true_error_type_ids": row.get("review_true_error_type_ids") or [],
        "review_false_negative_error_ids": row.get("review_false_negative_error_ids") or [],
        "error_type_ids": error_ids,
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "detector_scores": row.get("detector_scores") or {},
        "reason_codes": row.get("reason_codes") or row.get("evaluation_reasons") or [],
        "sample_set": row.get("sample_set") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "source_annotation_file": row.get("_annotation_file"),
        "source_annotation_files": row.get("_source_annotation_files") or [],
        "source_review_rows_count": row.get("_source_review_rows_count"),
        "all_review_labels": row.get("_all_review_labels") or [],
        "all_error_type_ids": row.get("_all_error_type_ids") or [],
        "policy_evaluation_sample_id": row.get("policy_evaluation_sample_id"),
        "evidence_excerpt": row.get("evidence_excerpt"),
        "review_notes": row.get("review_notes"),
        "review_span_or_split_notes": row.get("review_span_or_split_notes"),
        "review_good_text_loss_risk": row.get("review_good_text_loss_risk"),
        "doc_text_path": row.get("doc_text_path"),
        "text_source": text_meta.get("text_source"),
        "doc_text_path_exists": text_meta.get("doc_text_path_exists"),
        "doc_text_read_error": text_meta.get("doc_text_read_error"),
        "text_sha256_before": expected_hash or text_hash,
        "expected_text_sha256_before": expected_hash or None,
        "full_text_sha256": text_hash,
        "full_text_sha256_matches_before": hash_matches_before,
        "full_text_hash_check_status": hash_check_status,
        "chars_before": len(text),
        "tokens_before": tokens_before,
        "line_count": text.count("\n") + 1 if text else 0,
        "boundary_review_status": "needs_boundary_spec",
        "boundary_required_fields": infer_boundary_requirements(action),
        "span_ranges": [],
        "split_parts": [],
        "dropped_span_ranges": [],
        "chars_after": None,
        "chars_removed": None,
        "tokens_after": None,
        "tokens_removed": None,
        "good_text_loss_estimate": None,
        "full_text": text,
    }


def make_annotation_template_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "source_doc_id": row["source_doc_id"],
        "parent_source_doc_id": row["parent_source_doc_id"],
        "source_review_rows_count": row["source_review_rows_count"],
        "source_annotation_files": row["source_annotation_files"],
        "host": row.get("host"),
        "url": row.get("url"),
        "quality_bin": row.get("quality_bin"),
        "text_sha256_before": row.get("full_text_sha256"),
        "chars_before": row.get("chars_before"),
        "tokens_before": row.get("tokens_before"),
        "error_type_ids": row.get("error_type_ids") or [],
        "reviewed_action": row.get("action"),
        "boundary_review_status": "needs_boundary_spec",
        "boundary_review_action": row.get("action"),
        "span_ranges": [],
        "split_parts": [],
        "dropped_span_ranges": [],
        "chars_after": None,
        "chars_removed": None,
        "tokens_after": None,
        "tokens_removed": None,
        "good_text_loss_estimate": None,
        "boundary_notes": "",
        "reviewer": "",
        "reviewed_at_utc": None,
        "boundary_instructions": row.get("boundary_required_fields") or [],
    }


def markdown_excerpt(text: str, limit: int) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) <= limit:
        return normalized
    head = normalized[: limit // 2]
    tail = normalized[-(limit // 2) :]
    return head + "\n...[middle truncated in Markdown preview; full text is in JSONL]...\n" + tail


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], markdown_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# HPLT Boundary-Spec Review Pack\n\n")
        out.write("This pack is non-destructive. It exists to convert reviewed action labels into exact boundaries, split parts, or whole-doc holdout rationales before any broader duplicate/shadow overlay is built.\n\n")
        out.write("## Summary\n\n")
        for key in [
            "source_review_rows_missing_deltas",
            "resolved_by_existing_shadow_rows",
            "unique_source_docs",
            "output_rows",
            "read_error_count",
            "hash_mismatch_count",
            "tokenized_rows",
        ]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Actions\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["by_action"].items():
            out.write(f"| `{action}` | {count} |\n")
        out.write("\n## Review Fields Needed\n\n")
        out.write("- trim/normalize actions: fill `span_ranges` with exact character offsets.\n")
        out.write("- split actions: fill `split_parts` for retained child documents and `dropped_span_ranges` for separators or boilerplate.\n")
        out.write("- drop/quarantine actions: confirm the whole-doc rationale and good-text-loss estimate.\n")
        out.write("- all rows: fill char/token deltas after the boundary is known; do not edit source HPLT rows.\n\n")
        out.write("## Rows\n\n")
        for index, row in enumerate(rows, 1):
            out.write(f"### {index}. `{row['source_doc_id']}`\n\n")
            out.write(f"- action: `{row.get('action')}`\n")
            out.write(f"- errors: `{', '.join(row.get('error_type_ids') or [])}`\n")
            out.write(f"- host: `{row.get('host') or ''}`\n")
            out.write(f"- url: `{row.get('url') or ''}`\n")
            out.write(f"- chars/tokens before: `{row.get('chars_before')}` / `{row.get('tokens_before')}`\n")
            out.write(f"- text source: `{row.get('text_source')}`\n")
            out.write(f"- source review rows: `{row.get('source_review_rows_count')}`\n")
            out.write(f"- review notes: `{row.get('review_span_or_split_notes') or row.get('review_notes') or ''}`\n")
            out.write(f"- required fields: `{'; '.join(row.get('boundary_required_fields') or [])}`\n\n")
            out.write("```text\n")
            out.write(markdown_excerpt(row.get("full_text") or "", markdown_chars))
            out.write("\n```\n\n")


def make_summary(
    rows: list[dict[str, Any]],
    source_review_rows: int,
    resolved_rows: int,
    read_errors: list[dict[str, Any]],
    output_prefix: Path,
    tokenizer_json: str | None,
) -> dict[str, Any]:
    by_action = collections.Counter(row.get("action") for row in rows)
    by_text_source = collections.Counter(row.get("text_source") for row in rows)
    hash_status_counts = collections.Counter(row.get("full_text_hash_check_status") for row in rows)
    hash_mismatch_count = hash_status_counts.get("mismatch", 0)
    tokenized_rows = sum(1 for row in rows if row.get("tokens_before") is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": rows[0]["created_at_utc"] if rows else utc_timestamp(),
        "policy_note": "Boundary-spec review pack only. This does not approve automatic cleaning and does not mutate source HPLT rows.",
        "source_review_rows_missing_deltas": source_review_rows,
        "resolved_by_existing_shadow_rows": resolved_rows,
        "unique_source_docs": len(rows),
        "output_rows": len(rows),
        "by_action": dict(by_action.most_common()),
        "by_text_source": dict(by_text_source.most_common()),
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:100],
        "hash_status_counts": dict(hash_status_counts.most_common()),
        "hash_mismatch_count": hash_mismatch_count,
        "tokenized_rows": tokenized_rows,
        "total_chars_before": sum(row.get("chars_before") or 0 for row in rows),
        "total_tokens_before": sum(row.get("tokens_before") or 0 for row in rows if row.get("tokens_before") is not None),
        "tokenizer_json": tokenizer_json,
        "review_pack_jsonl": str(output_prefix) + ".jsonl",
        "review_pack_md": str(output_prefix) + ".md",
        "annotation_template_jsonl": str(output_prefix) + "_annotation_template.jsonl",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--input-parquet", default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--tokenizer-json", default=DEFAULT_TOKENIZER_JSON)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--markdown-chars", type=int, default=6000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_prefix = Path(args.output_prefix)
    if not TIMESTAMP_SUFFIX_RE.search(output_prefix.name):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    selected_rows, source_review_rows, resolved_rows = collect_unresolved_rows(reports_dir)
    tokenizer = load_tokenizer(args.tokenizer_json) if args.tokenizer_json else None

    text_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    needs_parquet: set[str] = set()
    for row in selected_rows:
        source_doc_id = row_source_doc_id(row)
        text, meta = read_existing_doc_text(row.get("doc_text_path"))
        if text is not None:
            text_by_id[source_doc_id] = (text, meta)
        else:
            needs_parquet.add(source_doc_id)

    parquet_rows = load_texts_from_parquet(args.input_parquet, needs_parquet, args.source_dataset, args.batch_size)
    output_rows: list[dict[str, Any]] = []
    read_errors: list[dict[str, Any]] = []
    for row in selected_rows:
        source_doc_id = row_source_doc_id(row)
        parquet_row = parquet_rows.get(source_doc_id)
        if source_doc_id in text_by_id:
            text, text_meta = text_by_id[source_doc_id]
        elif parquet_row:
            text = compact_text(parquet_row.get("text"))
            text_meta = {
                "text_source": "input_parquet",
                "doc_text_path_exists": bool(row.get("doc_text_path")),
                "doc_text_read_error": None,
            }
        else:
            read_errors.append(
                {
                    "source_doc_id": source_doc_id,
                    "doc_text_path": row.get("doc_text_path"),
                    "reason": "could not load text from doc_text_path or input parquet",
                }
            )
            continue
        output_rows.append(make_pack_row(row, text, text_meta, parquet_row, tokenizer, args.timestamp))

    annotation_template_rows = [make_annotation_template_row(row) for row in output_rows]
    summary = make_summary(output_rows, source_review_rows, resolved_rows, read_errors, output_prefix, args.tokenizer_json)

    write_jsonl(Path(str(output_prefix) + ".jsonl"), output_rows)
    write_jsonl(Path(str(output_prefix) + "_annotation_template.jsonl"), annotation_template_rows)
    write_json(Path(str(output_prefix) + "_summary.json"), summary)
    write_markdown(Path(str(output_prefix) + ".md"), output_rows, summary, args.markdown_chars)

    print(
        json.dumps(
            {
                "review_pack_jsonl": str(output_prefix) + ".jsonl",
                "annotation_template_jsonl": str(output_prefix) + "_annotation_template.jsonl",
                "summary": str(output_prefix) + "_summary.json",
                "markdown": str(output_prefix) + ".md",
                "source_review_rows_missing_deltas": source_review_rows,
                "resolved_by_existing_shadow_rows": resolved_rows,
                "output_rows": len(output_rows),
                "read_errors": len(read_errors),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

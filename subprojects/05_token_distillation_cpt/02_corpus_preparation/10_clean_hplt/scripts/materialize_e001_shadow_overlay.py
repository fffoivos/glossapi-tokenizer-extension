#!/usr/bin/env python3
"""Materialize reviewed E001 fixes as duplicate/shadow records.

This is intentionally narrow: it only consumes reviewed E001 rows, removes
literal E001 characters from duplicate text, and writes a manifest plus shadow
records. It does not rewrite or filter the source HPLT parquet.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
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
except Exception as exc:  # pragma: no cover - optional preflight catches this.
    Tokenizer = None
    TOKENIZERS_IMPORT_ERROR = exc
else:
    TOKENIZERS_IMPORT_ERROR = None


SOURCE_DATASET_DEFAULT = "HPLT/ell_Grek_ge8_no_mt_clean60"
E001_RE = re.compile(r"[\uFFFD\uE000-\uF8FF\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


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


def reviewed_e001_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_doc_id = compact_text(row.get("source_doc_id") or row.get("parent_source_doc_id"))
        if not source_doc_id:
            continue
        true_ids = set(row.get("review_true_error_type_ids") or row.get("error_type_ids") or [])
        if row.get("review_label") != "true_positive" or "E001" not in true_ids:
            continue
        selected[source_doc_id] = row
    if not selected:
        raise RuntimeError(f"No reviewed true-positive E001 rows found in {path}")
    return selected


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
        "source_metadata_json",
        "quality_bin",
        "source_mix_chars",
    ]
    return [name for name in wanted if name in schema_names]


def row_id(row: dict[str, Any], fallback_index: int) -> str:
    return compact_text(row.get("source_doc_id") or row.get("doc_key") or f"missing-id::{fallback_index}")


def remove_e001_chars(text: str) -> tuple[str, list[dict[str, Any]]]:
    spans: list[dict[str, Any]] = []
    chunks: list[str] = []
    last = 0
    for match in E001_RE.finditer(text):
        start, end = match.span()
        ch = match.group(0)
        chunks.append(text[last:start])
        spans.append(
            {
                "start": start,
                "end": end,
                "replacement": "",
                "codepoint": f"U+{ord(ch):04X}",
                "reason": "remove_e001_character",
            }
        )
        last = end
    chunks.append(text[last:])
    return "".join(chunks), spans


def make_derived_doc_id(source_doc_id: str, text_sha256_after: str) -> str:
    return f"{source_doc_id}::shadow::e001_remove::{text_sha256_after[:12]}"


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if PYARROW_IMPORT_ERROR is not None:
        raise RuntimeError(f"pyarrow import failed: {PYARROW_IMPORT_ERROR}")

    reviewed = reviewed_e001_rows(Path(args.annotations))
    tokenizer = load_tokenizer(args.tokenizer_json)
    dataset = ds.dataset(args.input, format="parquet")
    schema_names = set(dataset.schema.names)
    columns = columns_for_schema(schema_names)
    if "text" not in columns:
        raise RuntimeError("Input does not expose a text column")

    source_ids = set(reviewed)
    filter_expr = source_filter(schema_names, args.source_dataset, source_ids)
    seen: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []
    unchanged_after: list[str] = []
    missing_expected: list[str] = []
    row_counter = 0

    for batch in dataset.to_batches(columns=columns, filter=filter_expr, batch_size=args.batch_size):
        for row in batch.to_pylist():
            row_counter += 1
            source_doc_id = row_id(row, row_counter)
            if source_doc_id not in reviewed:
                continue
            seen.add(source_doc_id)
            annotation = reviewed[source_doc_id]
            text_before = compact_text(row.get("text"))
            before_hash = sha256_text(text_before)
            expected_hash = compact_text(annotation.get("text_sha256_before"))
            if expected_hash and before_hash != expected_hash:
                hash_mismatches.append(
                    {
                        "source_doc_id": source_doc_id,
                        "expected_text_sha256_before": expected_hash,
                        "actual_text_sha256_before": before_hash,
                    }
                )
                continue

            text_after, spans = remove_e001_chars(text_before)
            after_hash = sha256_text(text_after)
            if before_hash == after_hash:
                unchanged_after.append(source_doc_id)
                continue

            tokens_before = count_tokens(tokenizer, text_before)
            tokens_after = count_tokens(tokenizer, text_after)
            tokens_removed = None
            if tokens_before is not None and tokens_after is not None:
                tokens_removed = tokens_before - tokens_after
            derived_doc_id = make_derived_doc_id(source_doc_id, after_hash)
            chars_before = len(text_before)
            chars_after = len(text_after)

            common = {
                "schema_version": "hplt-cleaning-action-v1",
                "source_doc_id": source_doc_id,
                "parent_source_doc_id": source_doc_id,
                "derived_doc_id": derived_doc_id,
                "is_shadow_record": True,
                "doc_key": compact_text(row.get("doc_key") or annotation.get("doc_key") or source_doc_id),
                "source_dataset": compact_text(row.get("source_dataset") or annotation.get("source_dataset") or args.source_dataset),
                "url": row.get("url") or annotation.get("url"),
                "host": row.get("host") or annotation.get("host"),
                "quality_bin": row.get("quality_bin") or annotation.get("quality_bin"),
                "text_sha256_before": before_hash,
                "text_sha256_after": after_hash,
                "error_type_ids": ["E001"],
                "candidate_error_type_ids": ["E001"],
                "action": "normalize_or_trim_span",
                "candidate_action": annotation.get("candidate_action") or "normalize_or_trim_span",
                "action_status": "applied",
                "confidence": annotation.get("confidence"),
                "reason_codes": sorted(set(annotation.get("reason_codes") or []) | {"reviewed_e001_remove"}),
                "detector_scores": annotation.get("detector_scores") or {},
                "span_ranges": spans,
                "split_parts": [],
                "chars_before": chars_before,
                "chars_after": chars_after,
                "chars_removed": chars_before - chars_after,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_removed": tokens_removed,
                "good_text_loss_estimate": "low",
                "sample_set": sorted(set(annotation.get("sample_set") or []) | {"e001_shadow_overlay_pilot"}),
                "evidence_excerpt": annotation.get("evidence_excerpt"),
                "doc_text_path": None,
                "review_label": annotation.get("review_label"),
                "reviewer": annotation.get("reviewer"),
                "review_notes": annotation.get("review_notes"),
                "review_basis": annotation.get("review_basis"),
                "policy_id": "e001_shadow_overlay_pilot",
                "policy_version": args.policy_version,
                "created_at_utc": args.created_at_utc,
                "shadow_text_path": None,
            }
            shadow_rows.append(
                {
                    **common,
                    "text": text_after,
                    "title": row.get("title"),
                }
            )
            manifest_rows.append(common)

    missing_expected = sorted(source_ids - seen)
    manifest_rows.sort(key=lambda item: item["source_doc_id"])
    shadow_rows.sort(key=lambda item: item["source_doc_id"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"e001_shadow_overlay_{args.timestamp}"
    manifest_path = Path(str(base) + "_manifest.jsonl")
    shadow_path = Path(str(base) + "_shadow_records.jsonl")
    summary_path = Path(str(base) + "_summary.json")
    md_path = Path(str(base) + ".md")

    for row in manifest_rows:
        row["shadow_text_path"] = str(shadow_path)

    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(shadow_path, shadow_rows)

    by_host = collections.Counter(row.get("host") or "" for row in manifest_rows)
    by_action = collections.Counter(row["action"] for row in manifest_rows)
    chars_removed = sum(int(row.get("chars_removed") or 0) for row in manifest_rows)
    tokens_removed_values = [row.get("tokens_removed") for row in manifest_rows if row.get("tokens_removed") is not None]
    tokens_removed = sum(int(value) for value in tokens_removed_values)
    summary = {
        "schema_version": "hplt-e001-shadow-overlay-summary-v1",
        "created_at_utc": args.created_at_utc,
        "policy_note": "E001-only shadow overlay pilot. Original HPLT rows are immutable; changed text exists only in shadow_records.",
        "input": str(args.input),
        "annotations": str(args.annotations),
        "source_dataset": args.source_dataset,
        "tokenizer_json": str(args.tokenizer_json) if args.tokenizer_json else None,
        "expected_reviewed_rows": len(reviewed),
        "materialized_rows": len(manifest_rows),
        "shadow_records": len(shadow_rows),
        "missing_expected_source_doc_ids": missing_expected,
        "hash_mismatches": hash_mismatches,
        "unchanged_after_source_doc_ids": unchanged_after,
        "chars_removed": chars_removed,
        "tokens_removed": tokens_removed if tokens_removed_values else None,
        "token_delta_rows": len(tokens_removed_values),
        "by_action": dict(by_action.most_common()),
        "by_host": by_host.most_common(30),
        "manifest_jsonl": str(manifest_path),
        "shadow_records_jsonl": str(shadow_path),
    }
    write_json(summary_path, summary)

    with md_path.open("w", encoding="utf-8") as out:
        out.write("# E001 Shadow Overlay Pilot\n\n")
        out.write("This is a materialized duplicate/shadow overlay for reviewed E001 rows only. It does not rewrite source HPLT parquet rows.\n\n")
        out.write("- expected reviewed E001 rows: `%s`\n" % len(reviewed))
        out.write("- materialized shadow rows: `%s`\n" % len(manifest_rows))
        out.write("- chars removed: `%s`\n" % chars_removed)
        out.write("- token-delta rows: `%s`\n" % len(tokens_removed_values))
        out.write("- tokens removed: `%s`\n" % (summary["tokens_removed"] if summary["tokens_removed"] is not None else "not available"))
        out.write("- hash mismatches: `%s`\n" % len(hash_mismatches))
        out.write("- missing expected source docs: `%s`\n\n" % len(missing_expected))
        out.write("## Artifacts\n\n")
        out.write("- manifest: `%s`\n" % manifest_path)
        out.write("- shadow records: `%s`\n" % shadow_path)
        out.write("- summary: `%s`\n\n" % summary_path)
        out.write("## Host Counts\n\n")
        out.write("| Host | Rows |\n| --- | ---: |\n")
        for host, count in by_host.most_common():
            out.write("| `%s` | %s |\n" % (host, count))

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet file or dataset directory.")
    parser.add_argument("--annotations", required=True, help="Reviewed E001 annotation JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET_DEFAULT)
    parser.add_argument("--tokenizer-json", default=None, help="tokenizer.json path or HF tokenizer directory.")
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--created-at-utc", default=None)
    parser.add_argument("--policy-version", default="20260606")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.created_at_utc is None:
        args.created_at_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.source_dataset == "":
        args.source_dataset = None

    summary = materialize(args)
    print(
        json.dumps(
            {
                "manifest_jsonl": summary["manifest_jsonl"],
                "shadow_records_jsonl": summary["shadow_records_jsonl"],
                "materialized_rows": summary["materialized_rows"],
                "chars_removed": summary["chars_removed"],
                "tokens_removed": summary["tokens_removed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

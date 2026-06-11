#!/usr/bin/env python3
"""Bootstrap whole-document boundary annotations from a boundary-spec pack.

This only accepts already-reviewed whole-document actions: ``drop_doc`` and
``quarantine`` by default. It does not infer spans, split parts, or text
normalizations.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hplt-boundary-spec-annotation-template-v1"
DEFAULT_REVIEW_PACK = "reports/boundary_spec_review_pack_20260606T050329Z.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/boundary_manual_whole_doc_20260606T050329Z"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def annotation_row(source: dict[str, Any], timestamp: str) -> dict[str, Any]:
    action = compact_text(source.get("action"))
    good_text_loss = source.get("review_good_text_loss_risk")
    if action == "drop_doc" and compact_text(good_text_loss) == "none":
        good_text_loss = 0.0

    before_chars = int(source.get("chars_before") or 0)
    before_tokens = source.get("tokens_before")
    notes = compact_text(source.get("review_span_or_split_notes") or source.get("review_notes"))
    if action == "drop_doc":
        boundary_notes = "Drop whole row: " + notes
        chars_after = 0
        chars_removed = before_chars
        tokens_after = 0 if before_tokens is not None else None
        tokens_removed = before_tokens
    else:
        boundary_notes = "Quarantine whole row pending derived-stream policy: " + notes
        chars_after = None
        chars_removed = None
        tokens_after = None
        tokens_removed = None

    return {
        "schema_version": SCHEMA_VERSION,
        "source_doc_id": source.get("source_doc_id"),
        "parent_source_doc_id": source.get("parent_source_doc_id") or source.get("source_doc_id"),
        "source_review_rows_count": source.get("source_review_rows_count"),
        "source_annotation_files": source.get("source_annotation_files") or [],
        "host": source.get("host"),
        "url": source.get("url"),
        "quality_bin": source.get("quality_bin"),
        "text_sha256_before": source.get("full_text_sha256") or source.get("text_sha256_before"),
        "chars_before": before_chars,
        "tokens_before": before_tokens,
        "error_type_ids": source.get("error_type_ids") or [],
        "reviewed_action": action,
        "boundary_review_action": action,
        "boundary_review_status": "reviewed_accept",
        "span_ranges": [],
        "split_parts": [],
        "dropped_span_ranges": [],
        "chars_after": chars_after,
        "chars_removed": chars_removed,
        "tokens_after": tokens_after,
        "tokens_removed": tokens_removed,
        "good_text_loss_estimate": good_text_loss,
        "boundary_notes": boundary_notes,
        "reviewer": "codex",
        "reviewed_at_utc": timestamp,
        "boundary_instructions": [
            "Reviewed whole-document decision; source rows remain immutable.",
            "No spans or split parts are inferred by this bootstrap.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Whole-Document Boundary Annotation Bootstrap\n\n")
        out.write("This file documents accepted whole-document annotations bootstrapped from reviewed boundary-pack rows. It does not infer text spans, split parts, or normalizations.\n\n")
        out.write("## Summary\n\n")
        for key in ["review_pack", "selected_rows", "skipped_rows", "created_at_utc"]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Actions\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["selected_by_action"].items():
            out.write(f"| `{action}` | {count} |\n")
        out.write("\n## Rows\n\n")
        for row in rows:
            out.write(f"- `{row['source_doc_id']}` action=`{row['boundary_review_action']}` good_text_loss=`{row.get('good_text_loss_estimate')}` host=`{row.get('host')}`\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--actions", default="drop_doc,quarantine")
    args = parser.parse_args()

    review_pack = Path(args.review_pack)
    actions = {item.strip() for item in args.actions.split(",") if item.strip()}
    source_rows = read_jsonl(review_pack)
    selected = [row for row in source_rows if compact_text(row.get("action")) in actions]
    annotations = [annotation_row(row, args.timestamp) for row in selected]

    prefix = Path(args.output_prefix)
    annotation_path = Path(str(prefix) + "_annotations.jsonl")
    summary_path = Path(str(prefix) + "_summary.json")
    markdown_path = Path(str(prefix) + ".md")

    write_jsonl(annotation_path, annotations)
    summary = {
        "schema_version": "hplt-whole-doc-boundary-bootstrap-v1",
        "created_at_utc": args.timestamp,
        "review_pack": str(review_pack),
        "annotation_jsonl": str(annotation_path),
        "summary_json": str(summary_path),
        "markdown": str(markdown_path),
        "selected_rows": len(annotations),
        "skipped_rows": len(source_rows) - len(annotations),
        "selected_by_action": dict(collections.Counter(row.get("boundary_review_action") for row in annotations).most_common()),
        "selected_by_good_text_loss": dict(collections.Counter(compact_text(row.get("good_text_loss_estimate")) for row in annotations).most_common()),
        "policy_note": "Whole-document annotation bootstrap only. Source HPLT rows stay immutable; materialization must run separately on Clariden CPU-only xfer.",
    }
    write_json(summary_path, summary)
    write_markdown(markdown_path, summary, annotations)
    print(json.dumps({"annotations": str(annotation_path), "summary": str(summary_path), "markdown": str(markdown_path), "selected_rows": len(annotations)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

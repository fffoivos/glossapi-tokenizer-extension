#!/usr/bin/env python3
"""Build a focused review pack for boundary rows still pending.

The boundary-spec pack contains full text for all unresolved reviewed actions.
After whole-document `drop_doc` and `quarantine` decisions are materialized,
the remaining work is mostly exact trim/split boundaries. This script creates a
smaller pack for those pending rows with head/tail windows, structural marker
windows, and character offsets. It is review navigation only: it does not infer
cleaning spans, apply a policy, or mutate source HPLT rows.
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


SCHEMA_VERSION = "hplt-pending-boundary-review-pack-v1"
DEFAULT_REVIEW_PACK = "reports/boundary_spec_review_pack_20260606T002713Z.jsonl"
DEFAULT_RESOLVED_MANIFEST = "reports/boundary_whole_doc_materialization_20260606T011700Z_decision_manifest.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/pending_boundary_review_pack"
TIMESTAMP_SUFFIX_RE = re.compile(r"\d{8}T\d{6}Z$")

MARKER_PATTERNS = [
    ("url", re.compile(r"https?://|www\\.", re.I)),
    ("html_or_xml", re.compile(r"<!--|-->|<\\/?[a-zA-Z][^>]{0,80}>|&[a-z]+;", re.I)),
    ("feed_or_rss", re.compile(r"\\b(atom|rss|feed|xml|blogger|wordpress)\\b", re.I)),
    ("comment_or_share", re.compile(r"\\b(σχόλια|σχολια|comments?|share|tweet|facebook|twitter|instagram|linkedin)\\b", re.I)),
    ("date_like", re.compile(r"\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b|\\b\\d{4}-\\d{2}-\\d{2}\\b")),
    ("archive_like", re.compile(r"\\b(archive|αρχείο|αρχειο|category|tag|ετικέτα|ετικετα)\\b", re.I)),
    ("metadata_like", re.compile(r"\\b(author|posted|published|permalink|copyright|all rights reserved|privacy policy)\\b", re.I)),
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


def window(text: str, start: int, end: int, label: str) -> dict[str, Any]:
    start = max(0, min(len(text), start))
    end = max(start, min(len(text), end))
    return {
        "label": label,
        "start": start,
        "end": end,
        "chars": end - start,
        "text": text[start:end],
    }


def head_tail_windows(text: str, chars: int) -> list[dict[str, Any]]:
    if len(text) <= chars * 2:
        return [window(text, 0, len(text), "full_text")]
    return [
        window(text, 0, chars, "head"),
        window(text, len(text) - chars, len(text), "tail"),
    ]


def marker_windows(text: str, marker_context_chars: int, max_markers: int) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for marker_name, pattern in MARKER_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            context_start = max(0, start - marker_context_chars)
            context_end = min(len(text), end + marker_context_chars)
            if any(abs(start - prior_start) < marker_context_chars for prior_start, _ in occupied):
                continue
            occupied.append((start, end))
            markers.append(
                {
                    "marker": marker_name,
                    "match_start": start,
                    "match_end": end,
                    "match_text": text[start:end],
                    "window": window(text, context_start, context_end, f"marker:{marker_name}"),
                }
            )
            if len(markers) >= max_markers:
                return markers
    return markers


def quantile_windows(text: str, chars: int) -> list[dict[str, Any]]:
    if len(text) <= chars * 3:
        return []
    result: list[dict[str, Any]] = []
    for pct in (25, 50, 75):
        center = int(len(text) * (pct / 100.0))
        result.append(window(text, center - chars // 2, center + chars // 2, f"midpoint_{pct}pct"))
    return result


def review_focus(action: str) -> str:
    if action == "trim_suffix":
        return "Find exact terminal artifact start offset; fill span_ranges=[{start,end=len(text),replacement:''}]."
    if action == "trim_prefix":
        return "Find exact first useful-main-text offset; fill span_ranges=[{start=0,end,replacement:''}]."
    if action == "trim_span":
        return "Find bounded internal artifact span(s); fill span_ranges with exact start/end offsets."
    if action == "normalize_or_trim_span":
        return "Find localized artifact span(s) or explicit replacement text; fill span_ranges."
    if action == "split_doc":
        return "Find retained child-document ranges; fill split_parts with exact start/end offsets and dropped_span_ranges for separators/boilerplate."
    if action == "drop_doc":
        return "Confirm whole-row exclusion or leave pending/quarantine if useful Greek can be saved."
    if action == "quarantine":
        return "Confirm holdout rationale or replace with trim/split/drop after exact review."
    return "Fill exact boundary fields before materialization."


def annotation_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hplt-boundary-spec-annotation-template-v1",
        "source_doc_id": row["source_doc_id"],
        "parent_source_doc_id": row.get("parent_source_doc_id") or row["source_doc_id"],
        "source_review_rows_count": row.get("source_review_rows_count"),
        "source_annotation_files": row.get("source_annotation_files") or [],
        "host": row.get("host"),
        "url": row.get("url"),
        "quality_bin": row.get("quality_bin"),
        "text_sha256_before": row.get("full_text_sha256") or row.get("text_sha256_before"),
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
        "boundary_instructions": [review_focus(compact_text(row.get("action")))],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    review_rows = read_jsonl(Path(args.review_pack))
    resolved_ids: set[str] = set()
    for manifest_path in args.resolved_manifest:
        path = Path(manifest_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            resolved_ids.add(compact_text(row.get("source_doc_id")))

    pending_rows = [row for row in review_rows if compact_text(row.get("source_doc_id")) not in resolved_ids]
    pending_rows.sort(
        key=lambda row: (
            compact_text(row.get("action")),
            compact_text(row.get("host")),
            -(row.get("chars_before") or 0),
            compact_text(row.get("source_doc_id")),
        )
    )

    output_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    for row in pending_rows:
        text = compact_text(row.get("full_text"))
        action = compact_text(row.get("action"))
        windows = head_tail_windows(text, args.window_chars)
        if action == "split_doc":
            windows.extend(quantile_windows(text, args.window_chars))
        markers = marker_windows(text, args.marker_context_chars, args.max_markers)
        output = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": args.timestamp,
            "source_doc_id": row.get("source_doc_id"),
            "host": row.get("host"),
            "url": row.get("url"),
            "quality_bin": row.get("quality_bin"),
            "action": action,
            "error_type_ids": row.get("error_type_ids") or [],
            "chars_before": row.get("chars_before"),
            "tokens_before": row.get("tokens_before"),
            "full_text_sha256": row.get("full_text_sha256") or sha256_text(text),
            "review_notes": row.get("review_span_or_split_notes") or row.get("review_notes"),
            "review_focus": review_focus(action),
            "review_windows": windows,
            "marker_windows": markers,
            "full_text": text,
        }
        output_rows.append(output)
        annotation_rows.append(annotation_template(row))

    output_prefix = Path(args.output_prefix)
    if not TIMESTAMP_SUFFIX_RE.search(output_prefix.name):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    pack_path = Path(str(output_prefix) + ".jsonl")
    annotation_path = Path(str(output_prefix) + "_annotation_template.jsonl")
    summary_path = Path(str(output_prefix) + "_summary.json")
    md_path = Path(str(output_prefix) + ".md")

    write_jsonl(pack_path, output_rows)
    write_jsonl(annotation_path, annotation_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": args.timestamp,
        "policy_note": "Pending-boundary review navigation only. This does not apply a cleaning policy or mutate source rows.",
        "review_pack": args.review_pack,
        "resolved_manifests": args.resolved_manifest,
        "pack_jsonl": str(pack_path),
        "annotation_template_jsonl": str(annotation_path),
        "markdown": str(md_path),
        "review_pack_rows": len(review_rows),
        "resolved_source_docs": len(resolved_ids),
        "pending_rows": len(output_rows),
        "pending_by_action": dict(collections.Counter(row.get("action") for row in output_rows).most_common()),
        "pending_by_host": dict(collections.Counter(row.get("host") for row in output_rows).most_common(50)),
        "total_pending_chars": sum(row.get("chars_before") or 0 for row in output_rows),
        "total_pending_tokens": sum(row.get("tokens_before") or 0 for row in output_rows),
        "window_chars": args.window_chars,
        "marker_context_chars": args.marker_context_chars,
        "max_markers": args.max_markers,
    }
    write_json(summary_path, summary)
    write_markdown(md_path, output_rows, summary, args.markdown_window_chars)
    return summary


def markdown_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[window truncated in Markdown; full window text is in JSONL]..."


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], markdown_window_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Pending Boundary Review Pack\n\n")
        out.write("This is a navigation pack for exact boundary review. It provides character-offset windows and marker contexts, but does not infer or apply cleaning boundaries.\n\n")
        out.write("## Summary\n\n")
        for key in ["review_pack_rows", "resolved_source_docs", "pending_rows", "total_pending_chars", "total_pending_tokens"]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Pending By Action\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["pending_by_action"].items():
            out.write(f"| `{action}` | {count} |\n")
        out.write("\n## Rows\n\n")
        for index, row in enumerate(rows, 1):
            out.write(f"### {index}. `{row['source_doc_id']}`\n\n")
            out.write(f"- action: `{row.get('action')}`\n")
            out.write(f"- errors: `{', '.join(row.get('error_type_ids') or [])}`\n")
            out.write(f"- host: `{row.get('host') or ''}`\n")
            out.write(f"- url: `{row.get('url') or ''}`\n")
            out.write(f"- chars/tokens: `{row.get('chars_before')}` / `{row.get('tokens_before')}`\n")
            out.write(f"- focus: {row.get('review_focus')}\n")
            out.write(f"- prior notes: {row.get('review_notes') or ''}\n\n")
            for win in row.get("review_windows") or []:
                out.write(f"#### `{win['label']}` chars `{win['start']}:{win['end']}`\n\n")
                out.write("```text\n")
                out.write(markdown_text(win.get("text") or "", markdown_window_chars))
                out.write("\n```\n\n")
            markers = row.get("marker_windows") or []
            if markers:
                out.write("#### Marker Windows\n\n")
                for marker in markers[:5]:
                    win = marker["window"]
                    out.write(f"- `{marker['marker']}` match `{marker['match_text']}` at `{marker['match_start']}:{marker['match_end']}`, window `{win['start']}:{win['end']}`\n")
                out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--resolved-manifest", action="append", default=[])
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--window-chars", type=int, default=4000)
    parser.add_argument("--markdown-window-chars", type=int, default=2200)
    parser.add_argument("--marker-context-chars", type=int, default=600)
    parser.add_argument("--max-markers", type=int, default=12)
    args = parser.parse_args()
    if not args.resolved_manifest:
        args.resolved_manifest = [DEFAULT_RESOLVED_MANIFEST]

    summary = build(args)
    print(
        json.dumps(
            {
                "pack_jsonl": summary["pack_jsonl"],
                "annotation_template_jsonl": summary["annotation_template_jsonl"],
                "summary": summary["pack_jsonl"].removesuffix(".jsonl") + "_summary.json",
                "pending_rows": summary["pending_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

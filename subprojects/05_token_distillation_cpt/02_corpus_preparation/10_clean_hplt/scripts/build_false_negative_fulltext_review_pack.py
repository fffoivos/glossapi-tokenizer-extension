#!/usr/bin/env python3
"""Materialize full text for false-negative review rows.

The remaining global-random false-negative pack is a compact manifest. This
script reads its `doc_text_path` files and emits a full-text review bundle so
manual decisions can inspect whole documents, not excerpts. It is
non-destructive and never edits source HPLT rows.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import hashlib
import json
import os


SCHEMA_VERSION = "hplt-false-negative-fulltext-review-pack-v1"
DEFAULT_INPUT = "reports/false_negative_global_remaining_review_pack_20260606T014500Z.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/false_negative_global_remaining_fulltext_review_pack_20260606T020000Z"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError("Invalid JSON at %s:%s: %s" % (path, line_number, exc))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_csv(value):
    return set(part.strip() for part in value.split(",") if part.strip())


def read_text(path, max_doc_chars):
    if not path:
        return {
            "full_text_path_exists": False,
            "full_text_read_error": "missing doc_text_path",
            "full_text": "",
            "full_text_truncated": False,
        }
    if not os.path.exists(path):
        return {
            "full_text_path_exists": False,
            "full_text_read_error": "path does not exist",
            "full_text": "",
            "full_text_truncated": False,
        }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except Exception as exc:
        return {
            "full_text_path_exists": True,
            "full_text_read_error": str(exc),
            "full_text": "",
            "full_text_truncated": False,
        }

    truncated = False
    if max_doc_chars and len(text) > max_doc_chars:
        text = text[:max_doc_chars]
        truncated = True
    return {
        "full_text_path_exists": True,
        "full_text_read_error": None,
        "full_text": text,
        "full_text_truncated": truncated,
    }


def make_fulltext_row(row, created_at_utc, input_pack, max_doc_chars):
    text_info = read_text(row.get("doc_text_path"), max_doc_chars)
    text = text_info["full_text"]
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    before_hash = row.get("text_sha256_before")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_pack_schema_version": row.get("schema_version"),
        "source_pack": input_pack,
        "policy_evaluation_sample_id": row.get("policy_evaluation_sample_id"),
        "false_negative_triage_id": row.get("false_negative_triage_id"),
        "source_doc_id": row.get("source_doc_id"),
        "parent_source_doc_id": row.get("parent_source_doc_id") or row.get("source_doc_id"),
        "derived_doc_id": None,
        "is_shadow_record": False,
        "doc_text_path": row.get("doc_text_path"),
        "host": row.get("host"),
        "url": row.get("url"),
        "quality_bin": row.get("quality_bin"),
        "sample_set": row.get("sample_set") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "review_status": row.get("review_status"),
        "review_label": row.get("review_label"),
        "review_correct_action": row.get("review_correct_action"),
        "review_true_error_type_ids": row.get("review_true_error_type_ids") or [],
        "review_notes": row.get("review_notes"),
        "triage_score": row.get("triage_score"),
        "triage_bucket": row.get("triage_bucket"),
        "triage_reasons": row.get("triage_reasons") or [],
        "detector_scores": row.get("detector_scores") or {},
        "evidence_excerpt": row.get("evidence_excerpt"),
        "text_sha256_before": before_hash,
        "full_text_sha256": text_hash,
        "full_text_sha256_matches_before": bool(text_hash and before_hash and text_hash == before_hash),
        "chars_before": row.get("chars_before"),
        "full_text_chars": len(text),
        "full_text_path_exists": text_info["full_text_path_exists"],
        "full_text_read_error": text_info["full_text_read_error"],
        "full_text_truncated": text_info["full_text_truncated"],
        "full_text": text,
    }


def make_summary(rows, source_rows, statuses, input_pack):
    status_counts = collections.Counter(row.get("review_status") for row in rows)
    bucket_counts = collections.Counter(row.get("triage_bucket") for row in rows)
    read_error_count = sum(1 for row in rows if row.get("full_text_read_error"))
    truncated_count = sum(1 for row in rows if row.get("full_text_truncated"))
    hash_match_counts = collections.Counter(
        "match" if row.get("full_text_sha256_matches_before") else
        "missing_or_mismatch"
        for row in rows
    )
    total_chars = sum(row.get("full_text_chars") or 0 for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": rows[0]["created_at_utc"] if rows else None,
        "source_pack": input_pack,
        "source_rows": source_rows,
        "selected_statuses": sorted(statuses),
        "rows": len(rows),
        "review_status_counts": dict(status_counts),
        "triage_bucket_counts": dict(bucket_counts),
        "read_error_count": read_error_count,
        "truncated_count": truncated_count,
        "hash_match_counts": dict(hash_match_counts),
        "total_full_text_chars": total_chars,
        "policy_note": "Full-text review pack only. It does not approve an automatic cleaning policy or mutate source rows.",
    }


def markdown_excerpt(text, limit):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated in Markdown preview; full text is in JSONL]..."


def write_markdown(path, rows, summary, markdown_chars):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Full-Text False-Negative Review Pack\n\n")
        out.write("This is a non-destructive full-text materialization of remaining false-negative review rows. Use the JSONL for complete text and schema fields; this Markdown file previews the rows for navigation.\n\n")
        out.write("## Summary\n\n")
        out.write("- source rows: `%d`\n" % summary["source_rows"])
        out.write("- selected statuses: `%s`\n" % ", ".join(summary["selected_statuses"]))
        out.write("- output rows: `%d`\n" % summary["rows"])
        out.write("- read errors: `%d`\n" % summary["read_error_count"])
        out.write("- truncated full-text rows: `%d`\n" % summary["truncated_count"])
        out.write("- total full-text chars: `%d`\n\n" % summary["total_full_text_chars"])

        out.write("## Triage Buckets\n\n| Bucket | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["triage_bucket_counts"].items()):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n")

        out.write("## Rows\n\n")
        for row in rows:
            out.write("### `%s` - `%s`\n\n" % (row["policy_evaluation_sample_id"], row.get("host") or ""))
            out.write("- review status: `%s`\n" % row.get("review_status"))
            out.write("- triage score: `%s`\n" % row.get("triage_score"))
            out.write("- triage reasons: `%s`\n" % ", ".join(row.get("triage_reasons") or []))
            out.write("- source doc: `%s`\n" % row.get("source_doc_id"))
            out.write("- doc text path: `%s`\n" % row.get("doc_text_path"))
            out.write("- URL: `%s`\n" % row.get("url"))
            out.write("- full-text chars: `%d`\n" % (row.get("full_text_chars") or 0))
            out.write("- sha256 matches before: `%s`\n\n" % row.get("full_text_sha256_matches_before"))
            out.write("```text\n")
            out.write(markdown_excerpt(row.get("full_text") or "", markdown_chars))
            out.write("\n```\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pack", default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--statuses", default="needs_review")
    parser.add_argument("--max-doc-chars", type=int, default=0, help="0 means keep full text")
    parser.add_argument("--markdown-chars", type=int, default=4000)
    args = parser.parse_args()

    selected_statuses = parse_csv(args.statuses)
    source_rows = list(read_jsonl(args.input_pack))
    created_at_utc = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output_rows = [
        make_fulltext_row(row, created_at_utc, args.input_pack, args.max_doc_chars)
        for row in source_rows
        if row.get("review_status") in selected_statuses
    ]
    output_rows.sort(key=lambda row: (
        -(row.get("triage_score") or 0.0),
        row.get("host") or "",
        row.get("policy_evaluation_sample_id") or "",
    ))

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    summary = make_summary(output_rows, len(source_rows), selected_statuses, args.input_pack)
    write_jsonl(args.output_prefix + ".jsonl", output_rows)
    with open(args.output_prefix + "_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(args.output_prefix + ".md", output_rows, summary, args.markdown_chars)

    print("wrote", args.output_prefix + ".jsonl")
    print("rows", len(output_rows))
    print("read_errors", summary["read_error_count"])


if __name__ == "__main__":
    main()

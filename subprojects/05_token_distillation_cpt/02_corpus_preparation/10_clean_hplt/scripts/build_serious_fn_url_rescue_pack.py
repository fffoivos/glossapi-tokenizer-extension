#!/usr/bin/env python3
"""Build a non-destructive review pack for URL-shaped serious false negatives.

The closed false-negative review showed serious E010/E011 misses whose text
detector scores were weak, but whose URLs looked like archive, category, tag,
search, or paginated listing pages. This script tests that hypothesis by
selecting unreviewed candidate-keep rows with those URL/text signals and
materializing full text plus an annotation template. It does not mutate source
HPLT rows or approve any automatic policy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse


SCHEMA_VERSION = "hplt-serious-fn-url-rescue-review-pack-v1"
DEFAULT_POLICY_SAMPLE = "reports/policy_evaluation_sample_20260605T210154Z.jsonl"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_OUTPUT_PREFIX = "reports/serious_fn_url_rescue_pack"

ARCHIVE_PATH_PATTERNS = [
    (re.compile(r"/category(?:/|$)", re.I), "path_category"),
    (re.compile(r"/categories(?:/|$)", re.I), "path_categories"),
    (re.compile(r"/tag(?:/|$)", re.I), "path_tag"),
    (re.compile(r"/tags(?:/|$)", re.I), "path_tags"),
    (re.compile(r"/component/tags/tag/", re.I), "path_component_tags_tag"),
    (re.compile(r"/search/label/", re.I), "path_search_label"),
    (re.compile(r"/label/", re.I), "path_label"),
    (re.compile(r"/archive(?:/|$)", re.I), "path_archive"),
    (re.compile(r"/archives?(?:/|$)", re.I), "path_archives"),
    (re.compile(r"/itemlist/date/", re.I), "path_itemlist_date"),
    (re.compile(r"/page/\d+/?$", re.I), "path_page_number"),
    (re.compile(r"/20\d{2}/\d{1,2}/\d{1,2}/?$", re.I), "path_date_archive"),
]

ARCHIVE_QUERY_KEYS = {
    "catid",
    "category",
    "limit",
    "page",
    "pageno",
    "pagenum",
    "pname",
    "start",
    "tag",
    "view",
}

ARCHIVE_QUERY_VALUES = [
    (re.compile(r"articlelist", re.I), "query_articlelist"),
    (re.compile(r"category|tag|archive|itemlist|search", re.I), "query_listing_value"),
]

TEXT_SIGNAL_PATTERNS = [
    (re.compile(r"συν[εέ]χεια\s+αν[αά]γνωσης", re.I), "text_continue_reading_el"),
    (re.compile(r"read\s+more|continue\s+reading|full\s+article", re.I), "text_continue_reading_en"),
    (re.compile(r"ολ[οό]κληρο\s+το\s+[αά]ρθρο|περισσ[οό]τερα", re.I), "text_more_article_el"),
    (re.compile(r"\b\d{1,2}:\d{2}\b"), "text_timestamp"),
    (re.compile(r"\b(?:page|σελ[ιί]δα)\s+\d+\b", re.I), "text_page_marker"),
]


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_source_doc_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or "")


def annotation_files(reports_dir: Path) -> list[Path]:
    files = []
    for path in sorted(reports_dir.glob("*annotations*.jsonl")):
        name = path.name
        if "annotation_template" in name or name.startswith("boundary_"):
            continue
        files.append(path)
    global200 = reports_dir / "false_negative_global200_review_20260606T023000Z.jsonl"
    if global200.exists() and global200 not in files:
        files.append(global200)
    return sorted(files)


def latest_reviews_by_doc(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_doc: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            doc_id = row_source_doc_id(row)
            if not doc_id:
                continue
            row = dict(row)
            row["_annotation_file"] = str(path)
            by_doc[doc_id] = row
    return by_doc


def all_reviews(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["_annotation_file"] = str(path)
            rows.append(row)
    return rows


def url_signals(url: str) -> tuple[float, list[str]]:
    if not url:
        return 0.0, []
    decoded = unquote(url)
    parsed = urlparse(decoded)
    path = parsed.path or ""
    query = parsed.query or ""
    reasons: list[str] = []

    for pattern, reason in ARCHIVE_PATH_PATTERNS:
        if pattern.search(path):
            reasons.append(reason)

    query_items = parse_qsl(query, keep_blank_values=True)
    query_keys = {key.casefold() for key, _ in query_items}
    for key in sorted(query_keys.intersection(ARCHIVE_QUERY_KEYS)):
        reasons.append(f"query_key_{key}")
    for _, value in query_items:
        for pattern, reason in ARCHIVE_QUERY_VALUES:
            if pattern.search(value):
                reasons.append(reason)

    if "limit" in query_keys and "start" in query_keys:
        reasons.append("query_limit_start")
    if "pname" in query_keys and "catid" in query_keys:
        reasons.append("query_pname_catid")

    unique = sorted(set(reasons))
    score = 0.0
    if any(reason.startswith("path_") for reason in unique):
        score += 1.0
    if any(reason.startswith("query_") for reason in unique):
        score += 1.0
    if "path_search_label" in unique or "path_component_tags_tag" in unique or "path_itemlist_date" in unique:
        score += 0.75
    if "query_limit_start" in unique or "query_pname_catid" in unique:
        score += 0.75
    return score, unique


def read_text(path: str | None, max_chars: int) -> tuple[str, bool, str | None]:
    if not path:
        return "", False, "missing doc_text_path"
    if not os.path.exists(path):
        return "", False, "path does not exist"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except Exception as exc:
        return "", False, str(exc)
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, truncated, None


def text_signals(text: str) -> tuple[float, list[str]]:
    if not text:
        return 0.0, []
    reasons: list[str] = []
    for pattern, reason in TEXT_SIGNAL_PATTERNS:
        hits = len(pattern.findall(text))
        if hits:
            reasons.append(f"{reason}:{hits}")
    score = 0.0
    for reason in reasons:
        if reason.startswith("text_timestamp"):
            score += min(0.75, 0.05 * int(reason.rsplit(":", 1)[-1]))
        else:
            score += min(1.0, 0.25 * int(reason.rsplit(":", 1)[-1]))
    return score, reasons


def is_reviewed(row: dict[str, Any], reviewed_by_doc: dict[str, dict[str, Any]]) -> bool:
    doc_id = row_source_doc_id(row)
    if doc_id in reviewed_by_doc:
        return True
    return bool(row.get("review_label") or row.get("prior_review_label"))


def review_outcome(row: dict[str, Any]) -> str:
    action = row.get("review_correct_action")
    label = row.get("review_label")
    severity = row.get("review_severity")
    if action:
        return f"{action}|{severity or 'unknown'}|{label or 'unknown'}"
    return f"{label or 'unknown'}"


def make_pack_row(
    row: dict[str, Any],
    text: str,
    truncated: bool,
    read_error: str | None,
    url_score: float,
    url_reasons: list[str],
    text_score: float,
    text_reasons: list[str],
    timestamp: str,
    rank: int,
) -> dict[str, Any]:
    doc_id = row_source_doc_id(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "rescue_pack_id": f"url_rescue_{timestamp}_{rank:04d}_{stable_hash(doc_id)[:8]}",
        "source_doc_id": doc_id,
        "parent_source_doc_id": row.get("parent_source_doc_id") or doc_id,
        "policy_evaluation_sample_id": row.get("policy_evaluation_sample_id"),
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "evaluation_tasks": row.get("evaluation_tasks") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "quality_bin": row.get("quality_bin"),
        "host": row.get("host"),
        "url": row.get("url"),
        "chars_before": row.get("chars_before"),
        "text_sha256_before": row.get("text_sha256_before"),
        "doc_text_path": row.get("doc_text_path"),
        "detector_scores": row.get("detector_scores") or {},
        "url_rescue_score": round(url_score, 6),
        "url_rescue_reasons": url_reasons,
        "text_rescue_score": round(text_score, 6),
        "text_rescue_reasons": text_reasons,
        "combined_rescue_score": round(url_score + text_score, 6),
        "document_text_read_error": read_error,
        "document_text_truncated": truncated,
        "document_text": text,
        "policy_note": "Review pack only. Do not use URL rescue signals as an automatic destructive policy without precision/FN/good-text-loss gates.",
    }


def make_annotation_template(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    template = []
    for row in rows:
        template.append({
            "schema_version": "hplt-serious-fn-url-rescue-annotation-v1",
            "rescue_pack_id": row["rescue_pack_id"],
            "source_doc_id": row["source_doc_id"],
            "candidate_action": row["candidate_action"],
            "url_rescue_reasons": row["url_rescue_reasons"],
            "review_label": None,
            "review_true_error_type_ids": [],
            "review_correct_action": None,
            "review_severity": None,
            "review_good_text_loss_risk": None,
            "review_span_or_split_notes": None,
            "review_notes": None,
            "reviewer": None,
        })
    return template


def markdown_excerpt(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return text[:head] + "\n\n...[SNIP]...\n\n" + text[-tail:]


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], markdown_chars: int) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("# Serious FN URL Rescue Review Pack\n\n")
        out.write("This is a non-destructive review pack. It targets candidate-keep rows whose URLs look like archive/category/tag/search/listing pages, because the closed representative FN review found serious E010/E011 misses with weak text-detector scores but strong URL structure.\n\n")
        out.write("## Summary\n\n")
        out.write(f"- selected rows: `{summary['selected_rows']}`\n")
        out.write(f"- candidate keep rows scanned: `{summary['candidate_keep_rows']}`\n")
        out.write(f"- reviewed rows excluded: `{summary['reviewed_excluded_rows']}`\n")
        out.write(f"- known serious FN URL-signal recall: `{summary['known_serious_fn_url_signal_recall']}`\n")
        out.write(f"- policy note: {summary['policy_note']}\n\n")
        out.write("## URL Signal Reasons\n\n| Reason | Rows |\n| --- | ---: |\n")
        for key, count in summary["selected_by_url_reason"].items():
            out.write(f"| `{key}` | {count} |\n")
        out.write("\n## Reviewed Calibration Hits\n\n| Outcome | Rows |\n| --- | ---: |\n")
        for key, count in summary["reviewed_url_signal_outcomes"].items():
            out.write(f"| `{key}` | {count} |\n")
        out.write("\n## Records\n\n")
        for row in rows:
            out.write(f"### `{row['rescue_pack_id']}`\n\n")
            out.write(f"- source_doc_id: `{row['source_doc_id']}`\n")
            out.write(f"- policy_evaluation_sample_id: `{row.get('policy_evaluation_sample_id')}`\n")
            out.write(f"- host: `{row.get('host')}`\n")
            out.write(f"- url: `{row.get('url')}`\n")
            out.write(f"- qbin: `{row.get('quality_bin')}`\n")
            out.write(f"- chars_before: `{row.get('chars_before')}`\n")
            out.write(f"- candidate_action: `{row.get('candidate_action')}`\n")
            out.write(f"- candidate_error_type_ids: `{', '.join(row.get('candidate_error_type_ids') or [])}`\n")
            out.write(f"- URL rescue score/reasons: `{row['url_rescue_score']}` / `{', '.join(row['url_rescue_reasons'])}`\n")
            out.write(f"- text rescue score/reasons: `{row['text_rescue_score']}` / `{', '.join(row['text_rescue_reasons'])}`\n")
            out.write(f"- doc_text_path: `{row.get('doc_text_path')}`\n")
            out.write(f"- document_text_truncated: `{row.get('document_text_truncated')}`\n")
            out.write(f"- document_text_read_error: `{row.get('document_text_read_error')}`\n\n")
            out.write("```text\n")
            out.write(markdown_excerpt(row.get("document_text") or "", markdown_chars))
            out.write("\n```\n\n")


def deterministic_trim(rows: list[dict[str, Any]], max_records: int, seed: int) -> list[dict[str, Any]]:
    if max_records <= 0 or len(rows) <= max_records:
        return rows
    high = [row for row in rows if row["combined_rescue_score"] >= 2.0]
    rest = [row for row in rows if row["combined_rescue_score"] < 2.0]
    if len(high) >= max_records:
        return high[:max_records]
    rng = random.Random(seed)
    rng.shuffle(rest)
    selected = high + rest[: max_records - len(high)]
    return sorted(selected, key=lambda row: (-row["combined_rescue_score"], row.get("host") or "", row["source_doc_id"]))


def summarize(
    rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    reviewed_by_doc: dict[str, dict[str, Any]],
    known_serious_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    reason_counts = collections.Counter()
    for row in rows:
        reason_counts.update(row.get("url_rescue_reasons") or [])
    reviewed_signal_outcomes = collections.Counter()
    reviewed_signal_docs = 0
    for row in reviewed_by_doc.values():
        url_score, reasons = url_signals(str(row.get("url") or ""))
        if url_score >= threshold and str(row.get("candidate_action") or "") == "keep":
            reviewed_signal_docs += 1
            reviewed_signal_outcomes[review_outcome(row)] += 1

    caught_serious = []
    missed_serious = []
    for row in known_serious_rows:
        url_score, reasons = url_signals(str(row.get("url") or ""))
        target = caught_serious if url_score >= threshold else missed_serious
        target.append({
            "source_doc_id": row_source_doc_id(row),
            "host": row.get("host"),
            "url": row.get("url"),
            "review_correct_action": row.get("review_correct_action"),
            "review_true_error_type_ids": row.get("review_true_error_type_ids") or [],
            "url_rescue_score": round(url_score, 6),
            "url_rescue_reasons": reasons,
            "review_notes": row.get("review_notes"),
        })

    candidate_keep_rows = [row for row in policy_rows if row.get("candidate_action") == "keep"]
    reviewed_excluded = [row for row in candidate_keep_rows if row_source_doc_id(row) in reviewed_by_doc or row.get("review_label") or row.get("prior_review_label")]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": rows[0]["created_at_utc"] if rows else utc_timestamp(),
        "policy_sample_rows": len(policy_rows),
        "candidate_keep_rows": len(candidate_keep_rows),
        "reviewed_excluded_rows": len(reviewed_excluded),
        "selected_rows": len(rows),
        "selection_threshold": threshold,
        "selected_by_url_reason": dict(reason_counts.most_common()),
        "selected_by_quality_bin": dict(collections.Counter(str(row.get("quality_bin")) for row in rows).most_common()),
        "selected_by_host_top": dict(collections.Counter(str(row.get("host")) for row in rows).most_common(30)),
        "reviewed_url_signal_docs": reviewed_signal_docs,
        "reviewed_url_signal_outcomes": dict(reviewed_signal_outcomes.most_common()),
        "known_serious_fn_rows": len(known_serious_rows),
        "known_serious_fn_url_signal_caught": len(caught_serious),
        "known_serious_fn_url_signal_missed": len(missed_serious),
        "known_serious_fn_url_signal_recall": f"{len(caught_serious)}/{len(known_serious_rows)}",
        "known_serious_fn_caught_examples": caught_serious,
        "known_serious_fn_missed_examples": missed_serious,
        "policy_note": "Review pack only. URL rescue signals may be used for targeted review and detector design, not automatic destructive cleaning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-sample", default=DEFAULT_POLICY_SAMPLE)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--max-records", type=int, default=80)
    parser.add_argument("--full-text-chars", type=int, default=0)
    parser.add_argument("--markdown-chars", type=int, default=5000)
    parser.add_argument("--url-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    policy_sample = Path(args.policy_sample)
    reports_dir = Path(args.reports_dir)
    output_prefix = Path(f"{args.output_prefix}_{timestamp}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    policy_rows = read_jsonl(policy_sample)
    ann_paths = annotation_files(reports_dir)
    reviewed_rows = all_reviews(ann_paths)
    reviewed_by_doc = latest_reviews_by_doc(ann_paths)
    known_serious_by_doc: dict[str, dict[str, Any]] = {}
    for row in reviewed_rows:
        if row.get("review_severity") != "serious" or row.get("review_label") != "false_negative_found":
            continue
        doc_id = row_source_doc_id(row)
        if doc_id:
            known_serious_by_doc[doc_id] = row
    known_serious_rows = list(known_serious_by_doc.values())

    candidates: list[dict[str, Any]] = []
    for row in policy_rows:
        if row.get("candidate_action") != "keep":
            continue
        if is_reviewed(row, reviewed_by_doc):
            continue
        url_score, url_reasons = url_signals(str(row.get("url") or ""))
        if url_score < args.url_threshold:
            continue
        text, truncated, read_error = read_text(row.get("doc_text_path"), args.full_text_chars)
        text_score, text_reasons = text_signals(text)
        pack_row = make_pack_row(
            row,
            text,
            truncated,
            read_error,
            url_score,
            url_reasons,
            text_score,
            text_reasons,
            timestamp,
            len(candidates) + 1,
        )
        candidates.append(pack_row)

    candidates.sort(key=lambda row: (-row["combined_rescue_score"], row.get("host") or "", row["source_doc_id"]))
    selected = deterministic_trim(candidates, args.max_records, args.seed)
    for index, row in enumerate(selected, 1):
        row["rescue_pack_id"] = f"url_rescue_{timestamp}_{index:04d}_{stable_hash(row['source_doc_id'])[:8]}"

    summary = summarize(selected, policy_rows, reviewed_by_doc, known_serious_rows, args.url_threshold)
    summary.update({
        "inputs": {
            "policy_sample": str(policy_sample),
            "reports_dir": str(reports_dir),
            "annotation_files": [str(path) for path in ann_paths],
        },
        "outputs": {
            "review_pack_jsonl": str(output_prefix.with_suffix(".jsonl")),
            "annotation_template_jsonl": str(Path(str(output_prefix) + "_annotation_template.jsonl")),
            "markdown": str(output_prefix.with_suffix(".md")),
            "summary_json": str(Path(str(output_prefix) + "_summary.json")),
        },
        "max_records": args.max_records,
        "full_text_chars": args.full_text_chars,
        "markdown_chars": args.markdown_chars,
        "seed": args.seed,
    })

    write_jsonl(output_prefix.with_suffix(".jsonl"), selected)
    write_jsonl(Path(str(output_prefix) + "_annotation_template.jsonl"), make_annotation_template(selected))
    write_json(Path(str(output_prefix) + "_summary.json"), summary)
    write_markdown(output_prefix.with_suffix(".md"), selected, summary, args.markdown_chars)

    print(json.dumps({
        "event": "complete",
        "selected_rows": len(selected),
        "known_serious_fn_url_signal_recall": summary["known_serious_fn_url_signal_recall"],
        "review_pack_jsonl": str(output_prefix.with_suffix(".jsonl")),
        "summary_json": str(Path(str(output_prefix) + "_summary.json")),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and summarize serious-FN URL-rescue annotations.

This script consumes the non-destructive URL-rescue review pack plus filled
manual annotations. It reports how often archive/category/tag/search/listing
URL structure found real candidate-keep false negatives. It does not mutate
source rows or approve an automatic cleaning policy.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hplt-serious-fn-url-rescue-annotation-summary-v1"
ALLOWED_ACTIONS = {
    "drop_doc",
    "keep",
    "normalize_or_trim_span",
    "quarantine",
    "split_doc",
    "trim_prefix",
    "trim_span",
    "trim_suffix",
}
ALLOWED_LABELS = {"clean", "false_negative_found", "false_positive", "partial_true_positive", "true_positive", "unclear"}
ALLOWED_SEVERITIES = {"none", "minor", "moderate", "serious", "unknown"}
ALLOWED_LOSS = {"none", "low", "low_to_medium", "medium", "high", "unknown"}


def utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def percent_interval(values: list[float]) -> list[float]:
    return [value * 100 for value in values]


def validate(pack_rows: list[dict[str, Any]], ann_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    pack_by_id = {row.get("rescue_pack_id"): row for row in pack_rows}
    seen: set[str] = set()

    for row in ann_rows:
        rid = row.get("rescue_pack_id")
        if not rid:
            issues.append({"line": row.get("_line_number"), "issue": "missing_rescue_pack_id"})
            continue
        if rid in seen:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "duplicate_annotation"})
        seen.add(rid)
        pack = pack_by_id.get(rid)
        if not pack:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "not_in_review_pack"})
            continue
        if row.get("source_doc_id") != pack.get("source_doc_id"):
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "source_doc_id_mismatch"})
        if row.get("candidate_action") != pack.get("candidate_action"):
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "candidate_action_mismatch"})
        action = row.get("review_correct_action")
        label = row.get("review_label")
        severity = row.get("review_severity")
        loss = row.get("review_good_text_loss_risk")
        errors = row.get("review_true_error_type_ids")
        if action not in ALLOWED_ACTIONS:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "invalid_review_correct_action", "value": action})
        if label not in ALLOWED_LABELS:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "invalid_review_label", "value": label})
        if severity not in ALLOWED_SEVERITIES:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "invalid_review_severity", "value": severity})
        if loss not in ALLOWED_LOSS:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "invalid_review_good_text_loss_risk", "value": loss})
        if not isinstance(errors, list):
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "review_true_error_type_ids_not_list"})
        elif action != "keep" and not errors:
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "non_keep_missing_error_ids"})
        if action == "keep" and label != "clean":
            issues.append({"line": row.get("_line_number"), "rescue_pack_id": rid, "issue": "keep_not_clean"})

    missing = sorted(set(pack_by_id) - seen)
    for rid in missing:
        issues.append({"rescue_pack_id": rid, "issue": "missing_annotation"})
    return issues


def row_actionable(row: dict[str, Any]) -> bool:
    return row.get("review_correct_action") != "keep"


def make_summary(pack_rows: list[dict[str, Any]], ann_rows: list[dict[str, Any]], issues: list[dict[str, Any]], args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    rows = [row for row in ann_rows if row.get("rescue_pack_id")]
    n = len(rows)
    actionable = [row for row in rows if row_actionable(row)]
    serious = [row for row in rows if row_actionable(row) and row.get("review_severity") == "serious"]
    clean = [row for row in rows if row.get("review_correct_action") == "keep"]

    by_url_reason: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    pack_by_id = {row.get("rescue_pack_id"): row for row in pack_rows}
    for row in rows:
        pack = pack_by_id.get(row.get("rescue_pack_id")) or {}
        outcome = row.get("review_correct_action") or "unknown"
        for reason in pack.get("url_rescue_reasons") or row.get("url_rescue_reasons") or []:
            by_url_reason[reason][outcome] += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "inputs": {
            "review_pack": str(args.review_pack),
            "annotations": str(args.annotations),
        },
        "review_pack_rows": len(pack_rows),
        "annotation_rows": n,
        "validation_issue_count": len(issues),
        "validation_issues": issues,
        "review_action_counts": dict(collections.Counter(row.get("review_correct_action") for row in rows).most_common()),
        "review_label_counts": dict(collections.Counter(row.get("review_label") for row in rows).most_common()),
        "review_severity_counts": dict(collections.Counter(row.get("review_severity") for row in rows).most_common()),
        "review_good_text_loss_counts": dict(collections.Counter(row.get("review_good_text_loss_risk") for row in rows).most_common()),
        "review_true_error_type_counts": dict(collections.Counter(error for row in rows for error in row.get("review_true_error_type_ids") or []).most_common()),
        "url_rescue_actionable_rows": len(actionable),
        "url_rescue_clean_rows": len(clean),
        "url_rescue_serious_rows": len(serious),
        "url_rescue_actionable_rate_percent": (len(actionable) / n * 100) if n else 0.0,
        "url_rescue_actionable_wilson_95_ci_percent": percent_interval(wilson_interval(len(actionable), n)),
        "url_rescue_clean_rate_percent": (len(clean) / n * 100) if n else 0.0,
        "url_rescue_clean_wilson_95_ci_percent": percent_interval(wilson_interval(len(clean), n)),
        "url_rescue_serious_rate_percent": (len(serious) / n * 100) if n else 0.0,
        "url_rescue_serious_wilson_95_ci_percent": percent_interval(wilson_interval(len(serious), n)),
        "by_url_reason_action": {reason: dict(counter.most_common()) for reason, counter in sorted(by_url_reason.items())},
        "policy_interpretation": "URL-shaped archive/category/tag/search/listing pages are a high-yield targeted review signal for candidate-keep false negatives, but reviewed clean URL false positives remain too common for automatic destructive action.",
        "policy_note": "Annotation summary only. It does not mutate source HPLT rows or approve an automatic cleaning overlay.",
    }
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("# Serious FN URL-Rescue Annotation Summary\n\n")
        out.write("This is a reviewed evidence summary. It does not mutate source HPLT rows and does not approve an automatic cleaning overlay.\n\n")
        out.write("## Summary\n\n")
        out.write(f"- review pack rows: `{summary['review_pack_rows']}`\n")
        out.write(f"- annotation rows: `{summary['annotation_rows']}`\n")
        out.write(f"- validation issues: `{summary['validation_issue_count']}`\n")
        out.write(f"- actionable URL-rescue rows: `{summary['url_rescue_actionable_rows']}` ({summary['url_rescue_actionable_rate_percent']:.2f}%)\n")
        out.write(f"- serious URL-rescue rows: `{summary['url_rescue_serious_rows']}` ({summary['url_rescue_serious_rate_percent']:.2f}%)\n")
        out.write(f"- clean URL false positives: `{summary['url_rescue_clean_rows']}` ({summary['url_rescue_clean_rate_percent']:.2f}%)\n")
        out.write(f"- interpretation: {summary['policy_interpretation']}\n\n")

        out.write("## Actions\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["review_action_counts"].items():
            out.write(f"| `{action}` | {count} |\n")
        out.write("\n## True Error IDs\n\n| Error | Rows |\n| --- | ---: |\n")
        for error, count in summary["review_true_error_type_counts"].items():
            out.write(f"| `{error}` | {count} |\n")
        out.write("\n## URL Reason By Reviewed Action\n\n")
        for reason, counts in summary["by_url_reason_action"].items():
            out.write(f"### `{reason}`\n\n| Reviewed action | Rows |\n| --- | ---: |\n")
            for action, count in counts.items():
                out.write(f"| `{action}` | {count} |\n")
            out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    output_prefix = Path(f"{args.output_prefix}_{timestamp}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pack_rows = read_jsonl(args.review_pack)
    ann_rows = read_jsonl(args.annotations)
    issues = validate(pack_rows, ann_rows)
    summary = make_summary(pack_rows, ann_rows, issues, args, timestamp)
    write_json(Path(str(output_prefix) + "_summary.json"), summary)
    write_markdown(Path(str(output_prefix) + ".md"), summary)

    print(json.dumps({
        "event": "complete",
        "annotation_rows": summary["annotation_rows"],
        "validation_issue_count": summary["validation_issue_count"],
        "url_rescue_actionable_rows": summary["url_rescue_actionable_rows"],
        "url_rescue_serious_rows": summary["url_rescue_serious_rows"],
        "summary_json": str(Path(str(output_prefix) + "_summary.json")),
        "markdown": str(Path(str(output_prefix) + ".md")),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

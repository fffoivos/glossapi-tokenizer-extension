#!/usr/bin/env python3
"""Validate and summarize reviewed global-prevalence annotations.

This consumes a `global_prevalence_review_pack_*.jsonl` plus one or more filled
annotation JSONL files. It reports prevalence-review progress and confidence
intervals without mutating source HPLT rows or creating a cleaning overlay.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


VALID_LABELS = {
    "true_positive",
    "partial_true_positive",
    "false_positive",
    "false_negative_found",
    "clean",
    "unclear",
}

VALID_ACTIONS = {
    "keep",
    "normalize_or_trim_span",
    "trim_prefix",
    "trim_suffix",
    "trim_span",
    "split_doc",
    "drop_doc",
    "quarantine",
}

VALID_SEVERITIES = {"none", "minor", "moderate", "serious", "unclear"}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def source_doc_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("policy_evaluation_sample_id") or "")


def wilson(successes: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n))) / denom
    return [max(0.0, center - half) * 100.0, min(1.0, center + half) * 100.0]


def is_actionable(row: dict[str, Any]) -> bool:
    action = str(row.get("review_correct_action") or "")
    label = str(row.get("review_label") or "")
    return action != "keep" or label in {"true_positive", "partial_true_positive", "false_negative_found"}


def error_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("review_true_error_type_ids") or row.get("review_false_negative_error_ids") or []
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if str(item).startswith("E"))


def validate_annotation(row: dict[str, Any], pack_by_doc: dict[str, dict[str, Any]], source_path: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    doc_id = source_doc_id(row)
    if not doc_id:
        issues.append({"source_doc_id": doc_id, "issue": "missing_source_doc_id", "annotation_file": source_path})
        return issues
    if doc_id not in pack_by_doc:
        issues.append({"source_doc_id": doc_id, "issue": "source_doc_id_not_in_pack", "annotation_file": source_path})
        return issues

    label = row.get("review_label")
    action = row.get("review_correct_action")
    severity = row.get("review_severity")
    if label not in VALID_LABELS:
        issues.append({"source_doc_id": doc_id, "issue": "invalid_or_missing_review_label", "value": label, "annotation_file": source_path})
    if action not in VALID_ACTIONS:
        issues.append({"source_doc_id": doc_id, "issue": "invalid_or_missing_review_correct_action", "value": action, "annotation_file": source_path})
    if severity not in VALID_SEVERITIES:
        issues.append({"source_doc_id": doc_id, "issue": "invalid_or_missing_review_severity", "value": severity, "annotation_file": source_path})
    if action and action != "keep" and not error_ids(row):
        issues.append({"source_doc_id": doc_id, "issue": "changing_action_missing_error_ids", "action": action, "annotation_file": source_path})
    return issues


def counter_dict(rows: list[dict[str, Any]], key_fn) -> dict[str, int]:
    return dict(collections.Counter(str(key_fn(row)) for row in rows).most_common())


def by_stratum(rows: list[dict[str, Any]], annotations_by_doc: dict[str, dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        doc_id = source_doc_id(row)
        if doc_id not in annotations_by_doc:
            continue
        stratum = str(row.get(key) or "unknown")
        bucket = result.setdefault(stratum, {"reviewed": 0, "actionable": 0, "serious": 0})
        ann = annotations_by_doc[doc_id]
        bucket["reviewed"] += 1
        if is_actionable(ann):
            bucket["actionable"] += 1
        if ann.get("review_severity") == "serious":
            bucket["serious"] += 1
    return result


def make_summary(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pack_path = Path(args.review_pack)
    pack_rows = read_jsonl(pack_path)
    pack_by_doc = {source_doc_id(row): row for row in pack_rows}

    annotations_by_doc: dict[str, dict[str, Any]] = {}
    duplicate_annotations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    annotation_files: list[dict[str, Any]] = []
    for path_text in args.annotations:
        path = Path(path_text)
        rows = read_jsonl(path)
        annotation_files.append({"path": str(path), "rows": len(rows)})
        for row in rows:
            doc_id = source_doc_id(row)
            if doc_id in annotations_by_doc:
                duplicate_annotations.append({"source_doc_id": doc_id, "first_seen": annotations_by_doc[doc_id].get("_annotation_file"), "duplicate": str(path)})
                continue
            annotated = dict(row)
            annotated["_annotation_file"] = str(path)
            annotations_by_doc[doc_id] = annotated
            issues.extend(validate_annotation(annotated, pack_by_doc, str(path)))

    reviewed_docs = [doc_id for doc_id in annotations_by_doc if doc_id in pack_by_doc]
    reviewed_annotations = [annotations_by_doc[doc_id] for doc_id in reviewed_docs]
    actionable = sum(1 for row in reviewed_annotations if is_actionable(row))
    serious = sum(1 for row in reviewed_annotations if row.get("review_severity") == "serious")
    unclear = sum(1 for row in reviewed_annotations if row.get("review_label") == "unclear")

    base_summary = read_json(Path(args.base_summary)) if args.base_summary else {}
    base_reviewed = int(base_summary.get("reviewed_unique_rows") or 0)
    base_actionable = int(base_summary.get("reviewed_actionable_rows") or 0)
    base_serious = int(base_summary.get("reviewed_serious_rows") or 0)
    combined_reviewed = base_reviewed + len(reviewed_annotations)
    combined_actionable = base_actionable + actionable
    combined_serious = base_serious + serious

    error_counter: collections.Counter[str] = collections.Counter()
    for row in reviewed_annotations:
        error_counter.update(error_ids(row))

    summary = {
        "schema_version": "hplt-global-prevalence-annotation-summary-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Annotation summary only. It does not mutate source HPLT rows or approve an automatic cleaning overlay.",
        "inputs": {
            "review_pack": str(pack_path),
            "base_summary": str(args.base_summary) if args.base_summary else None,
            "annotation_files": annotation_files,
        },
        "pack_rows": len(pack_rows),
        "annotation_rows": sum(item["rows"] for item in annotation_files),
        "reviewed_unique_rows_in_pack": len(reviewed_annotations),
        "unreviewed_rows_in_pack": len(pack_rows) - len(reviewed_annotations),
        "validation_issue_count": len(issues),
        "duplicate_annotation_count": len(duplicate_annotations),
        "reviewed_actionable_rows_in_pack": actionable,
        "reviewed_serious_rows_in_pack": serious,
        "reviewed_unclear_rows_in_pack": unclear,
        "actionable_rate_in_pack_percent": (actionable / len(reviewed_annotations) * 100.0) if reviewed_annotations else 0.0,
        "actionable_wilson_95_ci_percent": wilson(actionable, len(reviewed_annotations)),
        "serious_rate_in_pack_percent": (serious / len(reviewed_annotations) * 100.0) if reviewed_annotations else 0.0,
        "serious_wilson_95_ci_percent": wilson(serious, len(reviewed_annotations)),
        "combined_with_base": {
            "base_reviewed_unique_rows": base_reviewed,
            "base_actionable_rows": base_actionable,
            "base_serious_rows": base_serious,
            "combined_reviewed_unique_rows": combined_reviewed,
            "combined_actionable_rows": combined_actionable,
            "combined_serious_rows": combined_serious,
            "combined_actionable_rate_percent": (combined_actionable / combined_reviewed * 100.0) if combined_reviewed else 0.0,
            "combined_actionable_wilson_95_ci_percent": wilson(combined_actionable, combined_reviewed),
            "combined_serious_rate_percent": (combined_serious / combined_reviewed * 100.0) if combined_reviewed else 0.0,
            "combined_serious_wilson_95_ci_percent": wilson(combined_serious, combined_reviewed),
        },
        "review_label_counts": counter_dict(reviewed_annotations, lambda row: row.get("review_label")),
        "review_action_counts": counter_dict(reviewed_annotations, lambda row: row.get("review_correct_action")),
        "review_severity_counts": counter_dict(reviewed_annotations, lambda row: row.get("review_severity")),
        "review_true_error_type_counts": dict(error_counter.most_common()),
        "by_quality_bin": by_stratum(pack_rows, annotations_by_doc, "quality_bin"),
        "by_risk_band": by_stratum(pack_rows, annotations_by_doc, "risk_band"),
        "by_length_bucket": by_stratum(pack_rows, annotations_by_doc, "length_bucket"),
        "by_candidate_action": by_stratum(pack_rows, annotations_by_doc, "candidate_action"),
        "validation_issues": issues[:200],
        "duplicate_annotations": duplicate_annotations[:200],
    }
    return summary, issues


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Global Prevalence Annotation Summary\n\n")
        out.write("This summarizes reviewed annotations for the global-prevalence review pack. It is non-destructive and does not approve a full overlay.\n\n")
        out.write("## Status\n\n")
        out.write("- reviewed unique rows in pack: `%s` / `%s`\n" % (summary["reviewed_unique_rows_in_pack"], summary["pack_rows"]))
        out.write("- validation issues: `%s`\n" % summary["validation_issue_count"])
        out.write("- duplicate annotations: `%s`\n" % summary["duplicate_annotation_count"])
        out.write("- actionable rows in reviewed pack slice: `%s`, rate `%.2f%%`, Wilson 95%% CI `%s`\n" % (
            summary["reviewed_actionable_rows_in_pack"],
            summary["actionable_rate_in_pack_percent"],
            summary["actionable_wilson_95_ci_percent"],
        ))
        out.write("- serious rows in reviewed pack slice: `%s`, rate `%.2f%%`, Wilson 95%% CI `%s`\n\n" % (
            summary["reviewed_serious_rows_in_pack"],
            summary["serious_rate_in_pack_percent"],
            summary["serious_wilson_95_ci_percent"],
        ))

        combined = summary["combined_with_base"]
        out.write("## Combined With Prior Coverage\n\n")
        out.write("- combined reviewed unique rows: `%s`\n" % combined["combined_reviewed_unique_rows"])
        out.write("- combined actionable rows: `%s`, rate `%.2f%%`, Wilson 95%% CI `%s`\n" % (
            combined["combined_actionable_rows"],
            combined["combined_actionable_rate_percent"],
            combined["combined_actionable_wilson_95_ci_percent"],
        ))
        out.write("- combined serious rows: `%s`, rate `%.2f%%`, Wilson 95%% CI `%s`\n\n" % (
            combined["combined_serious_rows"],
            combined["combined_serious_rate_percent"],
            combined["combined_serious_wilson_95_ci_percent"],
        ))

        out.write("## Counts\n\n")
        for name in ("review_label_counts", "review_action_counts", "review_severity_counts", "review_true_error_type_counts"):
            out.write(f"- `{name}`: `{summary.get(name)}`\n")
        out.write("\n")

        if summary["validation_issue_count"]:
            out.write("## Validation Issues\n\n")
            for issue in summary["validation_issues"]:
                out.write("- `%s`: `%s`\n" % (issue.get("source_doc_id"), issue.get("issue")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", required=True)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--base-summary")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    if not output_prefix.name.endswith(args.timestamp):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    summary, issues = make_summary(args)
    json_path = Path(str(output_prefix) + "_summary.json")
    md_path = Path(str(output_prefix) + ".md")
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(json.dumps({"summary": str(json_path), "markdown": str(md_path), "validation_issues": len(issues), "reviewed": summary["reviewed_unique_rows_in_pack"]}, sort_keys=True))
    if args.fail_on_issues and issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

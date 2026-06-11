#!/usr/bin/env python3
"""Build the remaining global-random false-negative review pack.

This is bookkeeping for finishing the representative false-negative estimate:
it lists the 120 global-random controls outside the first deterministic
80-row sample, marks the subset already reviewed by the high-risk pass, and
identifies the rows that still need manual review. It is non-destructive.
"""

from __future__ import print_function

import argparse
import collections
import json
import os


CREATED_AT_UTC = "2026-06-06T01:45:00Z"
SCHEMA_VERSION = "hplt-false-negative-global-remaining-pack-v1"
DEFAULT_TRIAGE = "reports/false_negative_control_triage_20260606T000000Z.jsonl"
DEFAULT_REPRESENTATIVE = "reports/false_negative_representative_review_annotations_20260606T011500Z.jsonl"
DEFAULT_HIGH_RISK = "reports/false_negative_triage_review_annotations_20260606T004500Z.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/false_negative_global_remaining_review_pack_20260606T014500Z"


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


def by_policy_id(rows):
    out = {}
    for row in rows:
        out[row["policy_evaluation_sample_id"]] = row
    return out


def triage_bucket(score):
    score = score or 0.0
    if score >= 0.24:
        return ">=0.24"
    if score >= 0.18:
        return ">=0.18"
    if score > 0.0:
        return ">0"
    return "0"


def make_pack_row(row, existing_review):
    if existing_review:
        review_status = "already_reviewed_high_risk"
        review_label = existing_review.get("review_label")
        review_correct_action = existing_review.get("review_correct_action")
        review_true_error_type_ids = existing_review.get("review_true_error_type_ids") or []
        review_notes = existing_review.get("review_notes")
        review_source = DEFAULT_HIGH_RISK
    else:
        review_status = "needs_review"
        review_label = None
        review_correct_action = None
        review_true_error_type_ids = []
        review_notes = None
        review_source = None

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": CREATED_AT_UTC,
        "sample_design": "census of the 120 global_random controls outside the preliminary 80-row representative review sample",
        "source_triage": DEFAULT_TRIAGE,
        "source_representative_review": DEFAULT_REPRESENTATIVE,
        "source_high_risk_review": DEFAULT_HIGH_RISK,
        "review_status": review_status,
        "review_source": review_source,
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
        "text_sha256_before": row.get("text_sha256_before"),
        "chars_before": row.get("chars_before"),
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "sample_set": row.get("sample_set") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "triage_score": row.get("triage_score"),
        "triage_bucket": triage_bucket(row.get("triage_score")),
        "triage_reasons": row.get("triage_reasons") or [],
        "detector_scores": {
            "triage_score": row.get("triage_score"),
            "word_count": row.get("word_count"),
            "greek_word_share": row.get("greek_word_share"),
            "short_line_density": row.get("short_line_density"),
            "list_line_density": row.get("list_line_density"),
            "url_hits": row.get("url_hits"),
            "comment_hits": row.get("comment_hits"),
            "related_hits": row.get("related_hits"),
            "feed_hits": row.get("feed_hits"),
            "date_boundary_hits": row.get("date_boundary_hits"),
        },
        "evidence_excerpt": row.get("review_excerpt") or row.get("evidence_excerpt"),
        "review_label": review_label,
        "review_correct_action": review_correct_action,
        "review_true_error_type_ids": review_true_error_type_ids,
        "review_notes": review_notes,
    }


def make_summary(pack_rows, global_population_size, representative_reviewed, global_reviewed_unique):
    status_counts = collections.Counter(row["review_status"] for row in pack_rows)
    bucket_counts = collections.Counter(row["triage_bucket"] for row in pack_rows)
    host_counts = collections.Counter(row.get("host") for row in pack_rows)
    existing_action_counts = collections.Counter(
        row["review_correct_action"]
        for row in pack_rows
        if row.get("review_correct_action")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": CREATED_AT_UTC,
        "global_random_population_size": global_population_size,
        "representative_reviewed_rows": representative_reviewed,
        "remaining_outside_representative_sample": len(pack_rows),
        "global_random_reviewed_unique_rows": global_reviewed_unique,
        "unreviewed_global_random_rows": status_counts.get("needs_review", 0),
        "already_reviewed_high_risk_rows": status_counts.get("already_reviewed_high_risk", 0),
        "review_status_counts": dict(status_counts),
        "triage_bucket_counts": dict(bucket_counts),
        "existing_review_correct_action_counts": dict(existing_action_counts),
        "top_hosts": [
            {"host": host, "rows": count}
            for host, count in host_counts.most_common(20)
        ],
        "policy_note": "Review pack only. It does not approve an automatic cleaning policy or mutate source rows.",
    }


def write_markdown(path, pack_rows, summary):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Remaining Global-Random False-Negative Review Pack\n\n")
        out.write("This pack is a census of the `global_random` false-negative controls outside the first 80-row representative review sample. It is non-destructive and was used to finish the representative false-negative estimate.\n\n")
        out.write("## Coverage\n\n")
        out.write("- global-random population: `%d`\n" % summary["global_random_population_size"])
        out.write("- already in preliminary representative review: `%d`\n" % summary["representative_reviewed_rows"])
        out.write("- outside that sample: `%d`\n" % summary["remaining_outside_representative_sample"])
        out.write("- outside-sample rows already reviewed by high-risk pass: `%d`\n" % summary["already_reviewed_high_risk_rows"])
        out.write("- outside-sample rows marked for full-text review: `%d`\n" % summary["unreviewed_global_random_rows"])
        out.write("- unique global-random rows reviewed so far: `%d`\n\n" % summary["global_random_reviewed_unique_rows"])
        out.write("Use the `needs_review` rows as the materialization input, then combine them with the preliminary representative review and the `already_reviewed_high_risk` rows to estimate false negatives over all 200 global-random controls.\n\n")

        out.write("## Triage Buckets\n\n| Bucket | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["triage_bucket_counts"].items()):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n")

        out.write("## Existing High-Risk Reviews In This Pack\n\n| Reviewed action | Rows |\n| --- | ---: |\n")
        if summary["existing_review_correct_action_counts"]:
            for key, count in sorted(summary["existing_review_correct_action_counts"].items(), key=lambda item: (-item[1], item[0])):
                out.write("| `%s` | %d |\n" % (key, count))
        else:
            out.write("| none | 0 |\n")
        out.write("\n")

        out.write("## Highest-Score Rows Still Needing Review\n\n")
        out.write("| Pack ID | Host | Score | Reasons | Excerpt |\n")
        out.write("| --- | --- | ---: | --- | --- |\n")
        needs_review = [row for row in pack_rows if row["review_status"] == "needs_review"]
        needs_review.sort(key=lambda row: (-(row.get("triage_score") or 0.0), row.get("host") or "", row["policy_evaluation_sample_id"]))
        for row in needs_review[:40]:
            reasons = ", ".join(row.get("triage_reasons") or [])
            excerpt = (row.get("evidence_excerpt") or "").replace("\n", " ")
            if len(excerpt) > 180:
                excerpt = excerpt[:177] + "..."
            out.write("| `%s` | `%s` | %.2f | `%s` | %s |\n" % (
                row["policy_evaluation_sample_id"],
                row.get("host") or "",
                row.get("triage_score") or 0.0,
                reasons,
                excerpt.replace("|", "\\|"),
            ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", default=DEFAULT_TRIAGE)
    parser.add_argument("--representative", default=DEFAULT_REPRESENTATIVE)
    parser.add_argument("--high-risk", default=DEFAULT_HIGH_RISK)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    global_rows = [
        row for row in read_jsonl(args.triage)
        if row.get("sample_set") == ["global_random"]
    ]
    representative_by_id = by_policy_id(read_jsonl(args.representative))
    high_risk_by_id = by_policy_id(read_jsonl(args.high_risk))

    remaining_rows = [
        row for row in global_rows
        if row["policy_evaluation_sample_id"] not in representative_by_id
    ]
    pack_rows = [
        make_pack_row(row, high_risk_by_id.get(row["policy_evaluation_sample_id"]))
        for row in remaining_rows
    ]
    pack_rows.sort(key=lambda row: (
        0 if row["review_status"] == "needs_review" else 1,
        -(row.get("triage_score") or 0.0),
        row.get("host") or "",
        row["policy_evaluation_sample_id"],
    ))

    global_reviewed_unique = len(set(representative_by_id) | (set(high_risk_by_id) & set(row["policy_evaluation_sample_id"] for row in global_rows)))
    summary = make_summary(
        pack_rows,
        len(global_rows),
        len(representative_by_id),
        global_reviewed_unique,
    )

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_jsonl(args.output_prefix + ".jsonl", pack_rows)
    with open(args.output_prefix + "_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(args.output_prefix + ".md", pack_rows, summary)

    print("wrote", args.output_prefix + ".jsonl")
    print("rows", len(pack_rows))
    print("needs_review", summary["unreviewed_global_random_rows"])


if __name__ == "__main__":
    main()

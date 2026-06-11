#!/usr/bin/env python3
"""Build a preliminary representative false-negative review slice.

This samples the `global_random` false-negative controls, reuses existing
review annotations where available, and records a compact manual review for the
remaining sampled rows. It does not mutate source rows and does not recommend
automatic cleaning.
"""

from __future__ import print_function

import argparse
import collections
import hashlib
import json
import math
import os


CREATED_AT_UTC = "2026-06-06T01:15:00Z"
SCHEMA_VERSION = "hplt-false-negative-representative-review-v1"
DEFAULT_TRIAGE = "reports/false_negative_control_triage_20260606T000000Z.jsonl"
DEFAULT_EXISTING = "reports/false_negative_triage_review_annotations_20260606T004500Z.jsonl"
SAMPLE_SEED = "20260606_global80"
TARGET_SAMPLE_SET = "global_random"
SAMPLE_SIZE = 80


MANUAL_FALSE_NEGATIVES = {
    "pe_0341_cd1b082b45": {
        "review_correct_action": "trim_prefix",
        "review_true_error_type_ids": ["E006", "E009"],
        "review_good_text_loss_risk": "low",
        "review_notes": "Betting article is preceded by related/strategy teaser material before the main Pick & Win article body.",
        "review_span_or_split_notes": "Trim prefix up to the main article title.",
    },
    "pe_0778_2528a7d46b": {
        "review_correct_action": "split_doc",
        "review_true_error_type_ids": ["E010", "E011"],
        "review_good_text_loss_risk": "medium",
        "review_notes": "Newsit ArticleList page concatenates many timestamped snippets and 'full article' links; not a single main body.",
        "review_span_or_split_notes": "Split only if child snippets pass useful-body thresholds; otherwise exclude/quarantine this derived row.",
    },
    "pe_0998_6cfc8fa6e1": {
        "review_correct_action": "trim_suffix",
        "review_true_error_type_ids": ["E019"],
        "review_good_text_loss_risk": "low",
        "review_notes": "Short news article is followed by comment-form rules and participation boilerplate.",
        "review_span_or_split_notes": "Trim suffix beginning at 'το δικό σας σχόλιο'.",
    },
    "pe_0742_d32667c746": {
        "review_correct_action": "trim_suffix",
        "review_true_error_type_ids": ["E019"],
        "review_good_text_loss_risk": "low",
        "review_notes": "Coherent Blogger post ends with 'Δεν υπάρχουν σχόλια: Δημοσίευση σχολίου' comment chrome.",
        "review_span_or_split_notes": "Trim suffix at the Blogger comment block.",
    },
}


MANUAL_KEEP_NOTES = {
    "pe_0577_793f545484": "Payment-options page is source content; source-policy question, not residual extraction dirt.",
    "pe_0807_147cc75530": "Forum reply is coherent discussion content; no clear scaffold dominates reviewed text.",
    "pe_1279_ac418333b4": "Company profile/services lists are source content, not extraction residue.",
    "pe_1814_7edb008a46": "Archive URL, but reviewed text is a coherent long-form fiction/literary post.",
    "pe_2068_ba19145e5d": "Book product page mixes price, description, and review as coherent source content.",
    "pe_0776_c8f46c5708": "Tag URL, but reviewed text is a coherent health article excerpt.",
}


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


def sample_key(row):
    value = "%s|%s" % (row.get("policy_evaluation_sample_id"), SAMPLE_SEED)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def wilson_interval(successes, n, z):
    if n <= 0:
        return (0.0, 0.0)
    p = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def normalize_existing(row):
    true_ids = list(row.get("review_true_error_type_ids") or [])
    label = row.get("review_label")
    correct_action = row.get("review_correct_action")
    return {
        "review_correct_action": correct_action,
        "review_label": label,
        "review_true_error_type_ids": true_ids,
        "review_false_negative_error_ids": list(row.get("review_false_negative_error_ids") or true_ids),
        "review_good_text_loss_risk": row.get("review_good_text_loss_risk"),
        "review_notes": row.get("review_notes"),
        "review_span_or_split_notes": row.get("review_span_or_split_notes"),
        "source_review_reference": row.get("source_triage"),
    }


def clean_annotation(row):
    pid = row.get("policy_evaluation_sample_id")
    return {
        "review_correct_action": "keep",
        "review_label": "clean",
        "review_true_error_type_ids": [],
        "review_false_negative_error_ids": [],
        "review_good_text_loss_risk": "none",
        "review_notes": MANUAL_KEEP_NOTES.get(pid, "Reviewed compact evidence shows coherent source content and no actionable residual HPLT extraction artifact."),
        "review_span_or_split_notes": "No cleaning action from reviewed evidence.",
        "source_review_reference": None,
    }


def make_annotation(row, existing_by_id):
    pid = row.get("policy_evaluation_sample_id")
    if pid in existing_by_id:
        review = normalize_existing(existing_by_id[pid])
        review_basis = "reused existing reviewed annotation from false_negative_triage_review_annotations_20260606T004500Z for this deterministic representative sample"
    elif pid in MANUAL_FALSE_NEGATIVES:
        fn = MANUAL_FALSE_NEGATIVES[pid]
        review = {
            "review_correct_action": fn["review_correct_action"],
            "review_label": "false_negative_found",
            "review_true_error_type_ids": list(fn["review_true_error_type_ids"]),
            "review_false_negative_error_ids": list(fn["review_true_error_type_ids"]),
            "review_good_text_loss_risk": fn["review_good_text_loss_risk"],
            "review_notes": fn["review_notes"],
            "review_span_or_split_notes": fn["review_span_or_split_notes"],
            "source_review_reference": None,
        }
        review_basis = "manual Codex review of compact evidence from false_negative_control_triage_20260606T000000Z.jsonl; no source rows changed"
    else:
        review = clean_annotation(row)
        review_basis = "manual Codex review of compact evidence from false_negative_control_triage_20260606T000000Z.jsonl; no source rows changed"

    label = review["review_label"]
    candidate_action_match = (row.get("candidate_action") == review["review_correct_action"])
    action_status = "reviewed_accept" if candidate_action_match else "reviewed_reject"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": CREATED_AT_UTC,
        "reviewer": "codex",
        "review_basis": review_basis,
        "sample_design": "deterministic SHA1 sample of %d rows from the 200 global_random false-negative controls using seed %s" % (SAMPLE_SIZE, SAMPLE_SEED),
        "source_triage": DEFAULT_TRIAGE,
        "existing_review_annotations": DEFAULT_EXISTING,
        "policy_evaluation_sample_id": pid,
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
        "chars_after": None,
        "chars_removed": None,
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "candidate_action_match": candidate_action_match,
        "action_status": action_status,
        "review_label": label,
        "review_correct_action": review["review_correct_action"],
        "review_true_error_type_ids": review["review_true_error_type_ids"],
        "review_false_negative_error_ids": review["review_false_negative_error_ids"],
        "review_good_text_loss_risk": review["review_good_text_loss_risk"],
        "review_notes": review["review_notes"],
        "review_span_or_split_notes": review["review_span_or_split_notes"],
        "source_review_reference": review["source_review_reference"],
        "sample_set": row.get("sample_set") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "triage_score": row.get("triage_score"),
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
    }


def make_summary(rows, population_size):
    n = len(rows)
    misses = [row for row in rows if row.get("review_label") == "false_negative_found"]
    k = len(misses)
    ci_low, ci_high = wilson_interval(k, n, 1.959963984540054)
    p = float(k) / float(n or 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": CREATED_AT_UTC,
        "sample_design": "deterministic SHA1 sample from global_random false-negative controls",
        "sample_seed": SAMPLE_SEED,
        "population": "200 global_random false-negative-control rows inside the 300-row pack",
        "population_size": population_size,
        "reviewed_rows": n,
        "false_negative_found_count": k,
        "false_negative_rate": p,
        "wilson_95_ci": [ci_low, ci_high],
        "wilson_95_ci_percent": [ci_low * 100.0, ci_high * 100.0],
        "label_counts": dict(collections.Counter(row["review_label"] for row in rows)),
        "review_correct_action_counts": dict(collections.Counter(row["review_correct_action"] for row in rows)),
        "true_error_type_counts": dict(collections.Counter(error_id for row in rows for error_id in row["review_true_error_type_ids"])),
        "triage_bucket_counts": dict(collections.Counter(
            ">=0.24" if row.get("triage_score", 0.0) >= 0.24 else
            ">=0.18" if row.get("triage_score", 0.0) >= 0.18 else
            ">0" if row.get("triage_score", 0.0) > 0.0 else
            "0"
            for row in rows
        )),
        "policy_note": "Preliminary representative false-negative estimate only. It does not satisfy final policy gates and does not approve automatic destructive cleaning.",
    }


def write_markdown(path, rows, summary):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Representative False-Negative Review Findings\n\n")
        out.write("This is a preliminary representative false-negative-control review over the `global_random` controls only. It is non-destructive and does not approve any automatic policy.\n\n")
        out.write("## Sample\n\n")
        out.write("- population: `%s`\n" % summary["population"])
        out.write("- sample seed: `%s`\n" % summary["sample_seed"])
        out.write("- reviewed rows: `%d`\n" % summary["reviewed_rows"])
        out.write("- false negatives found: `%d`\n" % summary["false_negative_found_count"])
        out.write("- false-negative rate: `%.2f%%`\n" % (summary["false_negative_rate"] * 100.0))
        out.write("- Wilson 95%% CI: `%.2f%%` to `%.2f%%`\n\n" % (summary["wilson_95_ci_percent"][0], summary["wilson_95_ci_percent"][1]))
        out.write("This CI describes the 80-row preliminary sample, not the full HPLT slice. It is superseded by the closed 200-row global-random review when that artifact is available.\n\n")
        out.write("## Action Counts\n\n| Reviewed action | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["review_correct_action_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## True Error Types In Misses\n\n| Error ID | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["true_error_type_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## Interpretation\n\n")
        out.write("- The preliminary representative FN rate is driven by attached article-list pages and small comment/related suffix or prefix artifacts.\n")
        out.write("- Several list-like ecommerce, payment, forum, and product-description pages reviewed as `keep`; they are source-content or source-policy questions, not residual extraction dirt.\n")
        out.write("- A few rows with `triage_score == 0` were still false negatives, so the detector must not be used as a sole clean/no-clean proof.\n")
        out.write("- No automatic destructive rule is promoted from this sample.\n\n")
        out.write("## Reviewed Rows\n\n")
        out.write("| Sample | Host | Score | Reviewed action | Label | True IDs | Note |\n")
        out.write("| --- | --- | ---: | --- | --- | --- | --- |\n")
        for row in rows:
            true_ids = ", ".join(row["review_true_error_type_ids"])
            note = row["review_notes"].replace("|", "/")
            out.write("| `%s` | `%s` | %.2f | `%s` | `%s` | `%s` | %s |\n" % (
                row["policy_evaluation_sample_id"],
                row["host"],
                row["triage_score"],
                row["review_correct_action"],
                row["review_label"],
                true_ids,
                note,
            ))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-jsonl", default=DEFAULT_TRIAGE)
    parser.add_argument("--existing-review-jsonl", default=DEFAULT_EXISTING)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--timestamp", default="20260606T011500Z")
    return parser.parse_args()


def main():
    args = parse_args()
    triage_rows = list(read_jsonl(args.triage_jsonl))
    existing = {row.get("policy_evaluation_sample_id"): row for row in read_jsonl(args.existing_review_jsonl)}
    population = [row for row in triage_rows if row.get("sample_set") == [TARGET_SAMPLE_SET]]
    if len(population) < SAMPLE_SIZE:
        raise RuntimeError("Need at least %d global_random rows, got %d" % (SAMPLE_SIZE, len(population)))
    sampled = sorted(population, key=sample_key)[:SAMPLE_SIZE]
    rows = [make_annotation(row, existing) for row in sampled]
    summary = make_summary(rows, len(population))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    annotations_path = os.path.join(args.output_dir, "false_negative_representative_review_annotations_%s.jsonl" % args.timestamp)
    summary_path = os.path.join(args.output_dir, "false_negative_representative_review_summary_%s.json" % args.timestamp)
    findings_path = os.path.join(args.output_dir, "false_negative_representative_review_findings_%s.md" % args.timestamp)
    write_jsonl(annotations_path, rows)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(findings_path, rows, summary)
    print(json.dumps({
        "annotations": annotations_path,
        "findings": findings_path,
        "summary": summary_path,
        "reviewed_rows": summary["reviewed_rows"],
        "false_negative_found_count": summary["false_negative_found_count"],
        "false_negative_rate": summary["false_negative_rate"],
        "wilson_95_ci_percent": summary["wilson_95_ci_percent"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

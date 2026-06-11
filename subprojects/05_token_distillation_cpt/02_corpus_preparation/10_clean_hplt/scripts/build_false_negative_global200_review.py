#!/usr/bin/env python3
"""Combine reviewed global-random false-negative controls into one estimate.

This normalizes:

- the preliminary 80-row representative review;
- high-risk reviews outside that 80-row sample;
- full-text Codex reviews for the remaining 111 rows.

It produces a 200-row representative false-negative estimate for the
`global_random` control population. It is review evidence only and does not
approve an automatic cleaning policy.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import json
import math
import os


SCHEMA_VERSION = "hplt-false-negative-global200-review-v1"
DEFAULT_TRIAGE = "reports/false_negative_control_triage_20260606T000000Z.jsonl"
DEFAULT_REP = "reports/false_negative_representative_review_annotations_20260606T011500Z.jsonl"
DEFAULT_HIGH = "reports/false_negative_triage_review_annotations_20260606T004500Z.jsonl"
DEFAULT_FULLTEXT = "reports/false_negative_global_remaining_fulltext_review_pack_20260606T020000Z.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/false_negative_global200_review_20260606T023000Z"


MANUAL_REVIEWS = {
    "pe_0426_1d116bad20": ("trim_suffix", ["E019", "E004"], "moderate", "Main page/introduction is followed by Blogger comments and a YouTube URL tail.", "Trim suffix starting at the first reader comment."),
    "pe_0497_bd904be146": ("trim_suffix", ["E009", "E007"], "moderate", "Coherent car article is followed by related-article chrome.", "Trim suffix beginning at 'Σχετικά άρθρα'."),
    "pe_0476_8477cf1e23": ("trim_span", ["E009"], "minor", "Coherent parenting article includes an inline related-link prompt after the lead.", "Remove the bounded 'Διαβάστε επίσης' related-link span only."),
    "pe_0962_464c3f1ee8": ("trim_suffix", ["E007", "E009"], "minor", "Short event announcement ends with related/category chrome.", "Trim suffix beginning at 'Δεν βρέθηκαν Σχετικά Άρθρα'."),
    "pe_1188_ce882a1c26": ("trim_span", ["E009", "E007"], "moderate", "News article has same-category list before the body plus read-more/related suffix after the body.", "Remove internal category list and suffix related/share blocks; preserve main Lillikas article."),
    "pe_0929_7e09664af8": ("trim_suffix", ["E007", "E009"], "minor", "Obituary article ends with Google News/YouTube/social follow CTA.", "Trim suffix starting at 'Σου άρεσε το άρθρο'."),
    "pe_1250_3456345a2b": ("trim_suffix", ["E008"], "minor", "Product-category prose ends with newsletter CTA.", "Trim final newsletter sentence."),
    "pe_2081_8eaa09ad48": ("trim_suffix", ["E019"], "minor", "Blogger article ends with 'Δεν υπάρχουν σχόλια' comment-form chrome.", "Trim Blogger comment suffix."),
    "pe_1276_53cbfa2f8e": ("trim_suffix", ["E019"], "minor", "Blogger sports post ends with no-comments/comment-form chrome.", "Trim Blogger comment suffix."),
    "pe_0453_0c5864d560": ("trim_suffix", ["E019"], "minor", "Coherent Android article ends with 'Γίνε ο πρώτος που θα σχολιάσει'.", "Trim final comment prompt."),
    "pe_1595_10165c6f2b": ("trim_suffix", ["E009", "E016", "E018"], "moderate", "Short article is followed by keyword label text and duplicated/teaser-like suffix.", "Trim suffix beginning at 'ΛΕΞΕΙΣ ΚΛΕΙΔΙΑ'."),
    "pe_0871_2f00391dc0": ("trim_suffix", ["E007", "E019"], "minor", "Entertainment article ends with ad/byline/comment-count chrome.", "Trim suffix starting at '– Advertisement –'."),
    "pe_2047_9df2135974": ("trim_suffix", ["E019", "E004"], "minor", "Corporate article ends with reader comments after the service URL.", "Trim reader-comment tail after the service link."),
    "pe_1427_11ba3ed840": ("trim_suffix", ["E009", "E004"], "minor", "Coherent article ends with a dangling 'read the bill here' link prompt.", "Trim final link prompt."),
    "pe_1407_6d8c86d5b3": ("split_doc", ["E010", "E011"], "serious", "Date/archive page contains multiple independent snippets separated by 'Συνέχεια ανάγνωσης'.", "Split only if child snippets pass useful-body thresholds; otherwise quarantine/drop derived row."),
    "pe_0608_b68a6ca698": ("normalize_or_trim_span", ["E002"], "moderate", "Legal/reference text contains literal NEWLINENEWLINE placeholders throughout.", "Normalize placeholder tokens to line breaks/spaces in a duplicate row."),
    "pe_1067_50358b08c5": ("trim_suffix", ["E019"], "minor", "Coherent news article ends with a bare 'Απάντηση' comment prompt.", "Trim final comment prompt."),
    "pe_0277_514d8f4322": ("trim_suffix", ["E018", "E007"], "minor", "Book-event article repeats credit/byline block at the end.", "Trim repeated terminal credits."),
    "pe_1945_153a66bb82": ("drop_doc", ["E016", "E020"], "serious", "Adult/product SEO advert dominates the row with machine-like promotional claims and keyword cross-links.", "Exclude from derived training stream; source row remains auditable."),
    "pe_1525_fb50b66e2d": ("trim_suffix", ["E009"], "minor", "Astrology article ends with a single related-link prompt.", "Trim final 'Διάβασε επίσης' link."),
    "pe_1094_04fb707cff": ("trim_prefix", ["E006"], "minor", "Article begins with a search-box prompt before the title/body.", "Trim leading 'Πληκτρολογήστε τις λέξεις'."),
    "pe_2145_d2d05e6aaf": ("trim_suffix", ["E009", "E004"], "minor", "Debt-law FAQ excerpt ends with read-whole-article/translate/share prompts.", "Trim suffix beginning at 'Διαβάστε όλο το άρθρο'."),
    "pe_1115_bf1769b25f": ("split_doc", ["E010", "E011"], "serious", "Economistas person/topic page contains several unrelated Novartis/Touloupaki snippets.", "Split useful snippets if they pass thresholds; otherwise exclude/quarantine derived row."),
    "pe_1380_2b47d66bb6": ("split_doc", ["E010", "E011"], "serious", "Economistas hotel topic page is a multi-snippet index, not one article.", "Split useful snippets if they pass thresholds; otherwise exclude/quarantine derived row."),
    "pe_1363_d7ee7a132a": ("trim_suffix", ["E019"], "minor", "Sports article ends with comment-form prompt.", "Trim final 'Το σχόλιο σας'."),
    "pe_0369_977e2b96d1": ("split_doc", ["E010", "E011"], "serious", "Culture category page concatenates many unrelated article snippets.", "Split only substantive child snippets; likely quarantine/drop short snippets."),
    "pe_0855_6fa13a6287": ("trim_suffix", ["E009", "E007"], "minor", "Celebrity article ends with a generic 'read all lifestyle news' link.", "Trim final related/category link."),
    "pe_0697_d34899c90d": ("trim_prefix", ["E003"], "moderate", "Legal text begins with JavaScript/source residue before the statute text.", "Trim leading script residue before numbered legal provisions."),
    "pe_0928_d276de70e9": ("trim_suffix", ["E006", "E008"], "minor", "Business profile ends with duplicated section/menu headings and a free-session CTA.", "Trim suffix beginning at repeated service/hour/CTA block."),
    "pe_1044_aecc9ef0d4": ("trim_span", ["E009"], "minor", "Psychology article contains several inline related-link prompts between sections.", "Remove bounded inline related-link spans; preserve numbered advice body."),
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


def by_id(rows):
    return dict((row["policy_evaluation_sample_id"], row) for row in rows)


def wilson(successes, n):
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def infer_severity(action):
    if action in ("drop_doc", "split_doc"):
        return "serious"
    if action in ("normalize_or_trim_span", "trim_prefix"):
        return "moderate"
    if action == "keep":
        return "none"
    return "minor"


def normalized_review(row, review, source_kind, source_path, created_at):
    action = review.get("review_correct_action")
    true_ids = list(review.get("review_true_error_type_ids") or [])
    label = review.get("review_label")
    if not label:
        label = "clean" if action == "keep" else "false_negative_found"
    severity = review.get("review_severity") or infer_severity(action)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at,
        "reviewer": "codex",
        "review_basis": "full global_random false-negative-control review; source_kind=%s" % source_kind,
        "review_source_kind": source_kind,
        "review_source_path": source_path,
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
        "chars_after": None,
        "chars_removed": None,
        "tokens_before": None,
        "tokens_after": None,
        "tokens_removed": None,
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "candidate_action_match": action == row.get("candidate_action"),
        "action_status": "reviewed_accept" if action == row.get("candidate_action") else "reviewed_reject",
        "review_label": label,
        "review_correct_action": action,
        "review_true_error_type_ids": true_ids,
        "review_false_negative_error_ids": list(review.get("review_false_negative_error_ids") or true_ids),
        "review_severity": severity,
        "review_good_text_loss_risk": review.get("review_good_text_loss_risk") or ("none" if action == "keep" else "low"),
        "review_notes": review.get("review_notes"),
        "review_span_or_split_notes": review.get("review_span_or_split_notes"),
        "sample_set": row.get("sample_set") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "triage_score": row.get("triage_score"),
        "triage_reasons": row.get("triage_reasons") or [],
        "detector_scores": row.get("detector_scores") or {},
        "evidence_excerpt": row.get("evidence_excerpt"),
    }


def manual_review_for_fulltext(row):
    pid = row["policy_evaluation_sample_id"]
    if pid in MANUAL_REVIEWS:
        action, ids, severity, notes, span_notes = MANUAL_REVIEWS[pid]
        return {
            "review_correct_action": action,
            "review_label": "false_negative_found",
            "review_true_error_type_ids": ids,
            "review_false_negative_error_ids": ids,
            "review_severity": severity,
            "review_good_text_loss_risk": "none" if action == "drop_doc" else "low",
            "review_notes": notes,
            "review_span_or_split_notes": span_notes,
        }
    return {
        "review_correct_action": "keep",
        "review_label": "clean",
        "review_true_error_type_ids": [],
        "review_false_negative_error_ids": [],
        "review_severity": "none",
        "review_good_text_loss_risk": "none",
        "review_notes": "Full text reviewed as coherent source content with no actionable residual HPLT extraction artifact.",
        "review_span_or_split_notes": "No cleaning action from reviewed evidence.",
    }


def make_summary(rows):
    n = len(rows)
    fn_rows = [row for row in rows if row["review_label"] == "false_negative_found"]
    serious_rows = [row for row in fn_rows if row.get("review_severity") == "serious"]
    fn_ci = wilson(len(fn_rows), n)
    serious_ci = wilson(len(serious_rows), n)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": rows[0]["created_at_utc"] if rows else None,
        "population": "all 200 global_random false-negative controls",
        "rows": n,
        "false_negative_found_count": len(fn_rows),
        "false_negative_rate": float(len(fn_rows)) / float(n or 1),
        "false_negative_wilson_95_ci": list(fn_ci),
        "false_negative_wilson_95_ci_percent": [fn_ci[0] * 100.0, fn_ci[1] * 100.0],
        "serious_false_negative_count": len(serious_rows),
        "serious_false_negative_rate": float(len(serious_rows)) / float(n or 1),
        "serious_false_negative_wilson_95_ci": list(serious_ci),
        "serious_false_negative_wilson_95_ci_percent": [serious_ci[0] * 100.0, serious_ci[1] * 100.0],
        "review_label_counts": dict(collections.Counter(row["review_label"] for row in rows)),
        "review_action_counts": dict(collections.Counter(row["review_correct_action"] for row in rows)),
        "review_severity_counts": dict(collections.Counter(row["review_severity"] for row in rows)),
        "true_error_type_counts": dict(collections.Counter(error_id for row in rows for error_id in row["review_true_error_type_ids"])),
        "source_kind_counts": dict(collections.Counter(row["review_source_kind"] for row in rows)),
        "policy_note": "Representative false-negative estimate for global_random controls only. This does not approve automatic cleaning or destructive policy promotion.",
    }


def write_markdown(path, rows, summary):
    fn_rows = [row for row in rows if row["review_label"] == "false_negative_found"]
    serious_rows = [row for row in fn_rows if row.get("review_severity") == "serious"]
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Global-Random False-Negative Review Findings\n\n")
        out.write("This combines the 80-row representative review, high-risk reviewed rows outside that sample, and full-text review of the remaining global-random controls. It is non-destructive review evidence only.\n\n")
        out.write("## Estimate\n\n")
        out.write("- population: `%s`\n" % summary["population"])
        out.write("- reviewed rows: `%d`\n" % summary["rows"])
        out.write("- any actionable false negatives: `%d` (`%.2f%%`, Wilson 95%% CI `%.2f%%` to `%.2f%%`)\n" % (
            summary["false_negative_found_count"],
            summary["false_negative_rate"] * 100.0,
            summary["false_negative_wilson_95_ci_percent"][0],
            summary["false_negative_wilson_95_ci_percent"][1],
        ))
        out.write("- serious false negatives: `%d` (`%.2f%%`, Wilson 95%% CI `%.2f%%` to `%.2f%%`)\n\n" % (
            summary["serious_false_negative_count"],
            summary["serious_false_negative_rate"] * 100.0,
            summary["serious_false_negative_wilson_95_ci_percent"][0],
            summary["serious_false_negative_wilson_95_ci_percent"][1],
        ))
        out.write("`any actionable` includes small suffix/inline related-link trims. `serious` is reserved for split/drop-style or body-dominating failures.\n\n")

        out.write("## Action Counts\n\n| Action | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["review_action_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## Severity Counts\n\n| Severity | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["review_severity_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## Error Type Counts In False Negatives\n\n| Error ID | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["true_error_type_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))

        out.write("\n## Serious Misses\n\n| ID | Host | Action | Error IDs | Note |\n| --- | --- | --- | --- | --- |\n")
        for row in serious_rows:
            out.write("| `%s` | `%s` | `%s` | `%s` | %s |\n" % (
                row["policy_evaluation_sample_id"],
                row.get("host") or "",
                row["review_correct_action"],
                ", ".join(row["review_true_error_type_ids"]),
                (row.get("review_notes") or "").replace("|", "\\|"),
            ))

        out.write("\n## Retrieval\n\n```bash\n")
        out.write("jq -c 'select(.review_label == \"false_negative_found\")' %s.jsonl\n" % DEFAULT_OUTPUT_PREFIX)
        out.write("jq -c 'select(.review_severity == \"serious\")' %s.jsonl\n" % DEFAULT_OUTPUT_PREFIX)
        out.write("jq -c 'select(.review_correct_action == \"split_doc\")' %s.jsonl\n" % DEFAULT_OUTPUT_PREFIX)
        out.write("jq -c 'select(.review_true_error_type_ids[]? == \"E011\")' %s.jsonl\n" % DEFAULT_OUTPUT_PREFIX)
        out.write("```\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", default=DEFAULT_TRIAGE)
    parser.add_argument("--representative", default=DEFAULT_REP)
    parser.add_argument("--high-risk", default=DEFAULT_HIGH)
    parser.add_argument("--fulltext", default=DEFAULT_FULLTEXT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    triage_rows = [row for row in read_jsonl(args.triage) if row.get("sample_set") == ["global_random"]]
    rep_by_id = by_id(read_jsonl(args.representative))
    high_by_id = by_id(read_jsonl(args.high_risk))
    full_by_id = by_id(read_jsonl(args.fulltext))
    created_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    output_rows = []
    for row in triage_rows:
        pid = row["policy_evaluation_sample_id"]
        if pid in rep_by_id:
            output_rows.append(normalized_review(row, rep_by_id[pid], "representative_80", args.representative, created_at))
        elif pid in high_by_id:
            output_rows.append(normalized_review(row, high_by_id[pid], "high_risk_reuse", args.high_risk, created_at))
        elif pid in full_by_id:
            output_rows.append(normalized_review(row, manual_review_for_fulltext(full_by_id[pid]), "fulltext_111", args.fulltext, created_at))
        else:
            raise RuntimeError("Missing review for global_random control %s" % pid)

    output_rows.sort(key=lambda row: row["policy_evaluation_sample_id"])
    summary = make_summary(output_rows)

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_jsonl(args.output_prefix + ".jsonl", output_rows)
    with open(args.output_prefix + "_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(args.output_prefix + ".md", output_rows, summary)

    print("wrote", args.output_prefix + ".jsonl")
    print("rows", len(output_rows))
    print("false_negatives", summary["false_negative_found_count"])
    print("serious_false_negatives", summary["serious_false_negative_count"])


if __name__ == "__main__":
    main()

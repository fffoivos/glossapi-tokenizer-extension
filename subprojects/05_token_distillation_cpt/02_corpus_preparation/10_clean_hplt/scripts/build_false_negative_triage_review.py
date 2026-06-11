#!/usr/bin/env python3
"""Build a reviewed high-risk false-negative triage slice.

The input triage file contains candidate `keep` rows from the policy-evaluation
false-negative controls. This script records a manual review of the highest
risk slice without changing source HPLT rows or emitting cleaned text.
"""

from __future__ import print_function

import argparse
import collections
import json
import os


CREATED_AT_UTC = "2026-06-06T00:45:00Z"
SCHEMA_VERSION = "hplt-false-negative-triage-review-v1"
SOURCE_TRIAGE = "reports/false_negative_control_triage_20260606T000000Z.jsonl"
SOURCE_TRIAGE_MD = "reports/false_negative_control_triage_20260606T000000Z.md"


ANNOTATIONS = {
    "pe_0505_80daa3e285": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E018", "E019"],
        "good_text_loss_risk": "low",
        "notes": "Coherent local political article; tail contains Blogger comment chrome and reviewed excerpt also shows likely duplicated body text.",
        "span_or_split_notes": "Trim comment suffix. Review any exact duplicated paragraph separately before materializing a trim_span.",
    },
    "pe_1300_a9899dc101": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E004", "E007"],
        "good_text_loss_risk": "low",
        "notes": "Useful Techgear article followed by RSS/follow-us instructions and a feed URL.",
        "span_or_split_notes": "Trim suffix beginning at the Google News/RSS subscription block.",
    },
    "pe_0399_68cc4ce22d": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Technical product review with specifications; list density is intrinsic article content.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1389_b019d536c0": {
        "review_correct_action": "drop_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E020"],
        "good_text_loss_risk": "medium",
        "notes": "Search/label URL yields a short orphan news snippet without a stable full article body.",
        "span_or_split_notes": "Drop from derived stream unless later full-source recovery can attach the complete article.",
    },
    "pe_1573_ea6228ca46": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E011"],
        "good_text_loss_risk": "medium",
        "notes": "Date/listing page concatenates several distinct crime/local-news snippets.",
        "span_or_split_notes": "Split only if child snippets pass minimum useful-body thresholds; otherwise quarantine/drop the derived row.",
    },
    "pe_1575_a4e7f26e86": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E007"],
        "good_text_loss_risk": "low",
        "notes": "Beauty article is coherent; source/category metadata at the end is removable suffix chrome.",
        "span_or_split_notes": "Trim final source/category lines only.",
    },
    "pe_2101_fb0d27554d": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Archive URL, but reviewed text is one coherent long post about Nikos Papazoglou.",
        "span_or_split_notes": "Archive URL alone is not sufficient for cleaning.",
    },
    "pe_1927_f943219372": {
        "review_correct_action": "trim_prefix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E006", "E017"],
        "good_text_loss_risk": "medium",
        "notes": "Book chapter body is preceded by contents/search/directory navigation blocks.",
        "span_or_split_notes": "Trim navigation prefix up to the chapter body boundary.",
    },
    "pe_1204_a1ad92a583": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Baby formula product page contains coherent source text and instructions; list format is source content.",
        "span_or_split_notes": "No extraction failure in reviewed text.",
    },
    "pe_1313_04698e5d1f": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E019"],
        "good_text_loss_risk": "low",
        "notes": "Useful political/health demands article ends with Blogger comment form text.",
        "span_or_split_notes": "Trim suffix at 'Δεν υπάρχουν σχόλια'.",
    },
    "pe_2068_ba19145e5d": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Book product page mixes price, description, and review, but this is coherent source content rather than HPLT extraction residue.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_0229_f6c330fbb9": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "ERT category URL resolves to a coherent regional news roundup under one topic.",
        "span_or_split_notes": "Archive/page URL is a false-positive signal here.",
    },
    "pe_0248_d9e7d18c99": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Sports interview article is coherent and complete in reviewed text.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_0461_8f4697cfd4": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E011", "E012"],
        "good_text_loss_risk": "medium",
        "notes": "Search/label page concatenates multiple cultivation posts separated by date/title resets.",
        "span_or_split_notes": "Split at date/title resets if each child has enough body text.",
    },
    "pe_0776_c8f46c5708": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Tag URL, but reviewed text is a coherent health article excerpt.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_0919_deb6f0b3e1": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E009", "E016"],
        "good_text_loss_risk": "low",
        "notes": "Airless pump article ends in a dense SEO/product-keyword list.",
        "span_or_split_notes": "Trim suffix beginning at the keyword list after the main article.",
    },
    "pe_0952_91617fbed6": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E011"],
        "good_text_loss_risk": "medium",
        "notes": "Basketball tag page concatenates many unrelated team/player news snippets.",
        "span_or_split_notes": "Split at snippet/article boundaries where child bodies are long enough.",
    },
    "pe_0983_120be4dde5": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Craft/blog label URL shows one coherent personal post in reviewed evidence.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_0986_124502ce67": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E011"],
        "good_text_loss_risk": "medium",
        "notes": "Category page concatenates several unrelated military/geopolitical snippets.",
        "span_or_split_notes": "Split if boundaries are recoverable; otherwise quarantine this derived row.",
    },
    "pe_1317_3368f95b2f": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "School invitation is short but coherent and complete.",
        "span_or_split_notes": "Shortness alone is not a cleaning error.",
    },
    "pe_1360_6a49a32229": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E011"],
        "good_text_loss_risk": "medium",
        "notes": "News menu/listing row contains multiple short political/news snippets rather than one main body.",
        "span_or_split_notes": "Split at article/snippet boundaries if recoverable; quarantine if boundary confidence is weak.",
    },
    "pe_1814_7edb008a46": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Archive URL, but reviewed text is a coherent long-form fiction/literary post.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1946_a32107ac3f": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Long weather/news article is coherent; source URL tail is not enough to justify a cleaning action.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_2060_a0371956b7": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E011", "E012"],
        "good_text_loss_risk": "medium",
        "notes": "Monthly archive page shows date/title reset structure and multiple long blog entries.",
        "span_or_split_notes": "Split at date/title boundaries; do not drop useful long-form text.",
    },
    "pe_2089_6dd0d1457b": {
        "review_correct_action": "split_doc",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E010", "E011"],
        "good_text_loss_risk": "medium",
        "notes": "Category page concatenates several event/wedding/party snippets.",
        "span_or_split_notes": "Split into event/topic child rows where enough text remains.",
    },
    "pe_0977_7ea5d49a7e": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Police/local news article is coherent; list-like structure is source prose.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1035_f0da62c911": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Ski-resort information page is coherent source content.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1279_ac418333b4": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Company profile and services list are source content, not extraction residue.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1314_12be1d4ea4": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E019"],
        "good_text_loss_risk": "low",
        "notes": "Conference press release is coherent but ends with Blogger comment form text.",
        "span_or_split_notes": "Trim suffix at '0 σχόλια'.",
    },
    "pe_1493_5eba6423bb": {
        "review_correct_action": "trim_suffix",
        "review_label": "false_negative_found",
        "true_error_type_ids": ["E004", "E019"],
        "good_text_loss_risk": "low",
        "notes": "Short tech article ends with source URL and Blogger comment form text.",
        "span_or_split_notes": "Trim suffix from source/comment block if retaining the article.",
    },
    "pe_1514_2a4a3f350e": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Short sports/support-card announcement is complete source content.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
    "pe_1542_12286092b9": {
        "review_correct_action": "keep",
        "review_label": "clean",
        "true_error_type_ids": [],
        "good_text_loss_risk": "none",
        "notes": "Travel deal amenity list is source content; this may be a corpus-source policy issue, not residual HPLT extraction dirt.",
        "span_or_split_notes": "No cleaning action from reviewed evidence.",
    },
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


def sorted_reviewed_rows(triage_rows):
    by_id = {row.get("policy_evaluation_sample_id"): row for row in triage_rows}
    missing = sorted(set(ANNOTATIONS) - set(by_id))
    if missing:
        raise RuntimeError("Annotation IDs missing from triage input: %s" % ", ".join(missing))
    rows = []
    for sample_id, ann in ANNOTATIONS.items():
        source = by_id[sample_id]
        candidate_action = source.get("candidate_action")
        correct_action = ann["review_correct_action"]
        candidate_action_match = candidate_action == correct_action
        true_ids = list(ann["true_error_type_ids"])
        false_positive_reasons = []
        if candidate_action_match:
            action_status = "reviewed_accept"
        else:
            action_status = "reviewed_reject"
            false_positive_reasons = list(source.get("triage_reasons") or [])
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": CREATED_AT_UTC,
            "reviewer": "codex",
            "review_basis": "manual review of the high-risk slice from %s and Markdown excerpts in %s; no source rows changed; targeted triage slice, not a representative false-negative-rate estimate" % (SOURCE_TRIAGE, SOURCE_TRIAGE_MD),
            "source_triage": SOURCE_TRIAGE,
            "source_triage_markdown": SOURCE_TRIAGE_MD,
            "false_negative_triage_id": source.get("false_negative_triage_id"),
            "policy_evaluation_sample_id": sample_id,
            "policy_review_pack_id": source.get("policy_review_pack_id"),
            "source_doc_id": source.get("source_doc_id"),
            "parent_source_doc_id": source.get("parent_source_doc_id") or source.get("source_doc_id"),
            "derived_doc_id": None,
            "is_shadow_record": False,
            "doc_text_path": source.get("doc_text_path"),
            "host": source.get("host"),
            "url": source.get("url"),
            "quality_bin": source.get("quality_bin"),
            "text_sha256_before": source.get("text_sha256_before"),
            "chars_before": source.get("chars_before"),
            "chars_after": None,
            "chars_removed": None,
            "candidate_action": candidate_action,
            "review_correct_action": correct_action,
            "candidate_action_match": candidate_action_match,
            "action_status": action_status,
            "review_label": ann["review_label"],
            "candidate_error_type_ids": source.get("candidate_error_type_ids") or [],
            "review_true_error_type_ids": true_ids,
            "review_false_negative_error_ids": true_ids if not candidate_action_match else [],
            "false_positive_triage_reasons": false_positive_reasons,
            "triage_score": source.get("triage_score"),
            "triage_reasons": source.get("triage_reasons") or [],
            "detector_scores": {
                "triage_score": source.get("triage_score"),
                "word_count": source.get("word_count"),
                "greek_word_share": source.get("greek_word_share"),
                "short_line_density": source.get("short_line_density"),
                "list_line_density": source.get("list_line_density"),
                "url_hits": source.get("url_hits"),
                "comment_hits": source.get("comment_hits"),
                "related_hits": source.get("related_hits"),
                "feed_hits": source.get("feed_hits"),
                "date_boundary_hits": source.get("date_boundary_hits"),
            },
            "sample_set": source.get("sample_set") or [],
            "sampling_strata": source.get("sampling_strata") or [],
            "evidence_excerpt": source.get("review_excerpt") or source.get("evidence_excerpt"),
            "review_good_text_loss_risk": ann["good_text_loss_risk"],
            "review_notes": ann["notes"],
            "review_span_or_split_notes": ann["span_or_split_notes"],
        })
    return sorted(rows, key=lambda row: (-float(row.get("triage_score") or 0.0), row.get("policy_evaluation_sample_id") or ""))


def make_summary(rows, triage_rows):
    reviewed_ids = set(row["policy_evaluation_sample_id"] for row in rows)
    score_threshold = min(row["triage_score"] for row in rows if row.get("triage_score") is not None)
    remaining = [row for row in triage_rows if row.get("policy_evaluation_sample_id") not in reviewed_ids]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": CREATED_AT_UTC,
        "source_triage": SOURCE_TRIAGE,
        "source_triage_markdown": SOURCE_TRIAGE_MD,
        "reviewed_rows": len(rows),
        "triage_rows": len(triage_rows),
        "review_slice": "targeted high-risk false-negative controls with triage_score >= %.2f" % score_threshold,
        "unreviewed_rows": len(remaining),
        "candidate_action_match_count": sum(1 for row in rows if row["candidate_action_match"]),
        "false_negative_found_count": sum(1 for row in rows if row["review_label"] == "false_negative_found"),
        "label_counts": dict(collections.Counter(row["review_label"] for row in rows)),
        "review_correct_action_counts": dict(collections.Counter(row["review_correct_action"] for row in rows)),
        "true_error_type_counts": dict(collections.Counter(error_id for row in rows for error_id in row["review_true_error_type_ids"])),
        "host_counts": dict(collections.Counter(row["host"] for row in rows)),
        "remaining_review_note": "This targeted slice cannot estimate the false-negative rate or confidence interval. Use representative controls before promoting a destructive policy.",
    }


def write_markdown(path, rows, summary):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# False-Negative Triage Review Findings\n\n")
        out.write("This review covers the targeted high-risk false-negative-control slice from `false_negative_control_triage_20260606T000000Z`. It is non-destructive: source rows are unchanged and no shadow rows are materialized.\n\n")
        out.write("## Summary\n\n")
        out.write("- reviewed rows: `%d` of `%d`\n" % (summary["reviewed_rows"], summary["triage_rows"]))
        out.write("- slice: `%s`\n" % summary["review_slice"])
        out.write("- candidate `keep` accepted: `%d`\n" % summary["candidate_action_match_count"])
        out.write("- missed cleaning actions found: `%d`\n" % summary["false_negative_found_count"])
        out.write("- remaining unreviewed controls: `%d`\n\n" % summary["unreviewed_rows"])
        out.write("This is a targeted high-risk review, not a prevalence estimate. It should not be used as the full 300-row false-negative rate or confidence interval.\n\n")
        out.write("## Action Counts\n\n| Reviewed action | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["review_correct_action_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## True Error Types In Misses\n\n| Error ID | Rows |\n| --- | ---: |\n")
        for key, count in sorted(summary["true_error_type_counts"].items(), key=lambda item: (-item[1], item[0])):
            out.write("| `%s` | %d |\n" % (key, count))
        out.write("\n## Interpretation\n\n")
        out.write("- Archive/tag/category URLs are high-recall but noisy: several rows are coherent single articles and should stay `keep`.\n")
        out.write("- Real misses concentrate in comment/RSS suffixes, SEO keyword tails, navigation prefixes, and archive/category pages that concatenate multiple snippets or posts.\n")
        out.write("- Product/deal/service pages are frequent false positives for the triage detector; most are source-content questions rather than residual HPLT extraction failures.\n")
        out.write("- The representative false-negative estimate must come from the random-control frame, not just this high-risk slice.\n\n")
        out.write("## Reviewed Rows\n\n")
        out.write("| Sample | Host | Score | Triage reasons | Reviewed action | Label | True IDs | Note |\n")
        out.write("| --- | --- | ---: | --- | --- | --- | --- | --- |\n")
        for row in rows:
            reasons = ", ".join(row["triage_reasons"])
            true_ids = ", ".join(row["review_true_error_type_ids"])
            note = row["review_notes"].replace("|", "/")
            out.write("| `%s` | `%s` | %.2f | `%s` | `%s` | `%s` | `%s` | %s |\n" % (
                row["policy_evaluation_sample_id"],
                row["host"],
                row["triage_score"],
                reasons,
                row["review_correct_action"],
                row["review_label"],
                true_ids,
                note,
            ))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-jsonl", default=SOURCE_TRIAGE)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--timestamp", default="20260606T004500Z")
    return parser.parse_args()


def main():
    args = parse_args()
    triage_rows = list(read_jsonl(args.triage_jsonl))
    rows = sorted_reviewed_rows(triage_rows)
    summary = make_summary(rows, triage_rows)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    annotations_path = os.path.join(args.output_dir, "false_negative_triage_review_annotations_%s.jsonl" % args.timestamp)
    summary_path = os.path.join(args.output_dir, "false_negative_triage_review_summary_%s.json" % args.timestamp)
    findings_path = os.path.join(args.output_dir, "false_negative_triage_review_findings_%s.md" % args.timestamp)
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
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

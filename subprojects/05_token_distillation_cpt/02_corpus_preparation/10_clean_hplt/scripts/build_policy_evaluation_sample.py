#!/usr/bin/env python3
"""Build a policy-evaluation review sample from an HPLT action manifest.

This is the next queue after small pilot packs. It selects rows for prevalence,
false-negative, per-action precision, per-error precision, WDS-bin, and
host-hotspot review. It is non-destructive: it only emits review worklists and
sampling metadata.
"""

from __future__ import print_function

import argparse
import collections
import csv
import datetime as _dt
import hashlib
import json
import os
import random


ERROR_NAMES = collections.OrderedDict([
    ("E001", "replacement/control/private-use characters"),
    ("E002", "escaped Unicode, HTML entities, percent-encoding residue"),
    ("E003", "CSS/JS/HTML/XML remnants"),
    ("E004", "URL/path/metadata-key dumps"),
    ("E005", "mojibake-like symbol and punctuation clutter"),
    ("E006", "top navigation/menu chrome"),
    ("E007", "footer/contact/social/copyright chrome"),
    ("E008", "cookie/login/newsletter overlays"),
    ("E009", "related-post/comment/share scaffolding"),
    ("E010", "heading/list-only snippets or truncated extraction"),
    ("E011", "multiple independent documents in one row"),
    ("E012", "repeated title/date/author metadata resets"),
    ("E013", "internal paragraph or sentence repetition loops"),
    ("E014", "localized gibberish or OCR-like noise islands"),
    ("E015", "low Greek body share or language drift"),
    ("E016", "SEO keyword lists or machine-generated register spam"),
    ("E017", "table/list extraction fragments without prose"),
    ("E018", "duplicated main body separated by boilerplate"),
    ("E019", "forum/comment/form scaffolding dominating content"),
    ("E020", "no main body: tiny fragments, index pages, orphan headings"),
])

DESTRUCTIVE_OR_CHANGE_ACTIONS = set([
    "drop_doc",
    "normalize_or_trim_span",
    "trim_prefix",
    "trim_suffix",
    "trim_span",
    "split_doc",
])

REVIEW_ACTIONS = sorted(DESTRUCTIVE_OR_CHANGE_ACTIONS | set(["quarantine"]))

SAMPLE_RATIONALE = collections.OrderedDict([
    (
        "global_prevalence",
        "Up to 1,067 global-random rows: about +/-3 percentage-point 95% CI "
        "half-width at worst-case binary prevalence before finite-population correction.",
    ),
    (
        "wds_bin",
        "Up to 200 rows per WDS bin: about +/-7 percentage-point within-bin "
        "95% CI half-width at worst-case prevalence.",
    ),
    (
        "action_precision",
        "Up to 60 rows per action, or all rows when fewer exist. With zero "
        "false positives, rule-of-three gives an upper FP bound near 5%, matching "
        "the default 95% destructive-rule precision gate.",
    ),
    (
        "error_precision",
        "Up to 60 rows per error ID, prioritizing high-confidence examples plus "
        "random remainder. With zero false positives this is just enough to make "
        "95% precision plausible; any observed FP implies more review or a stricter rule.",
    ),
    (
        "false_negative_controls",
        "Up to 300 keep/no-candidate or borderline controls. If zero serious "
        "false negatives are found, rule-of-three gives an upper serious-FN "
        "rate around 1% in that control frame.",
    ),
    (
        "host_hotspots",
        "Top hosts are sampled separately so a high-volume domain cannot hide a "
        "domain-specific extraction failure behind global averages.",
    ),
])


def utc_timestamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def stable_key(value):
    return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()


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


def deterministic_sample(items, size, seed, salt):
    items = list(items)
    if size <= 0:
        return []
    if len(items) <= size:
        return list(items)
    rng = random.Random("%s:%s" % (seed, salt))
    rng.shuffle(items)
    return items[:size]


def top_then_random(items, size, seed, salt, score_fn):
    items = list(items)
    if size <= 0:
        return []
    if len(items) <= size:
        return list(items)
    top_n = min(len(items), max(1, size // 2))
    top = sorted(items, key=score_fn, reverse=True)[:top_n]
    top_ids = set(id(item) for item in top)
    rest = [item for item in items if id(item) not in top_ids]
    return top + deterministic_sample(rest, size - len(top), seed, salt)


def as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def combined_risk(record):
    scores = record.get("detector_scores") or {}
    score_values = [
        as_float(scores.get("encoding_score")),
        as_float(scores.get("markup_score")),
        as_float(scores.get("symbol_score")),
        as_float(scores.get("boilerplate_score")),
        as_float(scores.get("internal_repetition_score")),
        as_float(scores.get("split_candidate_score")),
        as_float(scores.get("lang_drift_score")),
        as_float(scores.get("badness_score")),
        as_float(scores.get("archive_snippet_score")),
        as_float(scores.get("catalog_list_score")),
        as_float(scores.get("seo_spam_score")),
        as_float(scores.get("form_comment_score")),
        as_float(scores.get("no_main_body_score")),
    ]
    return max(score_values + [as_float(record.get("confidence"))])


def read_prior_reviews(paths):
    prior = {}
    for path in paths:
        if not path:
            continue
        for row in read_jsonl(path):
            source_doc_id = row.get("source_doc_id")
            if source_doc_id and source_doc_id not in prior:
                prior[source_doc_id] = row
    return prior


def add_selected(selected, record, task, reason, stratum, target_n):
    source_doc_id = record.get("source_doc_id")
    if source_doc_id not in selected:
        selected[source_doc_id] = {
            "record": record,
            "evaluation_tasks": set(),
            "evaluation_reasons": set(),
            "sampling_strata": set(),
            "target_ns": {},
        }
    payload = selected[source_doc_id]
    payload["evaluation_tasks"].add(task)
    payload["evaluation_reasons"].add(reason)
    payload["sampling_strata"].add(stratum)
    payload["target_ns"][stratum] = target_n


def select_records(records, args):
    selected = collections.OrderedDict()
    by_error = collections.defaultdict(list)
    by_action = collections.defaultdict(list)
    by_host = collections.defaultdict(list)
    by_qbin = collections.defaultdict(list)
    by_sample_set = collections.defaultdict(list)

    for record in records:
        for error_id in record.get("error_type_ids") or []:
            by_error[error_id].append(record)
        by_action[record.get("action")].append(record)
        by_host[record.get("host") or "unknown"].append(record)
        by_qbin[str(record.get("quality_bin"))].append(record)
        for sample_name in record.get("sample_set") or []:
            by_sample_set[sample_name].append(record)

    global_random = by_sample_set.get("global_random", [])
    for record in deterministic_sample(global_random, args.global_random_n, args.seed, "global_random"):
        add_selected(
            selected,
            record,
            "global_prevalence_review",
            "global_random_sample",
            "global_prevalence",
            args.global_random_n,
        )

    for qbin in args.wds_bins:
        items = by_sample_set.get("wds_bin_%s" % qbin, by_qbin.get(str(qbin), []))
        for record in deterministic_sample(items, args.wds_bin_n, args.seed, "wds_bin:%s" % qbin):
            add_selected(
                selected,
                record,
                "wds_bin_%s_review" % qbin,
                "wds_bin_%s_stratum" % qbin,
                "wds_bin_%s" % qbin,
                args.wds_bin_n,
            )

    for action in REVIEW_ACTIONS:
        items = by_action.get(action, [])
        chosen = top_then_random(
            items,
            args.action_precision_n,
            args.seed,
            "action:%s" % action,
            lambda item: (combined_risk(item), as_float(item.get("chars_before"))),
        )
        for record in chosen:
            add_selected(
                selected,
                record,
                "action_precision_%s" % action,
                "candidate_action_%s" % action,
                "action_precision_%s" % action,
                args.action_precision_n,
            )

    for error_id in ERROR_NAMES:
        items = by_error.get(error_id, [])
        chosen = top_then_random(
            items,
            args.error_precision_n,
            args.seed,
            "error:%s" % error_id,
            lambda item: (
                as_float((item.get("detector_scores") or {}).get("badness_score")),
                combined_risk(item),
                as_float(item.get("confidence")),
            ),
        )
        for record in chosen:
            add_selected(
                selected,
                record,
                "error_precision_%s" % error_id,
                "candidate_error_%s" % error_id,
                "error_precision_%s" % error_id,
                args.error_precision_n,
            )

    keep_no_candidate = [
        record for record in records
        if record.get("action") == "keep" and not (record.get("error_type_ids") or [])
    ]
    keep_global = [
        record for record in keep_no_candidate
        if "global_random" in (record.get("sample_set") or [])
    ]
    keep_borderline = [
        record for record in keep_no_candidate
        if "borderline_middle_risk" in (record.get("sample_set") or [])
    ]
    controls = deterministic_sample(
        keep_global,
        args.false_negative_global_n,
        args.seed,
        "fn:global_keep",
    ) + deterministic_sample(
        keep_borderline,
        args.false_negative_borderline_n,
        args.seed,
        "fn:borderline_keep",
    )
    for record in controls:
        add_selected(
            selected,
            record,
            "false_negative_control",
            "keep_no_candidate_or_borderline",
            "false_negative_controls",
            args.false_negative_global_n + args.false_negative_borderline_n,
        )

    top_hosts = sorted(by_host.items(), key=lambda item: len(item[1]), reverse=True)
    for host, items in top_hosts[:args.host_hotspot_count]:
        chosen = top_then_random(
            items,
            args.host_hotspot_n,
            args.seed,
            "host:%s" % host,
            lambda item: (combined_risk(item), as_float(item.get("chars_before"))),
        )
        for record in chosen:
            add_selected(
                selected,
                record,
                "host_hotspot_review",
                "host_hotspot_%s" % host,
                "host_hotspot_%s" % host,
                args.host_hotspot_n,
            )

    source_counts = {
        "by_action": {str(action): len(items) for action, items in sorted(by_action.items())},
        "by_error_type": {error_id: len(items) for error_id, items in sorted(by_error.items())},
        "by_host_top": dict((host, len(items)) for host, items in top_hosts[:50]),
        "by_quality_bin": {str(qbin): len(items) for qbin, items in sorted(by_qbin.items())},
        "by_sample_set": {name: len(items) for name, items in sorted(by_sample_set.items())},
    }
    return selected, source_counts


def selected_rows(selected, prior_reviews, timestamp):
    rows = []
    for index, (source_doc_id, payload) in enumerate(sorted(selected.items()), 1):
        record = payload["record"]
        prior = prior_reviews.get(source_doc_id) or {}
        row = collections.OrderedDict([
            ("policy_evaluation_sample_id", "pe_%04d_%s" % (index, stable_key(source_doc_id)[:10])),
            ("source_doc_id", source_doc_id),
            ("parent_source_doc_id", record.get("parent_source_doc_id")),
            ("derived_doc_id", record.get("derived_doc_id")),
            ("is_shadow_record", record.get("is_shadow_record")),
            ("evaluation_tasks", sorted(payload["evaluation_tasks"])),
            ("evaluation_reasons", sorted(payload["evaluation_reasons"])),
            ("sampling_strata", sorted(payload["sampling_strata"])),
            ("sampling_target_ns", dict(sorted(payload["target_ns"].items()))),
            ("candidate_error_type_ids", record.get("error_type_ids") or []),
            ("candidate_action", record.get("action")),
            ("action_status", record.get("action_status")),
            ("confidence", record.get("confidence")),
            ("quality_bin", record.get("quality_bin")),
            ("host", record.get("host")),
            ("url", record.get("url")),
            ("chars_before", record.get("chars_before")),
            ("sample_set", record.get("sample_set") or []),
            ("detector_scores", record.get("detector_scores") or {}),
            ("evidence_excerpt", record.get("evidence_excerpt")),
            ("doc_text_path", record.get("doc_text_path")),
            ("text_sha256_before", record.get("text_sha256_before")),
            ("prior_review_label", prior.get("review_label")),
            ("prior_review_true_error_type_ids", prior.get("review_true_error_type_ids") or []),
            ("prior_review_correct_action", prior.get("review_correct_action")),
            ("prior_review_notes", prior.get("review_notes")),
            ("review_label", prior.get("review_label")),
            ("review_true_error_type_ids", prior.get("review_true_error_type_ids") or []),
            ("review_correct_action", prior.get("review_correct_action")),
            ("review_span_or_split_notes", prior.get("review_span_or_split_notes")),
            ("review_good_text_loss_risk", prior.get("review_good_text_loss_risk")),
            ("review_false_positive_reason", prior.get("review_false_positive_reason")),
            ("review_false_negative_error_ids", prior.get("review_false_negative_error_ids") or []),
            ("review_notes", prior.get("review_notes")),
            ("schema_version", "hplt-policy-evaluation-sample-v1"),
            ("created_at_utc", timestamp),
        ])
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fields = [
        "policy_evaluation_sample_id",
        "source_doc_id",
        "evaluation_tasks",
        "evaluation_reasons",
        "sampling_strata",
        "candidate_error_type_ids",
        "candidate_action",
        "confidence",
        "quality_bin",
        "host",
        "url",
        "chars_before",
        "doc_text_path",
        "prior_review_label",
        "prior_review_correct_action",
        "review_label",
        "review_correct_action",
        "review_good_text_loss_risk",
        "review_false_positive_reason",
        "review_false_negative_error_ids",
        "review_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                payload[field] = value
            writer.writerow(payload)


def counter_for(rows, field):
    counter = collections.Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        else:
            counter[str(value)] += 1
    return counter


def write_summary(path, rows, args, source_counts, outputs):
    by_task = counter_for(rows, "evaluation_tasks")
    by_action = counter_for(rows, "candidate_action")
    by_error = collections.Counter()
    for row in rows:
        for error_id in row.get("candidate_error_type_ids") or []:
            by_error[error_id] += 1
    by_stratum = counter_for(rows, "sampling_strata")
    prior_reviewed = sum(1 for row in rows if row.get("prior_review_label"))
    summary = {
        "input_action_manifest": args.action_manifest,
        "records": len(rows),
        "prior_reviewed_records": prior_reviewed,
        "policy_evaluation_sample_jsonl": outputs["jsonl"],
        "policy_evaluation_sample_csv": outputs["csv"],
        "policy_evaluation_sample_md": outputs["md"],
        "by_evaluation_task": dict(by_task.most_common()),
        "by_candidate_action": dict(by_action.most_common()),
        "by_candidate_error_type": dict(by_error.most_common()),
        "by_sampling_stratum": dict(by_stratum.most_common()),
        "source_counts": source_counts,
        "selection": {
            "global_random_n": args.global_random_n,
            "wds_bin_n": args.wds_bin_n,
            "wds_bins": args.wds_bins,
            "action_precision_n": args.action_precision_n,
            "error_precision_n": args.error_precision_n,
            "false_negative_global_n": args.false_negative_global_n,
            "false_negative_borderline_n": args.false_negative_borderline_n,
            "host_hotspot_count": args.host_hotspot_count,
            "host_hotspot_n": args.host_hotspot_n,
            "seed": args.seed,
        },
        "sample_rationale": SAMPLE_RATIONALE,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def write_markdown(path, rows, args, summary):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# HPLT Cleaning Policy-Evaluation Sample\n\n")
        out.write("This worklist is non-destructive. It samples candidate rows for review before any cleaning overlay is promoted. Original HPLT rows remain immutable; accepted changes must later become duplicate/shadow records.\n\n")
        out.write("## Inputs\n\n")
        out.write("- action manifest: `%s`\n" % args.action_manifest)
        out.write("- records selected: `%d`\n" % len(rows))
        out.write("- prior reviewed records carried forward: `%d`\n" % summary["prior_reviewed_records"])
        out.write("- seed: `%s`\n\n" % args.seed)
        out.write("## Sample-Size Rationale\n\n")
        for key, note in SAMPLE_RATIONALE.items():
            out.write("- `%s`: %s\n" % (key, note))
        out.write("\n## Counts\n\n")
        out.write("### By Evaluation Task\n\n| Task | Rows |\n| --- | ---: |\n")
        for task, count in summary["by_evaluation_task"].items():
            out.write("| `%s` | %d |\n" % (task, count))
        out.write("\n### By Candidate Action\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in summary["by_candidate_action"].items():
            out.write("| `%s` | %d |\n" % (action, count))
        out.write("\n### By Candidate Error Type\n\n| Error ID | Definition | Rows |\n| --- | --- | ---: |\n")
        for error_id, count in summary["by_candidate_error_type"].items():
            out.write("| `%s` | %s | %d |\n" % (error_id, ERROR_NAMES.get(error_id, ""), count))
        out.write("\n## Review Contract\n\n")
        out.write("Each reviewed row must fill `review_label`, `review_true_error_type_ids`, `review_correct_action`, `review_good_text_loss_risk`, and notes. For any trim or split recommendation, the reviewer must add boundary notes precise enough to become `span_ranges` or `split_parts` in a later shadow manifest. Do not apply changes here.\n\n")
        out.write("## First Rows\n\n")
        out.write("| Sample ID | Tasks | Error IDs | Action | Host | QBin | Prior | Text path |\n")
        out.write("| --- | --- | --- | --- | --- | ---: | --- | --- |\n")
        for row in rows[:120]:
            out.write("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | `%s` | `%s` |\n" % (
                row.get("policy_evaluation_sample_id"),
                ",".join(row.get("evaluation_tasks") or []),
                ",".join(row.get("candidate_error_type_ids") or []),
                row.get("candidate_action"),
                row.get("host"),
                row.get("quality_bin"),
                row.get("prior_review_label") or "",
                row.get("doc_text_path"),
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--global-random-n", type=int, default=1067)
    parser.add_argument("--wds-bin-n", type=int, default=200)
    parser.add_argument("--wds-bins", nargs="+", default=["8", "9", "10"])
    parser.add_argument("--action-precision-n", type=int, default=60)
    parser.add_argument("--error-precision-n", type=int, default=60)
    parser.add_argument("--false-negative-global-n", type=int, default=200)
    parser.add_argument("--false-negative-borderline-n", type=int, default=100)
    parser.add_argument("--host-hotspot-count", type=int, default=20)
    parser.add_argument("--host-hotspot-n", type=int, default=10)
    parser.add_argument("--prior-annotations", nargs="*", default=[])
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    mkdir(args.output_dir)
    by_task_dir = os.path.join(args.output_dir, "by_task")
    mkdir(by_task_dir)

    records = list(read_jsonl(args.action_manifest))
    prior_reviews = read_prior_reviews(args.prior_annotations)
    selected, source_counts = select_records(records, args)
    rows = selected_rows(selected, prior_reviews, timestamp)

    jsonl_path = os.path.join(args.output_dir, "policy_evaluation_sample_%s.jsonl" % timestamp)
    csv_path = os.path.join(args.output_dir, "policy_evaluation_sample_%s.csv" % timestamp)
    md_path = os.path.join(args.output_dir, "policy_evaluation_sample_%s.md" % timestamp)
    summary_path = os.path.join(args.output_dir, "policy_evaluation_sample_summary_%s.json" % timestamp)

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    for task in sorted(set(task for row in rows for task in row.get("evaluation_tasks", []))):
        task_rows = [row for row in rows if task in row.get("evaluation_tasks", [])]
        write_jsonl(os.path.join(by_task_dir, "%s.jsonl" % task), task_rows)

    outputs = {"jsonl": jsonl_path, "csv": csv_path, "md": md_path, "summary": summary_path}
    summary = write_summary(summary_path, rows, args, source_counts, outputs)
    write_markdown(md_path, rows, args, summary)

    print(json.dumps({
        "event": "complete",
        "records": len(rows),
        "prior_reviewed_records": summary["prior_reviewed_records"],
        "policy_evaluation_sample_jsonl": jsonl_path,
        "policy_evaluation_sample_csv": csv_path,
        "policy_evaluation_sample_md": md_path,
        "summary": summary_path,
        "by_task_dir": by_task_dir,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

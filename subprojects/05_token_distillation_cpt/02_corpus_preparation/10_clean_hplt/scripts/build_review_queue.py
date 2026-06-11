#!/usr/bin/env python3
"""Build deterministic HPLT cleaning review queues from an action manifest.

The output is a labeling worklist. It does not edit original text and does not
apply a cleaning policy. It selects:

- precision-review examples for each candidate error type;
- all or sampled destructive-action candidates;
- clean-looking/global-random controls for false-negative review.
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

ERROR_TO_SCORE = {
    "E001": "encoding_score",
    "E002": "encoding_score",
    "E003": "markup_score",
    "E004": "markup_score",
    "E005": "symbol_score",
    "E006": "boilerplate_score",
    "E007": "boilerplate_score",
    "E008": "boilerplate_score",
    "E009": "boilerplate_score",
    "E010": "archive_snippet_score",
    "E011": "split_candidate_score",
    "E012": "split_candidate_score",
    "E013": "internal_repetition_score",
    "E014": "badness_score",
    "E015": "lang_drift_score",
    "E016": "seo_spam_score",
    "E017": "catalog_list_score",
    "E018": "internal_repetition_score",
    "E019": "form_comment_score",
    "E020": "no_main_body_score",
}

DESTRUCTIVE_ACTIONS = set(["normalize_or_trim_span", "trim_prefix", "trim_suffix", "trim_span", "split_doc", "drop_doc"])


def utc_timestamp():
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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


def score_for_error(record, error_id):
    score_name = ERROR_TO_SCORE.get(error_id)
    return float((record.get("detector_scores") or {}).get(score_name) or 0.0)


def deterministic_sample(items, size, seed, salt):
    if size <= 0 or len(items) <= size:
        return list(items)
    rng = random.Random("%s:%s" % (seed, salt))
    keyed = list(items)
    rng.shuffle(keyed)
    return keyed[:size]


def select_top_and_random(items, top_n, random_n, seed, salt, score_fn):
    top = sorted(items, key=score_fn, reverse=True)[:max(0, top_n)]
    top_ids = set(id(item) for item in top)
    rest = [item for item in items if id(item) not in top_ids]
    random_part = deterministic_sample(rest, random_n, seed, salt)
    return top + random_part


def add_task(selected, record, task, reason):
    source_doc_id = record.get("source_doc_id")
    if source_doc_id not in selected:
        selected[source_doc_id] = {
            "record": record,
            "review_tasks": set(),
            "priority_reasons": set(),
        }
    selected[source_doc_id]["review_tasks"].add(task)
    selected[source_doc_id]["priority_reasons"].add(reason)


def build_queue(records, args):
    selected = {}
    by_error = collections.defaultdict(list)
    by_action = collections.defaultdict(list)
    for record in records:
        for error_id in record.get("error_type_ids") or []:
            by_error[error_id].append(record)
        by_action[record.get("action")].append(record)

    for error_id, items in by_error.items():
        chosen = select_top_and_random(
            items,
            args.top_per_error,
            args.random_per_error,
            args.seed,
            "error:%s" % error_id,
            lambda item, err=error_id: (score_for_error(item, err), float(item.get("confidence") or 0.0)),
        )
        for record in chosen:
            add_task(selected, record, "precision_%s" % error_id, "candidate_error_%s" % error_id)

    destructive = [record for record in records if record.get("action") in DESTRUCTIVE_ACTIONS]
    destructive_chosen = select_top_and_random(
        destructive,
        args.top_destructive,
        args.random_destructive,
        args.seed,
        "destructive",
        lambda item: (float(item.get("confidence") or 0.0), float(item.get("chars_before") or 0.0)),
    )
    for record in destructive_chosen:
        add_task(selected, record, "destructive_action_review", "candidate_action_%s" % record.get("action"))

    clean_controls = [
        record for record in records
        if not (record.get("error_type_ids") or [])
        and record.get("action") == "keep"
        and "global_random" in (record.get("sample_set") or [])
    ]
    for record in deterministic_sample(clean_controls, args.false_negative_controls, args.seed, "false_negative_controls"):
        add_task(selected, record, "false_negative_control", "global_random_no_candidate_error")

    borderline_controls = [
        record for record in records
        if not (record.get("error_type_ids") or [])
        and "borderline_middle_risk" in (record.get("sample_set") or [])
    ]
    for record in deterministic_sample(borderline_controls, args.borderline_controls, args.seed, "borderline_controls"):
        add_task(selected, record, "borderline_false_negative_probe", "borderline_no_candidate_error")

    return selected, by_error, by_action


def queue_record(source_doc_id, payload, order):
    record = payload["record"]
    review_queue_id = "rq_%04d_%s" % (order, stable_key(source_doc_id)[:10])
    return collections.OrderedDict([
        ("review_queue_id", review_queue_id),
        ("source_doc_id", source_doc_id),
        ("parent_source_doc_id", record.get("parent_source_doc_id")),
        ("derived_doc_id", record.get("derived_doc_id")),
        ("is_shadow_record", record.get("is_shadow_record")),
        ("review_tasks", sorted(payload["review_tasks"])),
        ("priority_reasons", sorted(payload["priority_reasons"])),
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
        ("review_label", None),
        ("review_true_error_type_ids", []),
        ("review_correct_action", None),
        ("review_span_or_split_notes", None),
        ("review_good_text_loss_risk", None),
        ("review_false_positive_reason", None),
        ("review_false_negative_error_ids", []),
        ("review_notes", None),
    ])


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fields = [
        "review_queue_id", "source_doc_id", "review_tasks", "priority_reasons",
        "candidate_error_type_ids", "candidate_action", "confidence",
        "quality_bin", "host", "url", "chars_before", "doc_text_path",
        "review_label", "review_true_error_type_ids", "review_correct_action",
        "review_good_text_loss_risk", "review_false_positive_reason",
        "review_false_negative_error_ids", "review_notes",
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


def write_summary(path, args, rows, by_error, by_action, outputs):
    tasks = collections.Counter()
    errors = collections.Counter()
    actions = collections.Counter()
    hosts = collections.Counter()
    qbins = collections.Counter()
    for row in rows:
        for task in row.get("review_tasks") or []:
            tasks[task] += 1
        for error_id in row.get("candidate_error_type_ids") or []:
            errors[error_id] += 1
        actions[str(row.get("candidate_action"))] += 1
        hosts[str(row.get("host"))] += 1
        qbins[str(row.get("quality_bin"))] += 1

    payload = {
        "input_action_manifest": args.action_manifest,
        "queue_records": len(rows),
        "review_queue_jsonl": outputs["jsonl"],
        "review_queue_csv": outputs["csv"],
        "review_queue_md": outputs["md"],
        "by_review_task": dict(tasks.most_common()),
        "by_candidate_error_type": dict(errors.most_common()),
        "by_candidate_action": dict(actions.most_common()),
        "by_quality_bin": dict(qbins.most_common()),
        "top_hosts": dict(hosts.most_common(50)),
        "source_candidate_error_counts": {error_id: len(items) for error_id, items in sorted(by_error.items())},
        "source_action_counts": {action: len(items) for action, items in sorted(by_action.items())},
        "selection": {
            "top_per_error": args.top_per_error,
            "random_per_error": args.random_per_error,
            "top_destructive": args.top_destructive,
            "random_destructive": args.random_destructive,
            "false_negative_controls": args.false_negative_controls,
            "borderline_controls": args.borderline_controls,
            "seed": args.seed,
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def write_markdown(path, args, rows, summary):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# HPLT Cleaning Review Queue\n\n")
        out.write("This queue is for manual/reviewer labeling. It is non-destructive and uses original doc paths plus candidate action metadata.\n\n")
        out.write("## Inputs\n\n")
        out.write("- action manifest: `%s`\n" % args.action_manifest)
        out.write("- queue records: `%d`\n\n" % len(rows))
        out.write("## Review Labels\n\n")
        out.write("Use `review_label` values: `true_positive`, `false_positive`, `partial_true_positive`, `false_negative_found`, `clean`, `unclear`.\n\n")
        out.write("Use `review_correct_action` values: `keep`, `normalize_or_trim_span`, `trim_prefix`, `trim_suffix`, `trim_span`, `split_doc`, `drop_doc`, `quarantine`.\n\n")
        out.write("## Counts\n\n")
        out.write("### By Review Task\n\n| Task | Count |\n| --- | ---: |\n")
        for task, count in summary["by_review_task"].items():
            out.write("| `%s` | %d |\n" % (task, count))
        out.write("\n### By Candidate Error Type\n\n| ID | Type | Count |\n| --- | --- | ---: |\n")
        for error_id, count in summary["by_candidate_error_type"].items():
            out.write("| `%s` | %s | %d |\n" % (error_id, ERROR_NAMES.get(error_id, ""), count))
        out.write("\n### By Candidate Action\n\n| Action | Count |\n| --- | ---: |\n")
        for action, count in summary["by_candidate_action"].items():
            out.write("| `%s` | %d |\n" % (action, count))
        out.write("\n## First Records\n\n")
        out.write("| Queue ID | Tasks | Error IDs | Action | Host | QBin | Text path |\n| --- | --- | --- | --- | --- | ---: | --- |\n")
        for row in rows[:80]:
            out.write("| `%s` | `%s` | `%s` | `%s` | `%s` | %s | `%s` |\n" % (
                row.get("review_queue_id"),
                ",".join(row.get("review_tasks") or []),
                ",".join(row.get("candidate_error_type_ids") or []),
                row.get("candidate_action"),
                row.get("host"),
                row.get("quality_bin"),
                row.get("doc_text_path"),
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--top-per-error", type=int, default=8)
    parser.add_argument("--random-per-error", type=int, default=12)
    parser.add_argument("--top-destructive", type=int, default=40)
    parser.add_argument("--random-destructive", type=int, default=40)
    parser.add_argument("--false-negative-controls", type=int, default=100)
    parser.add_argument("--borderline-controls", type=int, default=60)
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    mkdir(args.output_dir)
    by_error_dir = os.path.join(args.output_dir, "by_task")
    mkdir(by_error_dir)

    records = list(read_jsonl(args.action_manifest))
    selected, by_error, by_action = build_queue(records, args)
    rows = [
        queue_record(source_doc_id, payload, index + 1)
        for index, (source_doc_id, payload) in enumerate(sorted(selected.items()))
    ]

    jsonl_path = os.path.join(args.output_dir, "review_queue_%s.jsonl" % timestamp)
    csv_path = os.path.join(args.output_dir, "review_queue_%s.csv" % timestamp)
    md_path = os.path.join(args.output_dir, "review_queue_%s.md" % timestamp)
    summary_path = os.path.join(args.output_dir, "review_queue_summary_%s.json" % timestamp)

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    for task in sorted(set(task for row in rows for task in row.get("review_tasks", []))):
        task_rows = [row for row in rows if task in row.get("review_tasks", [])]
        write_jsonl(os.path.join(by_error_dir, "%s.jsonl" % task), task_rows)

    outputs = {"jsonl": jsonl_path, "csv": csv_path, "md": md_path, "summary": summary_path}
    summary = write_summary(summary_path, args, rows, by_error, by_action, outputs)
    write_markdown(md_path, args, rows, summary)

    print(json.dumps({
        "event": "complete",
        "records": len(rows),
        "review_queue_jsonl": jsonl_path,
        "review_queue_csv": csv_path,
        "review_queue_md": md_path,
        "summary": summary_path,
        "by_task_dir": by_error_dir,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

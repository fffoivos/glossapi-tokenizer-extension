#!/usr/bin/env python3
"""Build a filtered review pack from a policy-evaluation sample.

The policy-evaluation sample is large. This script materializes focused review
packs, for example all candidate text-changing actions, without applying any
cleaning. It reads the review-bundle doc text paths and writes Markdown plus the
selected JSONL rows.
"""

from __future__ import print_function

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import random


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
    if size <= 0 or len(items) <= size:
        return items
    rng = random.Random("%s:%s" % (seed, salt))
    rng.shuffle(items)
    return items[:size]


def read_doc(path, max_chars):
    if not path or not os.path.exists(path):
        return "", False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    if max_chars > 0 and len(text) > max_chars:
        return compact_snippet(text, max_chars // 2, max_chars // 2), True
    return text, False


def compact_snippet(text, head_chars, tail_chars):
    text = text.replace("\r\n", "\n")
    if len(text) <= head_chars + tail_chars + 200:
        return text
    return text[:head_chars] + "\n\n...[SNIP]...\n\n" + text[-tail_chars:]


def parse_csv_arg(value):
    if not value:
        return set()
    out = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            out.add(item)
    return out


def row_matches(row, actions, tasks, strata, ids):
    if actions and row.get("candidate_action") not in actions:
        return False
    if ids and row.get("policy_evaluation_sample_id") not in ids and row.get("source_doc_id") not in ids:
        return False
    if tasks:
        row_tasks = set(row.get("evaluation_tasks") or [])
        if not row_tasks.intersection(tasks):
            return False
    if strata:
        row_strata = set(row.get("sampling_strata") or [])
        if not row_strata.intersection(strata):
            return False
    return True


def select_rows(rows, args):
    actions = parse_csv_arg(args.actions)
    tasks = parse_csv_arg(args.tasks)
    strata = parse_csv_arg(args.strata)
    ids = parse_csv_arg(args.ids)
    selected = [row for row in rows if row_matches(row, actions, tasks, strata, ids)]
    selected = sorted(selected, key=lambda row: (
        str(row.get("candidate_action")),
        str(row.get("host")),
        str(row.get("policy_evaluation_sample_id")),
    ))
    if args.max_records > 0:
        selected = deterministic_sample(selected, args.max_records, args.seed, "policy_review_pack")
        selected = sorted(selected, key=lambda row: str(row.get("policy_evaluation_sample_id")))
    return selected, {
        "actions": sorted(actions),
        "tasks": sorted(tasks),
        "strata": sorted(strata),
        "ids": sorted(ids),
        "max_records": args.max_records,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_md(path, rows, filters, args):
    action_counts = collections.Counter(row.get("candidate_action") for row in rows)
    prior_counts = collections.Counter(str(row.get("prior_review_label")) for row in rows)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# HPLT Policy Review Pack\n\n")
        out.write("This pack is non-destructive. It reads review-bundle doc text and policy-evaluation rows only. Any accepted cleaning must later become duplicate/shadow derived records; originals remain immutable.\n\n")
        out.write("## Selection\n\n")
        out.write("- policy evaluation sample: `%s`\n" % args.policy_evaluation_sample)
        out.write("- records: `%d`\n" % len(rows))
        out.write("- seed: `%s`\n" % args.seed)
        out.write("- max doc chars per row: `%s`\n" % args.max_doc_chars)
        out.write("- filters: `%s`\n\n" % json.dumps(filters, ensure_ascii=False, sort_keys=True))
        out.write("## Counts\n\n")
        out.write("### By Candidate Action\n\n| Action | Rows |\n| --- | ---: |\n")
        for action, count in action_counts.most_common():
            out.write("| `%s` | %d |\n" % (action, count))
        out.write("\n### By Prior Review Label\n\n| Prior label | Rows |\n| --- | ---: |\n")
        for label, count in prior_counts.most_common():
            out.write("| `%s` | %d |\n" % (label, count))
        out.write("\n## Review Guidance\n\n")
        out.write("For each row, decide the least destructive correct action: `keep`, `normalize_or_trim_span`, `trim_prefix`, `trim_suffix`, `trim_span`, `split_doc`, `drop_doc`, or `quarantine`. Record true error IDs, false positives, good-text-loss risk, and span/split boundary notes precise enough for a later shadow manifest.\n\n")
        out.write("## Records\n\n")
        for row in rows:
            doc_path = row.get("doc_text_path")
            text, truncated = read_doc(doc_path, args.max_doc_chars)
            pack_id = row.get("policy_evaluation_sample_id")
            out.write("### %s\n\n" % pack_id)
            out.write("- source_doc_id: `%s`\n" % row.get("source_doc_id"))
            out.write("- evaluation_tasks: `%s`\n" % "`, `".join(row.get("evaluation_tasks") or []))
            out.write("- sampling_strata: `%s`\n" % "`, `".join(row.get("sampling_strata") or []))
            out.write("- candidate_error_type_ids: `%s`\n" % "`, `".join(row.get("candidate_error_type_ids") or []))
            out.write("- candidate_action: `%s`\n" % row.get("candidate_action"))
            out.write("- prior_review_label: `%s`\n" % row.get("prior_review_label"))
            out.write("- prior_review_correct_action: `%s`\n" % row.get("prior_review_correct_action"))
            out.write("- prior_review_true_error_type_ids: `%s`\n" % "`, `".join(row.get("prior_review_true_error_type_ids") or []))
            out.write("- host: `%s`\n" % row.get("host"))
            out.write("- url: `%s`\n" % row.get("url"))
            out.write("- qbin: `%s`\n" % row.get("quality_bin"))
            out.write("- chars_before: `%s`\n" % row.get("chars_before"))
            out.write("- text_sha256_before: `%s`\n" % row.get("text_sha256_before"))
            out.write("- doc_text_path: `%s`\n" % doc_path)
            out.write("- text_truncated_in_pack: `%s`\n\n" % truncated)
            out.write("Evidence excerpt:\n\n```text\n%s\n```\n\n" % (row.get("evidence_excerpt") or ""))
            if row.get("prior_review_notes"):
                out.write("Prior review notes:\n\n```text\n%s\n```\n\n" % row.get("prior_review_notes"))
            out.write("Document text:\n\n```text\n%s\n```\n\n" % text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-evaluation-sample", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--actions", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--strata", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-doc-chars", type=int, default=80000)
    parser.add_argument("--pack-name", default="policy_review_pack")
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    mkdir(args.output_dir)
    rows, filters = select_rows(list(read_jsonl(args.policy_evaluation_sample)), args)
    for row in rows:
        row["policy_review_pack_id"] = "policy_pack_%s_%s" % (
            timestamp,
            stable_key(row.get("policy_evaluation_sample_id") or row.get("source_doc_id") or "")[:8],
        )

    base = "%s_%s" % (args.pack_name, timestamp)
    jsonl_path = os.path.join(args.output_dir, "%s.jsonl" % base)
    md_path = os.path.join(args.output_dir, "%s.md" % base)
    summary_path = os.path.join(args.output_dir, "%s_summary.json" % base)
    write_jsonl(jsonl_path, rows)
    write_md(md_path, rows, filters, args)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "policy_evaluation_sample": args.policy_evaluation_sample,
            "records": len(rows),
            "jsonl": jsonl_path,
            "markdown": md_path,
            "filters": filters,
            "seed": args.seed,
            "max_doc_chars": args.max_doc_chars,
            "by_candidate_action": dict(collections.Counter(row.get("candidate_action") for row in rows).most_common()),
            "by_prior_review_label": dict(collections.Counter(str(row.get("prior_review_label")) for row in rows).most_common()),
        }, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "event": "complete",
        "records": len(rows),
        "jsonl": jsonl_path,
        "markdown": md_path,
        "summary": summary_path,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

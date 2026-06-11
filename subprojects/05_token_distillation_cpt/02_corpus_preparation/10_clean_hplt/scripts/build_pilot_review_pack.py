#!/usr/bin/env python3
"""Build a compact pilot review pack from the HPLT cleaning review queue."""

from __future__ import print_function

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import random


TASK_TARGETS_LEGACY = collections.OrderedDict([
    ("precision_E003", 2),
    ("precision_E004", 2),
    ("precision_E005", 2),
    ("precision_E006", 2),
    ("precision_E008", 2),
    ("precision_E009", 2),
    ("precision_E011", 3),
    ("precision_E012", 2),
    ("precision_E013", 2),
    ("precision_E014", 3),
    ("precision_E015", 2),
    ("destructive_action_review", 4),
    ("false_negative_control", 4),
    ("borderline_false_negative_probe", 4),
])

TASK_TARGETS_V2_NEWCLASSES = collections.OrderedDict([
    ("precision_E010", 6),
    ("precision_E017", 6),
    ("precision_E019", 5),
    ("precision_E020", 6),
    ("precision_E016", 6),
    ("precision_E009", 3),
    ("precision_E011", 3),
    ("precision_E003", 2),
    ("precision_E005", 2),
    ("destructive_action_review", 8),
    ("false_negative_control", 4),
    ("borderline_false_negative_probe", 4),
])

TASK_TARGETS_V4_POLICY = collections.OrderedDict([
    ("destructive_action_review", 14),
    ("precision_E014", 6),
    ("precision_E016", 5),
    ("precision_E020", 6),
    ("precision_E010", 4),
    ("precision_E017", 4),
    ("precision_E011", 3),
    ("precision_E009", 3),
    ("precision_E003", 2),
    ("precision_E004", 2),
    ("false_negative_control", 6),
    ("borderline_false_negative_probe", 6),
])

TASK_TARGET_PROFILES = {
    "legacy": TASK_TARGETS_LEGACY,
    "v2_newclasses": TASK_TARGETS_V2_NEWCLASSES,
    "v4_policy": TASK_TARGETS_V4_POLICY,
}


def utc_timestamp():
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


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


def stable_key(value):
    return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()


def deterministic_sample(items, size, seed, salt):
    if len(items) <= size:
        return list(items)
    rng = random.Random("%s:%s" % (seed, salt))
    items = list(items)
    rng.shuffle(items)
    return items[:size]


def read_doc(path, max_chars):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read(max_chars)


def compact_snippet(text, head_chars, tail_chars):
    text = text.replace("\r\n", "\n")
    if len(text) <= head_chars + tail_chars + 200:
        return text
    return text[:head_chars] + "\n\n...[SNIP]...\n\n" + text[-tail_chars:]


def select_records(rows, seed, task_targets):
    by_task = collections.defaultdict(list)
    for row in rows:
        for task in row.get("review_tasks") or []:
            by_task[task].append(row)

    selected = collections.OrderedDict()
    selected_by_task = {}
    for task, target in task_targets.items():
        chosen = deterministic_sample(by_task.get(task, []), target, seed, task)
        selected_by_task[task] = [row.get("review_queue_id") for row in chosen]
        for row in chosen:
            selected[row["review_queue_id"]] = row
    return list(selected.values()), selected_by_task


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_md(path, rows, selected_by_task, args):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Pilot HPLT Cleaning Review Pack\n\n")
        out.write("This pack is for a first calibrated review pass. It is non-destructive and reads materialized review docs only.\n\n")
        out.write("## Selection\n\n")
        out.write("- queue: `%s`\n" % args.review_queue)
        out.write("- records: `%d`\n" % len(rows))
        out.write("- seed: `%s`\n\n" % args.seed)
        out.write("- target profile: `%s`\n\n" % args.target_profile)
        out.write("## Selected Queue IDs By Task\n\n")
        for task, ids in selected_by_task.items():
            out.write("- `%s`: `%s`\n" % (task, "`, `".join(ids)))
        out.write("\n## Review Records\n\n")

        for row in rows:
            doc_path = row.get("doc_text_path")
            text = read_doc(doc_path, args.max_doc_chars)
            snippet = compact_snippet(text, args.head_chars, args.tail_chars)
            out.write("### %s\n\n" % row.get("review_queue_id"))
            out.write("- source_doc_id: `%s`\n" % row.get("source_doc_id"))
            out.write("- tasks: `%s`\n" % "`, `".join(row.get("review_tasks") or []))
            out.write("- candidate_error_type_ids: `%s`\n" % "`, `".join(row.get("candidate_error_type_ids") or []))
            out.write("- candidate_action: `%s`\n" % row.get("candidate_action"))
            out.write("- host: `%s`\n" % row.get("host"))
            out.write("- qbin: `%s`\n" % row.get("quality_bin"))
            out.write("- chars_before: `%s`\n" % row.get("chars_before"))
            out.write("- doc_text_path: `%s`\n\n" % doc_path)
            out.write("Evidence excerpt:\n\n```text\n%s\n```\n\n" % (row.get("evidence_excerpt") or ""))
            out.write("Document snippet:\n\n```text\n%s\n```\n\n" % snippet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-doc-chars", type=int, default=25000)
    parser.add_argument("--head-chars", type=int, default=3500)
    parser.add_argument("--tail-chars", type=int, default=2500)
    parser.add_argument("--target-profile", choices=sorted(TASK_TARGET_PROFILES), default="legacy")
    args = parser.parse_args()

    timestamp = args.timestamp or utc_timestamp()
    mkdir(args.output_dir)

    rows = list(read_jsonl(args.review_queue))
    selected, selected_by_task = select_records(rows, args.seed, TASK_TARGET_PROFILES[args.target_profile])
    for row in selected:
        row["pilot_review_id"] = "pilot_%s_%s" % (timestamp, stable_key(row["review_queue_id"])[:8])

    jsonl_path = os.path.join(args.output_dir, "pilot_review_seed_%s.jsonl" % timestamp)
    md_path = os.path.join(args.output_dir, "pilot_review_pack_%s.md" % timestamp)
    summary_path = os.path.join(args.output_dir, "pilot_review_pack_summary_%s.json" % timestamp)

    write_jsonl(jsonl_path, selected)
    write_md(md_path, selected, selected_by_task, args)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "review_queue": args.review_queue,
            "records": len(selected),
            "pilot_seed_jsonl": jsonl_path,
            "pilot_pack_md": md_path,
            "selected_by_task": selected_by_task,
            "seed": args.seed,
            "target_profile": args.target_profile,
        }, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "event": "complete",
        "records": len(selected),
        "pilot_seed_jsonl": jsonl_path,
        "pilot_pack_md": md_path,
        "summary": summary_path,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

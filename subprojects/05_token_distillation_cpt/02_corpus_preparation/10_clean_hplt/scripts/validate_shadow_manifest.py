#!/usr/bin/env python3
"""Validate duplicate/shadow invariants for HPLT cleaning manifests.

Candidate review manifests are allowed to describe proposed actions. Once a row
is materialized with changed text, the original HPLT row must remain immutable:
the changed text must appear only as a duplicate/shadow row with parent and
derived IDs plus before/after checksums.
"""

import argparse
import collections
import json
import sys
from pathlib import Path


TEXT_CHANGING_ACTIONS = {
    "normalize_or_trim_span",
    "trim_prefix",
    "trim_suffix",
    "trim_span",
    "split_doc",
}

VALID_ACTIONS = TEXT_CHANGING_ACTIONS | {"keep", "drop_doc", "quarantine"}
MATERIALIZED_STATUSES = {"applied"}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def has_value(row, key):
    value = row.get(key)
    return value is not None and value != "" and value != []


def looks_materialized(row):
    if row.get("action_status") in MATERIALIZED_STATUSES:
        return True
    if row.get("is_shadow_record") is True:
        return True
    if has_value(row, "text_sha256_after"):
        return True
    if has_value(row, "chars_after") or has_value(row, "tokens_after"):
        return True
    return False


def add_violation(violations, path, line_number, source_doc_id, code, detail):
    violations.append({
        "path": str(path),
        "line_number": line_number,
        "source_doc_id": source_doc_id,
        "code": code,
        "detail": detail,
    })


def validate_row(path, line_number, row, violations):
    source_doc_id = row.get("source_doc_id")
    action = row.get("action")
    status = row.get("action_status")

    for key in ["source_doc_id", "action", "action_status"]:
        if not has_value(row, key):
            add_violation(violations, path, line_number, source_doc_id, "missing_base_field", key)

    if action and action not in VALID_ACTIONS:
        add_violation(violations, path, line_number, source_doc_id, "unknown_action", action)

    if not looks_materialized(row):
        return

    if action == "keep":
        add_violation(violations, path, line_number, source_doc_id, "materialized_keep", "keep rows should not materialize changed text")
        return

    if action == "quarantine":
        add_violation(violations, path, line_number, source_doc_id, "materialized_quarantine", "quarantine is a holdout decision, not changed text")
        return

    if action == "drop_doc":
        for key in ["parent_source_doc_id", "derived_doc_id", "text_sha256_after"]:
            if has_value(row, key):
                add_violation(
                    violations,
                    path,
                    line_number,
                    source_doc_id,
                    "drop_doc_has_shadow_fields",
                    f"{key} should be empty for exclusion-only drop_doc",
                )
        if row.get("is_shadow_record") is True:
            add_violation(violations, path, line_number, source_doc_id, "drop_doc_shadow_record", "drop_doc is manifest exclusion only")
        return

    for key in ["parent_source_doc_id", "derived_doc_id", "text_sha256_before", "text_sha256_after"]:
        if not has_value(row, key):
            add_violation(violations, path, line_number, source_doc_id, "missing_shadow_field", key)

    if row.get("is_shadow_record") is not True:
        add_violation(violations, path, line_number, source_doc_id, "not_shadow_record", "changed text must set is_shadow_record=true")

    if has_value(row, "parent_source_doc_id") and source_doc_id and row.get("parent_source_doc_id") != source_doc_id:
        add_violation(
            violations,
            path,
            line_number,
            source_doc_id,
            "parent_mismatch",
            f"parent_source_doc_id={row.get('parent_source_doc_id')!r}",
        )

    if has_value(row, "derived_doc_id") and row.get("derived_doc_id") == source_doc_id:
        add_violation(violations, path, line_number, source_doc_id, "derived_equals_source", "derived_doc_id must differ from source_doc_id")

    if row.get("text_sha256_before") == row.get("text_sha256_after"):
        add_violation(violations, path, line_number, source_doc_id, "unchanged_checksum", "changed action has identical before/after checksum")

    if action == "split_doc" and not has_value(row, "split_parts"):
        add_violation(violations, path, line_number, source_doc_id, "missing_split_parts", "split_doc requires split_parts")

    if action in {"trim_prefix", "trim_suffix", "trim_span", "normalize_or_trim_span"} and not has_value(row, "span_ranges"):
        add_violation(violations, path, line_number, source_doc_id, "missing_span_ranges", f"{action} requires span_ranges")


def validate(paths):
    violations = []
    records = 0
    materialized = 0
    by_action = collections.Counter()
    for path in paths:
        for line_number, row in read_jsonl(path):
            records += 1
            by_action[row.get("action")] += 1
            if looks_materialized(row):
                materialized += 1
            validate_row(path, line_number, row, violations)
    return {
        "records": records,
        "materialized_records": materialized,
        "violations": violations,
        "violation_count": len(violations),
        "by_action": dict(by_action.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Write machine-readable summary.")
    args = parser.parse_args()

    summary = validate(args.manifest)
    if args.json:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print("records:", summary["records"])
        print("materialized_records:", summary["materialized_records"])
        print("violation_count:", summary["violation_count"])
        for violation in summary["violations"][:50]:
            print(
                "{path}:{line_number}: {code}: {source_doc_id}: {detail}".format(
                    **violation
                )
            )
        if summary["violation_count"] > 50:
            print("... %d more violations" % (summary["violation_count"] - 50))

    return 1 if summary["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

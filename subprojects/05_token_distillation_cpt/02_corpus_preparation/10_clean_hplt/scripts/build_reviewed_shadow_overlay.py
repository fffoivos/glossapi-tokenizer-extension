#!/usr/bin/env python3
"""Aggregate reviewed HPLT cleaning decisions into one shadow-overlay artifact.

This is a reviewed-evidence overlay, not an automatic detector policy. It
concatenates already materialized decision manifests and shadow records,
deduplicates exact rows, validates duplicate/shadow invariants, and reports
char/token deltas by action. Source HPLT rows remain immutable.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import validate_shadow_manifest


SCHEMA_VERSION = "hplt-reviewed-shadow-overlay-v1"
TEXT_CHANGING_ACTIONS = {
    "normalize_or_trim_span",
    "trim_prefix",
    "trim_suffix",
    "trim_span",
    "split_doc",
}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def canonical_row(row: dict[str, Any]) -> str:
    cleaned = {key: value for key, value in row.items() if not key.startswith("_")}
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_doc_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or "")


def error_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("error_type_ids", "review_true_error_type_ids", "candidate_error_type_ids_reviewed_as_true"):
        value = row.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).startswith("E"))
    return sorted(set(ids))


def find_inputs(reports_dir: Path) -> tuple[list[Path], list[Path]]:
    decision_paths = sorted(reports_dir.glob("boundary_*materialization_*_decision_manifest.jsonl"))
    decision_paths += sorted(reports_dir.glob("e001_shadow_overlay_*_manifest.jsonl"))
    shadow_paths = sorted(reports_dir.glob("boundary_*materialization_*_shadow_records.jsonl"))
    shadow_paths += sorted(reports_dir.glob("e001_shadow_overlay_*_shadow_records.jsonl"))
    decision_paths = [path for path in decision_paths if "dryrun" not in path.name]
    shadow_paths = [path for path in shadow_paths if "dryrun" not in path.name]
    return decision_paths, shadow_paths


def load_deduped(paths: list[Path], source_field: str) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    duplicate_count = 0
    source_counts: dict[str, int] = {}
    for path in paths:
        path_rows = read_jsonl(path)
        source_counts[str(path)] = len(path_rows)
        for row in path_rows:
            key = canonical_row(row)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            out = dict(row)
            out[source_field] = str(path)
            rows.append(out)
    return rows, duplicate_count, source_counts


def collapse_source_deltas(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    collapsed: list[dict[str, Any]] = []
    for row in decision_rows:
        action = str(row.get("action") or "")
        key = (source_doc_id(row), action, str(row.get("text_sha256_before") or ""))
        if key in seen:
            continue
        seen.add(key)
        collapsed.append(row)
    return collapsed


def action_summary(decision_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_action: dict[str, dict[str, int]] = {}
    source_sets: dict[str, set[str]] = collections.defaultdict(set)
    for row in collapse_source_deltas(decision_rows):
        action = str(row.get("action") or "unknown")
        bucket = by_action.setdefault(
            action,
            {
                "decision_rows": 0,
                "source_docs": 0,
                "shadow_records": 0,
                "chars_removed": 0,
                "tokens_removed": 0,
                "held_out_chars": 0,
                "held_out_tokens": 0,
            },
        )
        bucket["decision_rows"] += 1
        source_sets[action].add(source_doc_id(row))
        if row.get("is_shadow_record") is True:
            bucket["shadow_records"] += 1
        bucket["chars_removed"] += int(row.get("chars_removed") or 0)
        bucket["tokens_removed"] += int(row.get("tokens_removed") or 0)
        bucket["held_out_chars"] += int(row.get("held_out_chars") or 0)
        bucket["held_out_tokens"] += int(row.get("held_out_tokens") or 0)
    for action, docs in source_sets.items():
        by_action[action]["source_docs"] = len(docs)
    return by_action


def error_action_summary(decision_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for row in collapse_source_deltas(decision_rows):
        action = str(row.get("action") or "unknown")
        for error_id in error_ids(row):
            bucket = result.setdefault(error_id, {}).setdefault(
                action,
                {
                    "source_docs": 0,
                    "chars_removed": 0,
                    "tokens_removed": 0,
                    "held_out_chars": 0,
                    "held_out_tokens": 0,
                },
            )
            bucket["source_docs"] += 1
            bucket["chars_removed"] += int(row.get("chars_removed") or 0)
            bucket["tokens_removed"] += int(row.get("tokens_removed") or 0)
            bucket["held_out_chars"] += int(row.get("held_out_chars") or 0)
            bucket["held_out_tokens"] += int(row.get("held_out_tokens") or 0)
    return result


def validate_shadow_records(decision_rows: list[dict[str, Any]], shadow_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    shadow_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(shadow_rows, 1):
        derived_id = str(row.get("derived_doc_id") or "")
        if not derived_id:
            issues.append({"kind": "shadow_record", "index": index, "code": "missing_derived_doc_id"})
            continue
        if derived_id in shadow_by_id:
            issues.append({"kind": "shadow_record", "derived_doc_id": derived_id, "code": "duplicate_derived_doc_id"})
            continue
        shadow_by_id[derived_id] = row

    expected_ids: set[str] = set()
    for index, row in enumerate(decision_rows, 1):
        action = str(row.get("action") or "")
        if action not in TEXT_CHANGING_ACTIONS:
            continue
        if row.get("is_shadow_record") is not True:
            continue
        derived_id = str(row.get("derived_doc_id") or "")
        expected_ids.add(derived_id)
        shadow = shadow_by_id.get(derived_id)
        if not shadow:
            issues.append({"kind": "decision", "index": index, "source_doc_id": source_doc_id(row), "derived_doc_id": derived_id, "code": "missing_shadow_record"})
            continue
        if shadow.get("source_doc_id") != row.get("source_doc_id"):
            issues.append({"kind": "decision", "index": index, "derived_doc_id": derived_id, "code": "shadow_source_mismatch"})
        if shadow.get("parent_source_doc_id") != row.get("parent_source_doc_id"):
            issues.append({"kind": "decision", "index": index, "derived_doc_id": derived_id, "code": "shadow_parent_mismatch"})
        if shadow.get("text_sha256_before") != row.get("text_sha256_before"):
            issues.append({"kind": "decision", "index": index, "derived_doc_id": derived_id, "code": "shadow_before_hash_mismatch"})
        if shadow.get("text_sha256_after") != row.get("text_sha256_after"):
            issues.append({"kind": "decision", "index": index, "derived_doc_id": derived_id, "code": "shadow_after_hash_mismatch"})
        shadow_text = str(
            shadow.get("full_text")
            if shadow.get("full_text") is not None
            else shadow.get("text")
            if shadow.get("text") is not None
            else ""
        )
        if sha256_text(shadow_text) != row.get("text_sha256_after"):
            issues.append({"kind": "decision", "index": index, "derived_doc_id": derived_id, "code": "shadow_full_text_hash_mismatch"})

    for derived_id in sorted(set(shadow_by_id) - expected_ids):
        issues.append({"kind": "shadow_record", "derived_doc_id": derived_id, "code": "unreferenced_shadow_record"})
    return issues


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Reviewed HPLT Shadow Overlay\n\n")
        out.write("This is a reviewed-evidence overlay artifact. It aggregates already reviewed/materialized duplicate and shadow decisions. It does not mutate source HPLT rows and does not approve broad detector actions.\n\n")
        out.write("## Summary\n\n")
        out.write(f"- created: `{summary['created_at_utc']}`\n")
        out.write(f"- decision rows: `{summary['decision_rows']}`\n")
        out.write(f"- shadow records: `{summary['shadow_records']}`\n")
        out.write(f"- unique source docs: `{summary['unique_source_docs']}`\n")
        out.write(f"- validation issues: `{summary['validation']['violation_count']}`\n")
        out.write(f"- shadow-link issues: `{summary['shadow_link_issue_count']}`\n")
        out.write(f"- source rows mutated: `false`\n\n")
        out.write("## By Action\n\n")
        out.write("| Action | Source docs | Chars removed | Tokens removed | Held-out chars | Held-out tokens | Shadow records |\n")
        out.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for action, payload in sorted(summary["by_action"].items()):
            out.write(
                "| `%s` | %s | %s | %s | %s | %s | %s |\n"
                % (
                    action,
                    payload.get("source_docs"),
                    payload.get("chars_removed"),
                    payload.get("tokens_removed"),
                    payload.get("held_out_chars"),
                    payload.get("held_out_tokens"),
                    payload.get("shadow_records"),
                )
            )
        out.write("\n## Inputs\n\n")
        for path_text, rows in sorted(summary["input_decision_manifest_rows"].items()):
            out.write(f"- decision manifest `{path_text}`: `{rows}` rows\n")
        for path_text, rows in sorted(summary["input_shadow_record_rows"].items()):
            out.write(f"- shadow records `{path_text}`: `{rows}` rows\n")


def make_overlay(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    timestamp = args.timestamp or utc_timestamp()
    output_prefix = Path(args.output_prefix)
    decision_paths, shadow_paths = find_inputs(reports_dir)
    decision_rows, duplicate_decisions, decision_source_counts = load_deduped(decision_paths, "source_manifest_path")
    shadow_rows, duplicate_shadows, shadow_source_counts = load_deduped(shadow_paths, "source_shadow_records_path")

    manifest_path = Path(f"{output_prefix}_decision_manifest.jsonl")
    shadow_path = Path(f"{output_prefix}_shadow_records.jsonl")
    summary_path = Path(f"{output_prefix}_summary.json")
    validation_path = Path(f"{output_prefix}_validation.json")
    md_path = Path(f"{output_prefix}.md")

    write_jsonl(manifest_path, decision_rows)
    write_jsonl(shadow_path, shadow_rows)

    validation = validate_shadow_manifest.validate([manifest_path])
    shadow_link_issues = validate_shadow_records(decision_rows, shadow_rows)
    validation["shadow_link_issues"] = shadow_link_issues[:200]
    validation["shadow_link_issue_count"] = len(shadow_link_issues)
    write_json(validation_path, validation)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": timestamp,
        "policy_note": "Reviewed overlay only. Source HPLT rows are immutable; changed text exists only in shadow_records. This does not approve broad detector actions.",
        "reports_dir": str(reports_dir),
        "decision_manifest_jsonl": str(manifest_path),
        "shadow_records_jsonl": str(shadow_path),
        "validation_json": str(validation_path),
        "markdown": str(md_path),
        "input_decision_manifests": [str(path) for path in decision_paths],
        "input_shadow_records": [str(path) for path in shadow_paths],
        "input_decision_manifest_rows": decision_source_counts,
        "input_shadow_record_rows": shadow_source_counts,
        "duplicate_decision_rows_removed": duplicate_decisions,
        "duplicate_shadow_rows_removed": duplicate_shadows,
        "decision_rows": len(decision_rows),
        "shadow_records": len(shadow_rows),
        "unique_source_docs": len({source_doc_id(row) for row in decision_rows if source_doc_id(row)}),
        "by_action": action_summary(decision_rows),
        "by_error_action": error_action_summary(decision_rows),
        "validation": {
            "records": validation.get("records"),
            "materialized_records": validation.get("materialized_records"),
            "violation_count": validation.get("violation_count"),
            "by_action": validation.get("by_action"),
        },
        "shadow_link_issue_count": len(shadow_link_issues),
        "shadow_link_issues": shadow_link_issues[:200],
        "source_rows_mutated": False,
    }
    write_json(summary_path, summary)
    write_markdown(md_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output-prefix", default="reports/reviewed_shadow_overlay")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    summary = make_overlay(args)
    print(
        json.dumps(
            {
                "summary": summary["markdown"],
                "decision_manifest": summary["decision_manifest_jsonl"],
                "shadow_records": summary["shadow_records_jsonl"],
                "validation": summary["validation_json"],
                "decision_rows": summary["decision_rows"],
                "shadow_records_count": summary["shadow_records"],
                "validation_issues": summary["validation"]["violation_count"],
                "shadow_link_issues": summary["shadow_link_issue_count"],
            },
            sort_keys=True,
        )
    )
    if args.fail_on_issues and (
        int(summary["validation"]["violation_count"] or 0) > 0
        or int(summary["shadow_link_issue_count"] or 0) > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

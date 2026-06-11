#!/usr/bin/env python3
"""Validate and materialize reviewed HPLT boundary annotations.

The boundary-spec review pack contains full text and an annotation template.
This script is the bridge from completed boundary annotations to auditable
overlay artifacts: it validates reviewer-provided spans/split parts, computes
before/after checksums and char/token deltas, and writes decision manifests plus
duplicate/shadow records. It never rewrites source HPLT rows.

With the current unfilled template, run this in validation mode to show which
rows are still pending boundary review.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from tokenizers import Tokenizer
except Exception as exc:  # pragma: no cover - optional preflight catches this.
    Tokenizer = None
    TOKENIZERS_IMPORT_ERROR = exc
else:
    TOKENIZERS_IMPORT_ERROR = None


SCHEMA_VERSION = "hplt-boundary-annotation-materialization-v1"
DECISION_SCHEMA_VERSION = "hplt-cleaning-action-v1"
SHADOW_SCHEMA_VERSION = "hplt-cleaning-shadow-record-v1"

DEFAULT_REVIEW_PACK = "reports/boundary_spec_review_pack_20260606T002713Z.jsonl"
DEFAULT_ANNOTATIONS = "reports/boundary_spec_review_pack_20260606T002713Z_annotation_template.jsonl"
DEFAULT_OUTPUT_PREFIX = "reports/boundary_annotation_materialization"
DEFAULT_TOKENIZER_JSON = "/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json"
TIMESTAMP_SUFFIX_RE = re.compile(r"\d{8}T\d{6}Z$")

TEXT_CHANGING_ACTIONS = {
    "normalize_or_trim_span",
    "trim_prefix",
    "trim_suffix",
    "trim_span",
    "split_doc",
}
VALID_ACTIONS = TEXT_CHANGING_ACTIONS | {"drop_doc", "quarantine", "keep"}
PENDING_STATUSES = {"needs_boundary_spec", "pending", ""}
APPLY_STATUSES = {"reviewed_accept", "accepted", "apply", "applied"}
REJECT_STATUSES = {"reviewed_reject", "rejected", "keep"}
QUARANTINE_STATUSES = {"reviewed_quarantine", "quarantine"}


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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
            row["_line_number"] = line_number
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
            row = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_tokenizer(path: str | None) -> Any:
    if not path:
        return None
    if TOKENIZERS_IMPORT_ERROR is not None:
        raise RuntimeError(f"tokenizers import failed: {TOKENIZERS_IMPORT_ERROR}")
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise RuntimeError(f"tokenizer.json not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def count_tokens(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text).ids)


def normalize_status(value: Any) -> str:
    return compact_text(value).strip()


def normalize_action(row: dict[str, Any]) -> str:
    return compact_text(row.get("boundary_review_action") or row.get("reviewed_action") or row.get("action")).strip()


def add_issue(issues: list[dict[str, Any]], row: dict[str, Any], code: str, detail: str) -> None:
    issues.append(
        {
            "source_doc_id": row.get("source_doc_id"),
            "line_number": row.get("_line_number"),
            "code": code,
            "detail": detail,
        }
    )


def check_int(value: Any, field: str, row: dict[str, Any], issues: list[dict[str, Any]]) -> int | None:
    if isinstance(value, bool):
        add_issue(issues, row, "invalid_integer", f"{field} is boolean")
        return None
    if isinstance(value, int):
        return value
    add_issue(issues, row, "invalid_integer", f"{field}={value!r}")
    return None


def validate_ranges(
    ranges: list[dict[str, Any]],
    text_len: int,
    row: dict[str, Any],
    issues: list[dict[str, Any]],
    kind: str,
    allow_touching: bool = True,
) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    if not isinstance(ranges, list):
        add_issue(issues, row, "invalid_ranges", f"{kind} must be a list")
        return clean
    for index, item in enumerate(ranges):
        if not isinstance(item, dict):
            add_issue(issues, row, "invalid_range", f"{kind}[{index}] is not an object")
            continue
        start = check_int(item.get("start"), f"{kind}[{index}].start", row, issues)
        end = check_int(item.get("end"), f"{kind}[{index}].end", row, issues)
        if start is None or end is None:
            continue
        if start < 0 or end < 0 or start >= end or end > text_len:
            add_issue(issues, row, "range_out_of_bounds", f"{kind}[{index}] start={start} end={end} text_len={text_len}")
            continue
        clean.append(dict(item, start=start, end=end))

    clean.sort(key=lambda item: (item["start"], item["end"]))
    previous_end: int | None = None
    for item in clean:
        if previous_end is not None:
            overlaps = item["start"] < previous_end if allow_touching else item["start"] <= previous_end
            if overlaps:
                add_issue(issues, row, "overlapping_ranges", f"{kind} ranges overlap near start={item['start']}")
        previous_end = item["end"]
    return clean


def apply_span_ranges(text: str, spans: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item["start"]):
        pieces.append(text[cursor:span["start"]])
        pieces.append(compact_text(span.get("replacement")))
        cursor = span["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)


def derive_doc_id(source_doc_id: str, action: str, suffix: str, text_hash: str) -> str:
    suffix = suffix.strip(":") or "0"
    return f"{source_doc_id}::shadow::{action}::{suffix}::{text_hash[:12]}"


def base_decision(
    source: dict[str, Any],
    annotation: dict[str, Any],
    action: str,
    created_at_utc: str,
    policy_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_doc_id": source["source_doc_id"],
        "parent_source_doc_id": source["source_doc_id"],
        "derived_doc_id": None,
        "is_shadow_record": False,
        "doc_key": source.get("doc_key"),
        "source_dataset": source.get("source_dataset"),
        "url": source.get("url"),
        "host": source.get("host"),
        "quality_bin": source.get("quality_bin"),
        "text_sha256_before": source.get("full_text_sha256") or annotation.get("text_sha256_before"),
        "error_type_ids": annotation.get("error_type_ids") or source.get("error_type_ids") or [],
        "action": action,
        "action_status": "reviewed_accept",
        "confidence": None,
        "reason_codes": source.get("reason_codes") or [],
        "detector_scores": source.get("detector_scores") or {},
        "span_ranges": annotation.get("span_ranges") or [],
        "split_parts": annotation.get("split_parts") or [],
        "text_sha256_after": None,
        "chars_before": source.get("chars_before"),
        "chars_after": None,
        "chars_removed": None,
        "tokens_before": source.get("tokens_before"),
        "tokens_after": None,
        "tokens_removed": None,
        "good_text_loss_estimate": annotation.get("good_text_loss_estimate"),
        "sample_set": source.get("sample_set") or [],
        "evidence_excerpt": source.get("evidence_excerpt"),
        "doc_text_path": source.get("doc_text_path"),
        "review_label": "boundary_reviewed",
        "reviewer": annotation.get("reviewer"),
        "review_notes": annotation.get("boundary_notes"),
        "policy_id": "boundary_annotation_materialization",
        "policy_version": policy_version,
        "source_annotation_files": source.get("source_annotation_files") or [],
    }


def materialize_trim(
    source: dict[str, Any],
    annotation: dict[str, Any],
    action: str,
    tokenizer: Any,
    created_at_utc: str,
    policy_version: str,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = compact_text(source.get("full_text"))
    spans = validate_ranges(annotation.get("span_ranges") or [], len(text), annotation, issues, "span_ranges")
    if not spans:
        add_issue(issues, annotation, "missing_span_ranges", f"{action} requires at least one span")
        return [], []
    if issues:
        return [], []
    text_after = apply_span_ranges(text, spans)
    before_hash = sha256_text(text)
    after_hash = sha256_text(text_after)
    tokens_before = count_tokens(tokenizer, text)
    tokens_after = count_tokens(tokenizer, text_after)
    decision = base_decision(source, annotation, action, created_at_utc, policy_version)
    decision.update(
        {
            "parent_source_doc_id": source["source_doc_id"],
            "derived_doc_id": derive_doc_id(source["source_doc_id"], action, "0", after_hash),
            "is_shadow_record": True,
            "action_status": "applied",
            "span_ranges": spans,
            "text_sha256_before": before_hash,
            "text_sha256_after": after_hash,
            "chars_before": len(text),
            "chars_after": len(text_after),
            "chars_removed": len(text) - len(text_after),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_removed": None if tokens_before is None or tokens_after is None else tokens_before - tokens_after,
        }
    )
    shadow = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_doc_id": source["source_doc_id"],
        "parent_source_doc_id": source["source_doc_id"],
        "derived_doc_id": decision["derived_doc_id"],
        "is_shadow_record": True,
        "action": action,
        "text_sha256_before": before_hash,
        "text_sha256_after": after_hash,
        "full_text": text_after,
    }
    return [decision], [shadow]


def materialize_split(
    source: dict[str, Any],
    annotation: dict[str, Any],
    tokenizer: Any,
    created_at_utc: str,
    policy_version: str,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text = compact_text(source.get("full_text"))
    split_parts = validate_ranges(annotation.get("split_parts") or [], len(text), annotation, issues, "split_parts")
    dropped_spans = validate_ranges(annotation.get("dropped_span_ranges") or [], len(text), annotation, issues, "dropped_span_ranges")
    if not split_parts:
        add_issue(issues, annotation, "missing_split_parts", "split_doc requires at least one retained split part")
        return [], []
    if issues:
        return [], []

    before_hash = sha256_text(text)
    tokens_before = count_tokens(tokenizer, text)
    child_texts: list[tuple[dict[str, Any], str, str, int | None]] = []
    for index, part in enumerate(split_parts):
        child_text = text[part["start"]:part["end"]]
        child_hash = sha256_text(child_text)
        child_tokens = count_tokens(tokenizer, child_text)
        child_texts.append((part, child_text, child_hash, child_tokens))

    chars_after_total = sum(len(child_text) for _, child_text, _, _ in child_texts)
    tokens_after_total: int | None = None
    if tokens_before is not None and all(child_tokens is not None for _, _, _, child_tokens in child_texts):
        tokens_after_total = sum(int(child_tokens) for _, _, _, child_tokens in child_texts)

    decisions: list[dict[str, Any]] = []
    shadows: list[dict[str, Any]] = []
    for index, (part, child_text, child_hash, child_tokens) in enumerate(child_texts):
        part_id = compact_text(part.get("part_id") or part.get("derived_doc_id") or str(index))
        decision = base_decision(source, annotation, "split_doc", created_at_utc, policy_version)
        decision.update(
            {
                "parent_source_doc_id": source["source_doc_id"],
                "derived_doc_id": derive_doc_id(source["source_doc_id"], "split_doc", part_id, child_hash),
                "is_shadow_record": True,
                "action_status": "applied",
                "split_parts": split_parts,
                "dropped_span_ranges": dropped_spans,
                "active_split_part": dict(part, part_index=index, part_id=part_id),
                "text_sha256_before": before_hash,
                "text_sha256_after": child_hash,
                "chars_before": len(text),
                "chars_after": chars_after_total,
                "chars_removed": len(text) - chars_after_total,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after_total,
                "tokens_removed": None if tokens_before is None or tokens_after_total is None else tokens_before - tokens_after_total,
            }
        )
        shadow = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "created_at_utc": created_at_utc,
            "source_doc_id": source["source_doc_id"],
            "parent_source_doc_id": source["source_doc_id"],
            "derived_doc_id": decision["derived_doc_id"],
            "is_shadow_record": True,
            "action": "split_doc",
            "split_part": dict(part, part_index=index, part_id=part_id),
            "text_sha256_before": before_hash,
            "text_sha256_after": child_hash,
            "full_text": child_text,
        }
        decisions.append(decision)
        shadows.append(shadow)
    return decisions, shadows


def materialize_drop_or_keep(
    source: dict[str, Any],
    annotation: dict[str, Any],
    action: str,
    tokenizer: Any,
    created_at_utc: str,
    policy_version: str,
) -> dict[str, Any]:
    text = compact_text(source.get("full_text"))
    tokens_before = count_tokens(tokenizer, text)
    decision = base_decision(source, annotation, action, created_at_utc, policy_version)
    if action == "drop_doc":
        decision.update(
            {
                "parent_source_doc_id": None,
                "action_status": "applied",
                "chars_before": len(text),
                "chars_after": 0,
                "chars_removed": len(text),
                "tokens_before": tokens_before,
                "tokens_after": 0 if tokens_before is not None else None,
                "tokens_removed": tokens_before,
            }
        )
    elif action == "quarantine":
        decision.update(
            {
                "action_status": "quarantine",
                "chars_before": len(text),
                "tokens_before": tokens_before,
                "held_out_chars": len(text),
                "held_out_tokens": tokens_before,
            }
        )
    else:
        decision.update(
            {
                "action_status": "reviewed_reject",
                "chars_before": len(text),
                "chars_after": len(text),
                "chars_removed": 0,
                "tokens_before": tokens_before,
                "tokens_after": tokens_before,
                "tokens_removed": 0,
            }
        )
    return decision


def status_bucket(status: str) -> str:
    if status in PENDING_STATUSES:
        return "pending"
    if status in APPLY_STATUSES:
        return "apply"
    if status in QUARANTINE_STATUSES:
        return "quarantine"
    if status in REJECT_STATUSES:
        return "reject"
    return "unknown"


def source_level_delta_totals(decisions: list[dict[str, Any]]) -> tuple[int, int]:
    """Aggregate text deltas once per source/action/hash.

    Split actions emit one decision row per retained child, and each child
    carries the source-level removed chars/tokens for auditability. Summaries
    must not multiply that source-level delta by the number of children.
    """
    seen: set[tuple[str, str, str]] = set()
    chars_removed = 0
    tokens_removed = 0
    for row in decisions:
        if row.get("action") == "quarantine":
            continue
        key = (
            compact_text(row.get("source_doc_id")),
            compact_text(row.get("action")),
            compact_text(row.get("text_sha256_before")),
        )
        if key in seen:
            continue
        seen.add(key)
        chars_removed += int(row.get("chars_removed") or 0)
        if row.get("tokens_removed") is not None:
            tokens_removed += int(row.get("tokens_removed") or 0)
    return chars_removed, tokens_removed


def validate_and_materialize(args: argparse.Namespace) -> dict[str, Any]:
    review_pack_rows = read_jsonl(Path(args.review_pack))
    annotations = read_jsonl(Path(args.annotations))
    source_by_id = {compact_text(row.get("source_doc_id")): row for row in review_pack_rows}
    tokenizer: Any = None

    def get_tokenizer() -> Any:
        nonlocal tokenizer
        if tokenizer is None and args.tokenizer_json:
            tokenizer = load_tokenizer(args.tokenizer_json)
        return tokenizer

    created_at_utc = args.timestamp
    decisions: list[dict[str, Any]] = []
    shadow_records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    status_counts: collections.Counter[str] = collections.Counter()
    action_counts: collections.Counter[str] = collections.Counter()

    for annotation in annotations:
        source_doc_id = compact_text(annotation.get("source_doc_id"))
        source = source_by_id.get(source_doc_id)
        status = normalize_status(annotation.get("boundary_review_status"))
        bucket = status_bucket(status)
        action = normalize_action(annotation)
        status_counts[bucket] += 1
        action_counts[action or "missing"] += 1

        if source is None:
            add_issue(issues, annotation, "source_doc_not_in_review_pack", source_doc_id)
            continue
        expected_hash = compact_text(annotation.get("text_sha256_before"))
        actual_hash = compact_text(source.get("full_text_sha256") or sha256_text(compact_text(source.get("full_text"))))
        if expected_hash and expected_hash != actual_hash:
            add_issue(issues, annotation, "text_hash_mismatch", f"expected={expected_hash} actual={actual_hash}")
            continue
        if not action:
            add_issue(issues, annotation, "missing_action", "boundary_review_action/reviewed_action/action is required")
            continue
        if action not in VALID_ACTIONS:
            add_issue(issues, annotation, "unknown_action", action)
            continue

        if bucket == "pending":
            pending_rows.append(
                {
                    "source_doc_id": source_doc_id,
                    "action": action,
                    "boundary_review_status": status or "needs_boundary_spec",
                }
            )
            continue
        if bucket == "unknown":
            add_issue(issues, annotation, "unknown_boundary_review_status", status)
            continue
        if bucket == "reject":
            skipped_rows.append(
                {
                    "source_doc_id": source_doc_id,
                    "action": action,
                    "boundary_review_status": status,
                }
            )
            if action == "keep":
                decisions.append(materialize_drop_or_keep(source, annotation, "keep", get_tokenizer(), created_at_utc, args.policy_version))
            continue
        if bucket == "quarantine":
            decisions.append(materialize_drop_or_keep(source, annotation, "quarantine", get_tokenizer(), created_at_utc, args.policy_version))
            continue

        if not compact_text(annotation.get("boundary_notes")):
            add_issue(issues, annotation, "missing_boundary_notes", "accepted actions require boundary_notes")
            continue
        if annotation.get("good_text_loss_estimate") is None and action != "keep":
            add_issue(issues, annotation, "missing_good_text_loss_estimate", "accepted actions require good_text_loss_estimate")
            continue

        local_issues: list[dict[str, Any]] = []
        if action in {"trim_prefix", "trim_suffix", "trim_span", "normalize_or_trim_span"}:
            new_decisions, new_shadows = materialize_trim(
                source,
                annotation,
                action,
                get_tokenizer(),
                created_at_utc,
                args.policy_version,
                local_issues,
            )
        elif action == "split_doc":
            new_decisions, new_shadows = materialize_split(
                source,
                annotation,
                get_tokenizer(),
                created_at_utc,
                args.policy_version,
                local_issues,
            )
        else:
            new_decisions = [materialize_drop_or_keep(source, annotation, action, get_tokenizer(), created_at_utc, args.policy_version)]
            new_shadows = []
        if local_issues:
            issues.extend(local_issues)
            continue
        decisions.extend(new_decisions)
        shadow_records.extend(new_shadows)

    output_prefix = Path(args.output_prefix)
    if not TIMESTAMP_SUFFIX_RE.search(output_prefix.name):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    decision_path = Path(str(output_prefix) + "_decision_manifest.jsonl")
    shadow_path = Path(str(output_prefix) + "_shadow_records.jsonl")
    issues_path = Path(str(output_prefix) + "_issues.jsonl")
    summary_path = Path(str(output_prefix) + "_summary.json")
    md_path = Path(str(output_prefix) + ".md")

    write_jsonl(decision_path, decisions)
    write_jsonl(shadow_path, shadow_records)
    write_jsonl(issues_path, issues)

    token_delta_rows = sum(1 for row in decisions if row.get("tokens_removed") is not None)
    chars_removed, tokens_removed = source_level_delta_totals(decisions)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "policy_note": "Boundary annotation materialization only. Source HPLT rows are immutable; changed text exists only in shadow_records.",
        "review_pack": args.review_pack,
        "annotations": args.annotations,
        "decision_manifest_jsonl": str(decision_path),
        "shadow_records_jsonl": str(shadow_path),
        "issues_jsonl": str(issues_path),
        "markdown": str(md_path),
        "review_pack_rows": len(review_pack_rows),
        "annotation_rows": len(annotations),
        "pending_rows": len(pending_rows),
        "skipped_rows": len(skipped_rows),
        "decision_manifest_rows": len(decisions),
        "shadow_records": len(shadow_records),
        "issue_count": len(issues),
        "status_counts": dict(status_counts.most_common()),
        "action_counts": dict(action_counts.most_common()),
        "decision_actions": dict(collections.Counter(row.get("action") for row in decisions).most_common()),
        "chars_removed": chars_removed,
        "token_delta_rows": token_delta_rows,
        "tokens_removed": tokens_removed,
        "tokenizer_json": args.tokenizer_json,
    }
    write_json(summary_path, summary)
    write_markdown(md_path, summary, pending_rows, skipped_rows, issues)
    return summary


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    pending_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Boundary Annotation Materialization\n\n")
        out.write("This report validates boundary annotations and materializes only reviewed accepted actions. It does not mutate source HPLT rows.\n\n")
        out.write("## Summary\n\n")
        for key in [
            "review_pack_rows",
            "annotation_rows",
            "pending_rows",
            "skipped_rows",
            "decision_manifest_rows",
            "shadow_records",
            "issue_count",
            "chars_removed",
            "token_delta_rows",
            "tokens_removed",
        ]:
            out.write(f"- {key}: `{summary.get(key)}`\n")
        out.write("\n## Status Counts\n\n| Status | Rows |\n| --- | ---: |\n")
        for key, count in summary["status_counts"].items():
            out.write(f"| `{key}` | {count} |\n")
        out.write("\n## Decision Actions\n\n| Action | Rows |\n| --- | ---: |\n")
        for key, count in summary["decision_actions"].items():
            out.write(f"| `{key}` | {count} |\n")
        out.write("\n## Pending Rows\n\n")
        out.write("Pending rows need exact boundary review before materialization. Showing first 50.\n\n")
        for row in pending_rows[:50]:
            out.write(f"- `{row['source_doc_id']}` action=`{row['action']}` status=`{row['boundary_review_status']}`\n")
        if len(pending_rows) > 50:
            out.write(f"- ... {len(pending_rows) - 50} more pending rows\n")
        if skipped_rows:
            out.write("\n## Skipped Rows\n\n")
            for row in skipped_rows[:50]:
                out.write(f"- `{row['source_doc_id']}` action=`{row['action']}` status=`{row['boundary_review_status']}`\n")
        if issues:
            out.write("\n## Issues\n\n")
            for issue in issues[:100]:
                out.write(f"- `{issue.get('source_doc_id')}` line `{issue.get('line_number')}` `{issue.get('code')}`: {issue.get('detail')}\n")
            if len(issues) > 100:
                out.write(f"- ... {len(issues) - 100} more issues\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", default=DEFAULT_REVIEW_PACK)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--tokenizer-json", default=DEFAULT_TOKENIZER_JSON)
    parser.add_argument("--policy-version", default="20260606")
    parser.add_argument("--timestamp", default=utc_timestamp())
    args = parser.parse_args()

    summary = validate_and_materialize(args)
    print(
        json.dumps(
            {
                "summary": summary["markdown"].removesuffix(".md") + "_summary.json",
                "decision_manifest_rows": summary["decision_manifest_rows"],
                "shadow_records": summary["shadow_records"],
                "pending_rows": summary["pending_rows"],
                "issue_count": summary["issue_count"],
            },
            sort_keys=True,
        )
    )
    return 1 if summary["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize current HPLT cleaning evidence against policy gates.

This is a non-destructive audit. It reads local review annotations and reports
whether the current evidence is sufficient for a cleaning overlay. It does not
edit HPLT rows, generate shadow text, or approve any policy.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_ERROR_CATALOG = Path("ERROR_CATALOG.md")
DEFAULT_OUTPUT_PREFIX = "reports/policy_gate_audit"

CHANGING_ACTIONS = {
    "drop_doc",
    "normalize_or_trim_span",
    "quarantine",
    "split_doc",
    "trim_prefix",
    "trim_span",
    "trim_suffix",
}
KEEP_ACTION = "keep"
INTERVENTION_PRECISION_THRESHOLD = 0.95
ACTION_EXACTNESS_THRESHOLD = 0.95


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
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def percent(value: float) -> float:
    return value * 100.0


def parse_catalog_ids(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    all_ids = sorted(set(re.findall(r"\bE\d{3}\b", text)))

    confirmed_text = ""
    candidate_text = ""
    confirmed_match = re.search(r"## Confirmed Error Types(?P<body>.*?)(?:\n## |\Z)", text, re.S)
    candidate_match = re.search(r"## Candidate Types To Test(?P<body>.*?)(?:\n## |\Z)", text, re.S)
    if confirmed_match:
        confirmed_text = confirmed_match.group("body")
    if candidate_match:
        candidate_text = candidate_match.group("body")

    return {
        "all_ids": all_ids,
        "confirmed_ids": sorted(set(re.findall(r"\bE\d{3}\b", confirmed_text))),
        "candidate_ids": sorted(set(re.findall(r"\bE\d{3}\b", candidate_text))),
    }


def annotation_files(reports_dir: Path) -> list[Path]:
    files = []
    for path in sorted(reports_dir.glob("*annotations*.jsonl")):
        name = path.name
        if name.startswith("boundary_"):
            continue
        if "annotation_template" in name:
            continue
        files.append(path)
    global200 = reports_dir / "false_negative_global200_review_20260606T023000Z.jsonl"
    if global200.exists() and global200 not in files:
        files.append(global200)
    return sorted(files)


def latest_candidate_summary(reports_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    summaries = sorted(reports_dir.glob("candidate_issue_summary_*.json"))
    if not summaries:
        return (None, {})
    path = summaries[-1]
    return (path, read_json(path))


def latest_e001_summary(reports_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    summaries = sorted(reports_dir.glob("e001_control_char_scan_*_summary.json"))
    if not summaries:
        return (None, {})
    path = summaries[-1]
    return (path, read_json(path))


def latest_e001_shadow_overlay(
    reports_dir: Path,
) -> tuple[Path | None, dict[str, Any], Path | None, dict[str, Any], Path | None, list[dict[str, Any]]]:
    summaries = sorted(reports_dir.glob("e001_shadow_overlay_*_summary.json"))
    if not summaries:
        return (None, {}, None, {}, None, [])

    summary_path = summaries[-1]
    summary = read_json(summary_path)
    base = summary_path.name.removesuffix("_summary.json")
    validation_path = reports_dir / f"{base}_manifest_validation.json"
    manifest_path = reports_dir / f"{base}_manifest.jsonl"
    validation = read_json(validation_path) if validation_path.exists() else {}
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    return (summary_path, summary, validation_path if validation_path.exists() else None, validation, manifest_path if manifest_path.exists() else None, manifest_rows)


def latest_reviewed_shadow_overlay(
    reports_dir: Path,
) -> tuple[Path | None, dict[str, Any], Path | None, dict[str, Any], Path | None, list[dict[str, Any]], Path | None]:
    summaries = sorted(
        reports_dir.glob("reviewed_shadow_overlay_*_summary.json"),
        key=lambda path: (read_json(path).get("created_at_utc") or path.name),
    )
    if not summaries:
        return (None, {}, None, {}, None, [], None)

    summary_path = summaries[-1]
    summary = read_json(summary_path)
    base = summary_path.name.removesuffix("_summary.json")
    validation_path = reports_dir / f"{base}_validation.json"
    manifest_path = reports_dir / f"{base}_decision_manifest.jsonl"
    shadow_path = reports_dir / f"{base}_shadow_records.jsonl"
    validation = read_json(validation_path) if validation_path.exists() else {}
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    return (
        summary_path,
        summary,
        validation_path if validation_path.exists() else None,
        validation,
        manifest_path if manifest_path.exists() else None,
        manifest_rows,
        shadow_path if shadow_path.exists() else None,
    )


def latest_boundary_spec_summary(reports_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    summaries = sorted(reports_dir.glob("boundary_spec_review_pack_*_summary.json"))
    if not summaries:
        return (None, {})
    path = summaries[-1]
    return (path, read_json(path))


def latest_boundary_materialization_artifacts(
    reports_dir: Path,
) -> tuple[Path | None, dict[str, Any], Path | None, list[dict[str, Any]]]:
    summary_paths = sorted(
        set(reports_dir.glob("boundary_*materialization_*_summary.json")),
        key=lambda path: (read_json(path).get("created_at_utc") or path.name),
    )
    if not summary_paths:
        return (None, {}, None, [])

    all_decision_rows: list[dict[str, Any]] = []
    summary_entries: list[dict[str, Any]] = []
    decision_paths: list[Path] = []
    latest_decision_path: Path | None = None
    def source_delta_totals(rows: list[dict[str, Any]]) -> tuple[int, int]:
        seen: set[tuple[str, str, str]] = set()
        chars_removed = 0
        tokens_removed = 0
        for row in rows:
            if row.get("action") == "quarantine":
                continue
            key = (
                row_source_doc_id(row),
                row_action(row),
                str(row.get("text_sha256_before") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            chars_removed += int(row.get("chars_removed") or 0)
            if row.get("tokens_removed") is not None:
                tokens_removed += int(row.get("tokens_removed") or 0)
        return chars_removed, tokens_removed

    for summary_path in summary_paths:
        summary = read_json(summary_path)
        base = summary_path.name.removesuffix("_summary.json")
        decision_path = reports_dir / f"{base}_decision_manifest.jsonl"
        decision_rows = read_jsonl(decision_path) if decision_path.exists() else []
        run_chars_removed, run_tokens_removed = source_delta_totals(decision_rows)
        if decision_path.exists():
            decision_paths.append(decision_path)
            latest_decision_path = decision_path
        all_decision_rows.extend(decision_rows)
        summary_entries.append(
            {
                "summary": str(summary_path),
                "decision_manifest": str(decision_path) if decision_path.exists() else None,
                "created_at_utc": summary.get("created_at_utc"),
                "decision_manifest_rows": len(decision_rows),
                "shadow_records": summary.get("shadow_records"),
                "issue_count": summary.get("issue_count"),
                "chars_removed": run_chars_removed,
                "tokens_removed": run_tokens_removed,
                "decision_actions": summary.get("decision_actions"),
            }
        )

    latest_summary_path = summary_paths[-1]
    latest_summary = read_json(latest_summary_path)
    aggregate_summary = dict(latest_summary)
    aggregate_summary.update(
        {
            "materialization_runs": len(summary_paths),
            "summary_paths": [str(path) for path in summary_paths],
            "decision_manifest_paths": [str(path) for path in decision_paths],
            "decision_manifest_rows": len(all_decision_rows),
            "shadow_records": sum(read_json(path).get("shadow_records") or 0 for path in summary_paths),
            "issue_count": sum(read_json(path).get("issue_count") or 0 for path in summary_paths),
            "chars_removed": source_delta_totals(all_decision_rows)[0],
            "tokens_removed": source_delta_totals(all_decision_rows)[1],
            "decision_actions": dict(collections.Counter(row.get("action") for row in all_decision_rows).most_common()),
            "materialization_run_summaries": summary_entries,
        }
    )
    return (latest_summary_path, aggregate_summary, latest_decision_path, all_decision_rows)


def latest_pending_boundary_pack_summary(reports_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    summaries = sorted(
        reports_dir.glob("pending_boundary_review_pack_*_summary.json"),
        key=lambda path: (read_json(path).get("created_at_utc") or path.name),
    )
    if not summaries:
        return (None, {})
    path = summaries[-1]
    return (path, read_json(path))


def collect_review_evidence(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    global_prevalence_by_doc: dict[str, dict[str, Any]] = {}
    for path in paths:
        file_rows = read_jsonl(path)
        if path.name.startswith("global_prevalence_review_annotations_"):
            for row in file_rows:
                doc_id = row_source_doc_id(row)
                if doc_id:
                    global_prevalence_by_doc[doc_id] = row
            files.append({"path": str(path), "rows": len(file_rows), "dedupe_family": "global_prevalence"})
            continue
        rows.extend(file_rows)
        files.append({"path": str(path), "rows": len(file_rows)})
    if global_prevalence_by_doc:
        rows.extend(global_prevalence_by_doc.values())
        files.append(
            {
                "path": "global_prevalence_review_annotations_*",
                "rows": len(global_prevalence_by_doc),
                "deduplicated": True,
            }
        )
    return rows, files


def true_error_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in (
        "review_true_error_type_ids",
        "true_error_type_ids",
        "review_false_negative_error_ids",
        "missed_true_error_type_ids",
    ):
        value = row.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).startswith("E"))
    return sorted(set(ids))


def action_precision(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        candidate = row.get("candidate_action")
        reviewed = row.get("review_correct_action")
        if not candidate or not reviewed:
            continue
        if candidate == KEEP_ACTION:
            continue
        by_action[str(candidate)].append(row)

    result: dict[str, dict[str, Any]] = {}
    for action, action_rows in sorted(by_action.items()):
        exact = sum(1 for row in action_rows if row.get("candidate_action") == row.get("review_correct_action"))
        intervention_warranted = sum(
            1
            for row in action_rows
            if str(row.get("review_correct_action") or "") in CHANGING_ACTIONS
        )
        true_false_positives = sum(
            1
            for row in action_rows
            if str(row.get("review_correct_action") or "") == KEEP_ACTION
            or str(row.get("review_label") or "") == "false_positive"
        )
        n = len(action_rows)
        exact_ci = wilson(exact, n)
        intervention_ci = wilson(intervention_warranted, n)
        exact_precision = float(exact) / float(n or 1)
        intervention_precision = float(intervention_warranted) / float(n or 1)
        result[action] = {
            "reviewed_rows": n,
            "exact_matches": exact,
            "exact_precision": exact_precision,
            "exact_wilson_95_ci": list(exact_ci),
            "exact_wilson_95_ci_percent": [percent(exact_ci[0]), percent(exact_ci[1])],
            "intervention_warranted_matches": intervention_warranted,
            "intervention_warranted_precision": intervention_precision,
            "intervention_warranted_wilson_95_ci": list(intervention_ci),
            "intervention_warranted_wilson_95_ci_percent": [percent(intervention_ci[0]), percent(intervention_ci[1])],
            "true_false_positives": true_false_positives,
            "exact_action_gate_status": "met" if exact_ci[0] >= ACTION_EXACTNESS_THRESHOLD else "not_met",
            "intervention_warranted_gate_status": "met" if intervention_ci[0] >= INTERVENTION_PRECISION_THRESHOLD else "not_met",
            "gate_status": (
                "met"
                if exact_ci[0] >= ACTION_EXACTNESS_THRESHOLD
                and intervention_ci[0] >= INTERVENTION_PRECISION_THRESHOLD
                else "not_met"
            ),
            "gate_reason": (
                "Intervention-warranted precision asks whether the row should be touched at all; "
                "exact-action precision asks whether this specific automatic action is correct. "
                "Both use Wilson lower-bound >= 95% before broad automatic use."
            ),
        }
    return result


def row_source_doc_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("policy_evaluation_sample_id") or "")


def row_action(row: dict[str, Any]) -> str:
    return str(row.get("review_correct_action") or row.get("candidate_action") or row.get("action") or "")


def missing_delta_rows(
    rows: list[dict[str, Any]],
    resolved_delta_keys: set[tuple[str, str, str]] | None = None,
    resolved_source_error_keys: set[tuple[str, str]] | None = None,
    resolved_source_action_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    resolved_delta_keys = resolved_delta_keys or set()
    resolved_source_error_keys = resolved_source_error_keys or set()
    resolved_source_action_keys = resolved_source_action_keys or set()
    missing: list[dict[str, Any]] = []
    for row in rows:
        action = row_action(row)
        if action not in CHANGING_ACTIONS:
            continue
        if row.get("chars_removed") is None or row.get("tokens_removed") is None:
            source_doc_id = row_source_doc_id(row)
            error_ids = true_error_ids(row)
            if error_ids and any(
                (source_doc_id, action, error_id) in resolved_delta_keys
                or (source_doc_id, error_id) in resolved_source_error_keys
                for error_id in error_ids
            ):
                continue
            if not error_ids and (source_doc_id, action) in resolved_source_action_keys:
                continue
            missing.append(row)
    return missing


def make_summary(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    catalog = parse_catalog_ids(Path(args.error_catalog))
    ann_paths = annotation_files(reports_dir)
    rows, file_summaries = collect_review_evidence(ann_paths)
    latest_summary_path, latest_summary = latest_candidate_summary(reports_dir)
    latest_e001_path, latest_e001 = latest_e001_summary(reports_dir)
    (
        latest_e001_shadow_path,
        latest_e001_shadow,
        latest_e001_shadow_validation_path,
        latest_e001_shadow_validation,
        latest_e001_shadow_manifest_path,
        latest_e001_shadow_manifest_rows,
    ) = latest_e001_shadow_overlay(reports_dir)
    (
        latest_reviewed_overlay_path,
        latest_reviewed_overlay,
        latest_reviewed_overlay_validation_path,
        latest_reviewed_overlay_validation,
        latest_reviewed_overlay_manifest_path,
        latest_reviewed_overlay_manifest_rows,
        latest_reviewed_overlay_shadow_path,
    ) = latest_reviewed_shadow_overlay(reports_dir)
    latest_boundary_spec_path, latest_boundary_spec = latest_boundary_spec_summary(reports_dir)
    (
        latest_boundary_materialization_path,
        latest_boundary_materialization_summary,
        latest_boundary_materialization_decision_path,
        latest_boundary_materialization_decision_rows,
    ) = latest_boundary_materialization_artifacts(reports_dir)
    latest_pending_boundary_pack_path, latest_pending_boundary_pack = latest_pending_boundary_pack_summary(reports_dir)

    e001_shadow_valid = (
        bool(latest_e001_shadow)
        and bool(latest_e001_shadow_validation)
        and int(latest_e001_shadow_validation.get("violation_count") or 0) == 0
        and int(latest_e001_shadow.get("materialized_rows") or 0) > 0
        and not latest_e001_shadow.get("hash_mismatches")
        and not latest_e001_shadow.get("missing_expected_source_doc_ids")
        and not latest_e001_shadow.get("unchanged_after_source_doc_ids")
    )
    e001_shadow_delta_keys: set[tuple[str, str, str]] = set()
    resolved_source_error_keys: set[tuple[str, str]] = set()
    resolved_source_action_keys: set[tuple[str, str]] = set()
    if e001_shadow_valid:
        for manifest_row in latest_e001_shadow_manifest_rows:
            if manifest_row.get("chars_removed") is None or manifest_row.get("tokens_removed") is None:
                continue
            source_doc_id = row_source_doc_id(manifest_row)
            action = row_action(manifest_row)
            for error_id in true_error_ids(manifest_row) or [str(item) for item in manifest_row.get("error_type_ids") or []]:
                e001_shadow_delta_keys.add((source_doc_id, action, error_id))
                resolved_source_error_keys.add((source_doc_id, error_id))
            resolved_source_action_keys.add((source_doc_id, action))
    reviewed_overlay_valid = (
        bool(latest_reviewed_overlay)
        and bool(latest_reviewed_overlay_validation)
        and int(latest_reviewed_overlay_validation.get("violation_count") or 0) == 0
        and int(latest_reviewed_overlay_validation.get("shadow_link_issue_count") or 0) == 0
        and int(latest_reviewed_overlay.get("decision_rows") or 0) > 0
        and latest_reviewed_overlay.get("source_rows_mutated") is False
    )
    boundary_materialization_delta_keys: set[tuple[str, str, str]] = set()
    for decision_row in latest_boundary_materialization_decision_rows:
        action = row_action(decision_row)
        has_delta = decision_row.get("chars_removed") is not None and decision_row.get("tokens_removed") is not None
        has_holdout_delta = action == "quarantine" and decision_row.get("held_out_chars") is not None and decision_row.get("held_out_tokens") is not None
        if not has_delta and not has_holdout_delta:
            continue
        source_doc_id = row_source_doc_id(decision_row)
        for error_id in true_error_ids(decision_row) or [str(item) for item in decision_row.get("error_type_ids") or []]:
            boundary_materialization_delta_keys.add((source_doc_id, action, error_id))
            resolved_source_error_keys.add((source_doc_id, error_id))
        resolved_source_action_keys.add((source_doc_id, action))
    resolved_delta_keys = e001_shadow_delta_keys | boundary_materialization_delta_keys

    reviewed_id_counts: collections.Counter[str] = collections.Counter()
    reviewed_unique_doc_ids: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        doc_id = row_source_doc_id(row)
        for error_id in true_error_ids(row):
            reviewed_id_counts[error_id] += 1
            if doc_id:
                reviewed_unique_doc_ids[error_id].add(doc_id)

    action_counts = collections.Counter(str(row.get("review_correct_action") or row.get("candidate_action") or "unknown") for row in rows)
    label_counts = collections.Counter(str(row.get("review_label") or "unknown") for row in rows)
    severity_counts = collections.Counter(str(row.get("review_severity") or "unknown") for row in rows)
    changing_missing_delta_before_shadow = missing_delta_rows(rows)
    changing_missing_delta = missing_delta_rows(
        rows,
        resolved_delta_keys,
        resolved_source_error_keys,
        resolved_source_action_keys,
    )

    latest_candidate_counts = dict(latest_summary.get("by_error_type") or {})
    reviewed_ids = sorted(reviewed_id_counts)
    missing_review_ids = sorted(set(catalog["candidate_ids"]) - set(reviewed_ids))
    candidate_no_current_hits = sorted(error_id for error_id in catalog["candidate_ids"] if latest_candidate_counts.get(error_id, 0) == 0)

    global200_summary_path = reports_dir / "false_negative_global200_review_20260606T023000Z_summary.json"
    global200_summary = read_json(global200_summary_path) if global200_summary_path.exists() else {}
    candidate_precision = action_precision(rows)
    intervention_gate_met = bool(candidate_precision) and all(
        payload.get("intervention_warranted_gate_status") == "met"
        for payload in candidate_precision.values()
    )
    exact_action_gate_met = bool(candidate_precision) and all(
        payload.get("exact_action_gate_status") == "met"
        for payload in candidate_precision.values()
    )

    gates = {
        "issue_discovery": {
            "status": "not_met" if missing_review_ids else "met",
            "catalog_candidate_ids": len(catalog["candidate_ids"]),
            "reviewed_issue_ids": len(reviewed_ids),
            "missing_review_ids": missing_review_ids,
            "e001_scan_summary": str(latest_e001_path) if latest_e001_path else None,
            "e001_scan_rows_scanned": latest_e001.get("rows_scanned"),
            "e001_scan_match_rows": latest_e001.get("match_rows"),
            "e001_scan_review_pack_rows": latest_e001.get("review_pack_rows"),
            "note": "Goal requires at least 20 distinct issue types, or reviewed proof that fewer actionable types exist.",
        },
        "destructive_precision": {
            "status": "met" if intervention_gate_met and exact_action_gate_met else "not_met",
            "intervention_warranted_gate_status": "met" if intervention_gate_met else "not_met",
            "exact_action_gate_status": "met" if exact_action_gate_met else "not_met",
            "intervention_precision_threshold": INTERVENTION_PRECISION_THRESHOLD,
            "action_exactness_threshold": ACTION_EXACTNESS_THRESHOLD,
            "candidate_action_precision": candidate_precision,
            "note": (
                "Changing actions need two distinct checks: intervention-warranted precision "
                "(would any non-keep intervention be justified?) and exact-action precision "
                "(is this specific automatic action correct?)."
            ),
        },
        "char_token_deltas": {
            "status": "not_met" if changing_missing_delta else "met",
            "changing_review_rows_missing_char_or_token_delta": len(changing_missing_delta),
            "changing_review_rows_missing_char_or_token_delta_before_shadow_evidence": len(changing_missing_delta_before_shadow),
            "changing_review_rows_missing_char_or_token_delta_by_action": dict(collections.Counter(str(row.get("review_correct_action") or row.get("candidate_action")) for row in changing_missing_delta).most_common()),
            "shadow_delta_rows_covered": len(changing_missing_delta_before_shadow) - len(changing_missing_delta),
            "e001_shadow_delta_rows_covered": len(e001_shadow_delta_keys),
            "boundary_materialization_delta_keys": len(boundary_materialization_delta_keys),
            "resolved_source_error_keys": len(resolved_source_error_keys),
            "resolved_source_action_keys": len(resolved_source_action_keys),
            "e001_shadow_chars_removed": latest_e001_shadow.get("chars_removed"),
            "e001_shadow_tokens_removed": latest_e001_shadow.get("tokens_removed"),
            "boundary_spec_summary": str(latest_boundary_spec_path) if latest_boundary_spec_path else None,
            "boundary_spec_unique_source_docs": latest_boundary_spec.get("unique_source_docs"),
            "boundary_spec_output_rows": latest_boundary_spec.get("output_rows"),
            "boundary_spec_source_review_rows_missing_deltas": latest_boundary_spec.get("source_review_rows_missing_deltas"),
            "boundary_spec_read_errors": latest_boundary_spec.get("read_error_count"),
            "boundary_spec_hash_mismatches": latest_boundary_spec.get("hash_mismatch_count"),
            "boundary_spec_tokenized_rows": latest_boundary_spec.get("tokenized_rows"),
            "boundary_materialization_summary": str(latest_boundary_materialization_path) if latest_boundary_materialization_path else None,
            "boundary_materialization_decision_manifest": str(latest_boundary_materialization_decision_path) if latest_boundary_materialization_decision_path else None,
            "boundary_materialization_runs": latest_boundary_materialization_summary.get("materialization_runs"),
            "boundary_materialization_summary_paths": latest_boundary_materialization_summary.get("summary_paths"),
            "boundary_materialization_decision_manifest_paths": latest_boundary_materialization_summary.get("decision_manifest_paths"),
            "boundary_materialization_pending_rows": latest_boundary_materialization_summary.get("pending_rows"),
            "boundary_materialization_decision_rows": latest_boundary_materialization_summary.get("decision_manifest_rows"),
            "boundary_materialization_decision_actions": latest_boundary_materialization_summary.get("decision_actions"),
            "boundary_materialization_shadow_records": latest_boundary_materialization_summary.get("shadow_records"),
            "boundary_materialization_issue_count": latest_boundary_materialization_summary.get("issue_count"),
            "boundary_materialization_chars_removed": latest_boundary_materialization_summary.get("chars_removed"),
            "boundary_materialization_tokens_removed": latest_boundary_materialization_summary.get("tokens_removed"),
            "pending_boundary_pack_summary": str(latest_pending_boundary_pack_path) if latest_pending_boundary_pack_path else None,
            "pending_boundary_pack_rows": latest_pending_boundary_pack.get("pending_rows"),
            "pending_boundary_pack_by_action": latest_pending_boundary_pack.get("pending_by_action"),
            "pending_boundary_pack_total_chars": latest_pending_boundary_pack.get("total_pending_chars"),
            "pending_boundary_pack_total_tokens": latest_pending_boundary_pack.get("total_pending_tokens"),
            "note": "The goal requires char/token deltas by error type and action before applying a policy.",
        },
        "remaining_false_negatives": {
            "status": "not_met",
            "actionable_false_negatives": global200_summary.get("false_negative_found_count"),
            "actionable_false_negative_wilson_95_ci_percent": global200_summary.get("false_negative_wilson_95_ci_percent"),
            "serious_false_negatives": global200_summary.get("serious_false_negative_count"),
            "serious_false_negative_wilson_95_ci_percent": global200_summary.get("serious_false_negative_wilson_95_ci_percent"),
            "note": "Representative FN evidence is now present, but serious residual FNs remain and must be reduced or explicitly accepted.",
        },
        "shadow_overlay": {
            "status": "met" if reviewed_overlay_valid else "not_met",
            "e001_shadow_summary": str(latest_e001_shadow_path) if latest_e001_shadow_path else None,
            "e001_shadow_manifest": str(latest_e001_shadow_manifest_path) if latest_e001_shadow_manifest_path else None,
            "e001_shadow_validation": str(latest_e001_shadow_validation_path) if latest_e001_shadow_validation_path else None,
            "e001_shadow_valid": e001_shadow_valid,
            "e001_shadow_materialized_rows": latest_e001_shadow.get("materialized_rows"),
            "e001_shadow_records": latest_e001_shadow.get("shadow_records"),
            "e001_shadow_chars_removed": latest_e001_shadow.get("chars_removed"),
            "e001_shadow_tokens_removed": latest_e001_shadow.get("tokens_removed"),
            "e001_shadow_validation_violations": latest_e001_shadow_validation.get("violation_count"),
            "reviewed_overlay_summary": str(latest_reviewed_overlay_path) if latest_reviewed_overlay_path else None,
            "reviewed_overlay_manifest": str(latest_reviewed_overlay_manifest_path) if latest_reviewed_overlay_manifest_path else None,
            "reviewed_overlay_shadow_records": str(latest_reviewed_overlay_shadow_path) if latest_reviewed_overlay_shadow_path else None,
            "reviewed_overlay_validation": str(latest_reviewed_overlay_validation_path) if latest_reviewed_overlay_validation_path else None,
            "reviewed_overlay_valid": reviewed_overlay_valid,
            "reviewed_overlay_decision_rows": latest_reviewed_overlay.get("decision_rows"),
            "reviewed_overlay_shadow_records_count": latest_reviewed_overlay.get("shadow_records"),
            "reviewed_overlay_unique_source_docs": latest_reviewed_overlay.get("unique_source_docs"),
            "reviewed_overlay_validation_violations": latest_reviewed_overlay_validation.get("violation_count"),
            "reviewed_overlay_shadow_link_issues": latest_reviewed_overlay_validation.get("shadow_link_issue_count"),
            "reviewed_overlay_source_rows_mutated": latest_reviewed_overlay.get("source_rows_mutated"),
            "note": "Reviewed duplicate/shadow overlay must exist, validate with zero manifest/shadow-link issues, and mutate no source HPLT rows. This gate does not override destructive-precision or false-negative gates.",
        },
    }

    if "E001" in missing_review_ids:
        e001_next = (
            "Manually review and label the E001 discovery pack, because the CPU-only scan found "
            f"{latest_e001.get('match_rows')} candidate rows in {latest_e001.get('rows_scanned')} scanned rows but E001 still has no reviewed proof yet."
            if latest_e001_path
            else "Run an E001-targeted discovery/negative-evidence scan on Clariden CPU-only xfer, because E001 has no reviewed proof yet."
        )
    else:
        e001_next = None

    pending_boundary_next = None
    if latest_pending_boundary_pack:
        if int(latest_pending_boundary_pack.get("pending_rows") or 0) > 0:
            pending_boundary_next = (
                "Fill exact boundary annotations for the pending-boundary review pack: "
                f"{latest_pending_boundary_pack.get('pending_rows')} source docs, "
                f"{latest_pending_boundary_pack.get('total_pending_tokens')} Apertus tokens, "
                f"actions {latest_pending_boundary_pack.get('pending_by_action')}."
            )
        else:
            pending_boundary_next = (
                "Exact-boundary review pack is closed: 0 pending source docs and 0 pending Apertus tokens remain."
            )

    next_required_evidence = [
        item
        for item in [
            e001_next,
            pending_boundary_next,
            "Continue materializing reviewed candidate transformations only as duplicate/shadow rows with before/after checksums; the E001 pilot is narrow and does not approve a full overlay.",
            "Compute chars/tokens removed by error type and action, plus good-text-loss estimates on representative samples.",
            "Re-review action-specific precision after tightening split/drop/quarantine rules; require at least 95% reviewed precision before promotion.",
            "Reduce or explicitly bound remaining serious false negatives, especially E010/E011 archive/topic/category rows.",
        ]
        if item
    ]

    return {
        "schema_version": "hplt-policy-gate-audit-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Audit only. This does not approve automatic cleaning or mutate source rows.",
        "inputs": {
            "error_catalog": str(args.error_catalog),
            "reports_dir": str(reports_dir),
            "annotation_files": file_summaries,
            "latest_candidate_summary": str(latest_summary_path) if latest_summary_path else None,
            "latest_e001_summary": str(latest_e001_path) if latest_e001_path else None,
            "latest_e001_shadow_overlay_summary": str(latest_e001_shadow_path) if latest_e001_shadow_path else None,
            "latest_e001_shadow_overlay_manifest": str(latest_e001_shadow_manifest_path) if latest_e001_shadow_manifest_path else None,
            "latest_e001_shadow_overlay_validation": str(latest_e001_shadow_validation_path) if latest_e001_shadow_validation_path else None,
            "latest_reviewed_shadow_overlay_summary": str(latest_reviewed_overlay_path) if latest_reviewed_overlay_path else None,
            "latest_reviewed_shadow_overlay_manifest": str(latest_reviewed_overlay_manifest_path) if latest_reviewed_overlay_manifest_path else None,
            "latest_reviewed_shadow_overlay_shadow_records": str(latest_reviewed_overlay_shadow_path) if latest_reviewed_overlay_shadow_path else None,
            "latest_reviewed_shadow_overlay_validation": str(latest_reviewed_overlay_validation_path) if latest_reviewed_overlay_validation_path else None,
            "latest_boundary_spec_summary": str(latest_boundary_spec_path) if latest_boundary_spec_path else None,
            "latest_boundary_materialization_summary": str(latest_boundary_materialization_path) if latest_boundary_materialization_path else None,
            "latest_boundary_materialization_decision_manifest": str(latest_boundary_materialization_decision_path) if latest_boundary_materialization_decision_path else None,
            "boundary_materialization_summary_paths": latest_boundary_materialization_summary.get("summary_paths"),
            "boundary_materialization_decision_manifest_paths": latest_boundary_materialization_summary.get("decision_manifest_paths"),
            "latest_pending_boundary_pack_summary": str(latest_pending_boundary_pack_path) if latest_pending_boundary_pack_path else None,
            "global200_summary": str(global200_summary_path) if global200_summary_path.exists() else None,
        },
        "catalog": catalog,
        "latest_candidate_error_counts": latest_candidate_counts,
        "candidate_ids_with_no_current_candidate_hits": candidate_no_current_hits,
        "reviewed_rows": len(rows),
        "reviewed_unique_source_docs": len(set(str(row.get("source_doc_id") or row.get("parent_source_doc_id") or "") for row in rows if row.get("source_doc_id") or row.get("parent_source_doc_id"))),
        "reviewed_true_error_type_counts": dict(reviewed_id_counts.most_common()),
        "reviewed_unique_doc_counts_by_error_type": {key: len(value) for key, value in sorted(reviewed_unique_doc_ids.items())},
        "reviewed_issue_ids": reviewed_ids,
        "missing_review_ids": missing_review_ids,
        "review_action_counts": dict(action_counts.most_common()),
        "review_label_counts": dict(label_counts.most_common()),
        "review_severity_counts": dict(severity_counts.most_common()),
        "candidate_action_precision": candidate_precision,
        "changing_review_rows_missing_char_or_token_delta": len(changing_missing_delta),
        "changing_review_rows_missing_char_or_token_delta_before_shadow_evidence": len(changing_missing_delta_before_shadow),
        "shadow_delta_rows_covered": len(changing_missing_delta_before_shadow) - len(changing_missing_delta),
        "e001_shadow_delta_rows_covered": len(e001_shadow_delta_keys),
        "boundary_materialization_delta_keys": len(boundary_materialization_delta_keys),
        "resolved_source_error_keys": len(resolved_source_error_keys),
        "resolved_source_action_keys": len(resolved_source_action_keys),
        "changing_review_rows_missing_char_or_token_delta_by_action": dict(collections.Counter(str(row.get("review_correct_action") or row.get("candidate_action")) for row in changing_missing_delta).most_common()),
        "gates": gates,
        "next_required_evidence": next_required_evidence,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# HPLT Cleaning Policy Gate Audit\n\n")
        out.write("This is a non-destructive evidence audit. It does not approve automatic cleaning and does not mutate source HPLT rows.\n\n")

        out.write("## Gate Status\n\n")
        out.write("| Gate | Status | Evidence |\n")
        out.write("| --- | --- | --- |\n")
        for gate, payload in summary["gates"].items():
            evidence = payload.get("note", "")
            if gate == "issue_discovery":
                e001_evidence = ""
                if payload.get("e001_scan_summary"):
                    e001_evidence = " E001 scan: %s candidate rows in %s scanned rows, review-pack rows %s." % (
                        payload.get("e001_scan_match_rows"),
                        payload.get("e001_scan_rows_scanned"),
                        payload.get("e001_scan_review_pack_rows"),
                    )
                evidence = "%s Reviewed %s/%s catalog issue IDs; missing: %s.%s" % (
                    evidence,
                    payload.get("reviewed_issue_ids"),
                    payload.get("catalog_candidate_ids"),
                    ", ".join(payload.get("missing_review_ids") or []) or "none",
                    e001_evidence,
                )
            elif gate == "char_token_deltas":
                evidence = "%s Missing deltas on %s changing-action review rows after %s rows resolved by validated E001 shadow evidence or boundary materialization; before those artifacts: %s." % (
                    evidence,
                    payload.get("changing_review_rows_missing_char_or_token_delta"),
                    payload.get("shadow_delta_rows_covered"),
                    payload.get("changing_review_rows_missing_char_or_token_delta_before_shadow_evidence"),
                )
                if payload.get("boundary_spec_summary"):
                    evidence += " Boundary-spec pack: %s unique docs, %s rows, read/hash errors %s/%s." % (
                        payload.get("boundary_spec_unique_source_docs"),
                        payload.get("boundary_spec_output_rows"),
                        payload.get("boundary_spec_read_errors"),
                        payload.get("boundary_spec_hash_mismatches"),
                    )
                if payload.get("boundary_materialization_summary"):
                    evidence += " Boundary materialization: runs=%s, pending(latest)=%s, decisions=%s, issues=%s, chars/tokens removed=%s/%s." % (
                        payload.get("boundary_materialization_runs"),
                        payload.get("boundary_materialization_pending_rows"),
                        payload.get("boundary_materialization_decision_rows"),
                        payload.get("boundary_materialization_issue_count"),
                        payload.get("boundary_materialization_chars_removed"),
                        payload.get("boundary_materialization_tokens_removed"),
                    )
                if payload.get("pending_boundary_pack_summary"):
                    evidence += " Pending boundary pack: %s rows, chars/tokens=%s/%s, by action=%s." % (
                        payload.get("pending_boundary_pack_rows"),
                        payload.get("pending_boundary_pack_total_chars"),
                        payload.get("pending_boundary_pack_total_tokens"),
                        payload.get("pending_boundary_pack_by_action"),
                    )
            elif gate == "remaining_false_negatives":
                evidence = "%s Serious FN: %s, CI %s." % (
                    evidence,
                    payload.get("serious_false_negatives"),
                    payload.get("serious_false_negative_wilson_95_ci_percent"),
                )
            elif gate == "shadow_overlay" and payload.get("e001_shadow_summary"):
                evidence = "%s E001 pilot: valid=%s, materialized=%s, violations=%s, chars/tokens removed=%s/%s." % (
                    evidence,
                    payload.get("e001_shadow_valid"),
                    payload.get("e001_shadow_materialized_rows"),
                    payload.get("e001_shadow_validation_violations"),
                    payload.get("e001_shadow_chars_removed"),
                    payload.get("e001_shadow_tokens_removed"),
                )
                if payload.get("reviewed_overlay_summary"):
                    evidence += " Reviewed overlay: valid=%s, decisions=%s, shadows=%s, source docs=%s, violations=%s, shadow-link issues=%s." % (
                        payload.get("reviewed_overlay_valid"),
                        payload.get("reviewed_overlay_decision_rows"),
                        payload.get("reviewed_overlay_shadow_records_count"),
                        payload.get("reviewed_overlay_unique_source_docs"),
                        payload.get("reviewed_overlay_validation_violations"),
                        payload.get("reviewed_overlay_shadow_link_issues"),
                    )
            out.write("| `%s` | `%s` | %s |\n" % (gate, payload.get("status"), evidence))

        out.write("\n## Evidence Loaded\n\n")
        out.write("- reviewed rows: `%s`\n" % summary["reviewed_rows"])
        out.write("- reviewed unique source docs: `%s`\n" % summary["reviewed_unique_source_docs"])
        out.write("- reviewed issue IDs: `%s`\n" % "`, `".join(summary["reviewed_issue_ids"]))
        out.write("- missing reviewed issue IDs: `%s`\n" % ("`, `".join(summary["missing_review_ids"]) or "none"))
        out.write("- candidate IDs with zero hits in latest candidate summary: `%s`\n\n" % ("`, `".join(summary["candidate_ids_with_no_current_candidate_hits"]) or "none"))
        latest_e001 = summary["inputs"].get("latest_e001_summary")
        if latest_e001:
            e001_gate = summary["gates"]["issue_discovery"]
            out.write("- latest E001 scan summary: `%s`\n" % latest_e001)
            out.write("- E001 scan rows/matches/review rows: `%s` / `%s` / `%s`\n\n" % (
                e001_gate.get("e001_scan_rows_scanned"),
                e001_gate.get("e001_scan_match_rows"),
                e001_gate.get("e001_scan_review_pack_rows"),
            ))
        latest_e001_shadow = summary["inputs"].get("latest_e001_shadow_overlay_summary")
        if latest_e001_shadow:
            shadow_gate = summary["gates"]["shadow_overlay"]
            out.write("- latest E001 shadow overlay summary: `%s`\n" % latest_e001_shadow)
            out.write("- latest E001 shadow overlay manifest: `%s`\n" % summary["inputs"].get("latest_e001_shadow_overlay_manifest"))
            out.write("- latest E001 shadow overlay validation: `%s`\n" % summary["inputs"].get("latest_e001_shadow_overlay_validation"))
            out.write("- E001 shadow materialized rows / validation violations: `%s` / `%s`\n" % (
                shadow_gate.get("e001_shadow_materialized_rows"),
                shadow_gate.get("e001_shadow_validation_violations"),
            ))
            out.write("- E001 shadow chars/tokens removed: `%s` / `%s`\n\n" % (
                shadow_gate.get("e001_shadow_chars_removed"),
                shadow_gate.get("e001_shadow_tokens_removed"),
            ))
        latest_reviewed_overlay = summary["inputs"].get("latest_reviewed_shadow_overlay_summary")
        if latest_reviewed_overlay:
            shadow_gate = summary["gates"]["shadow_overlay"]
            out.write("- latest reviewed shadow overlay summary: `%s`\n" % latest_reviewed_overlay)
            out.write("- latest reviewed shadow overlay manifest: `%s`\n" % summary["inputs"].get("latest_reviewed_shadow_overlay_manifest"))
            out.write("- latest reviewed shadow overlay shadow records: `%s`\n" % summary["inputs"].get("latest_reviewed_shadow_overlay_shadow_records"))
            out.write("- latest reviewed shadow overlay validation: `%s`\n" % summary["inputs"].get("latest_reviewed_shadow_overlay_validation"))
            out.write("- reviewed overlay valid / validation violations / shadow-link issues: `%s` / `%s` / `%s`\n" % (
                shadow_gate.get("reviewed_overlay_valid"),
                shadow_gate.get("reviewed_overlay_validation_violations"),
                shadow_gate.get("reviewed_overlay_shadow_link_issues"),
            ))
            out.write("- reviewed overlay decisions / shadows / source docs: `%s` / `%s` / `%s`\n\n" % (
                shadow_gate.get("reviewed_overlay_decision_rows"),
                shadow_gate.get("reviewed_overlay_shadow_records_count"),
                shadow_gate.get("reviewed_overlay_unique_source_docs"),
            ))
        latest_boundary = summary["inputs"].get("latest_boundary_spec_summary")
        if latest_boundary:
            delta_gate = summary["gates"]["char_token_deltas"]
            out.write("- latest boundary-spec summary: `%s`\n" % latest_boundary)
            out.write("- boundary-spec unresolved review rows / unique docs / output rows: `%s` / `%s` / `%s`\n" % (
                delta_gate.get("boundary_spec_source_review_rows_missing_deltas"),
                delta_gate.get("boundary_spec_unique_source_docs"),
                delta_gate.get("boundary_spec_output_rows"),
            ))
            out.write("- boundary-spec read errors / hash mismatches / tokenized rows: `%s` / `%s` / `%s`\n\n" % (
                delta_gate.get("boundary_spec_read_errors"),
                delta_gate.get("boundary_spec_hash_mismatches"),
                delta_gate.get("boundary_spec_tokenized_rows"),
            ))
        latest_boundary_materialization = summary["inputs"].get("latest_boundary_materialization_summary")
        if latest_boundary_materialization:
            delta_gate = summary["gates"]["char_token_deltas"]
            out.write("- latest boundary materialization summary: `%s`\n" % latest_boundary_materialization)
            out.write("- latest boundary materialization decision manifest: `%s`\n" % summary["inputs"].get("latest_boundary_materialization_decision_manifest"))
            out.write("- boundary materialization runs / pending(latest) / decisions / shadows / issues: `%s` / `%s` / `%s` / `%s` / `%s`\n" % (
                delta_gate.get("boundary_materialization_runs"),
                delta_gate.get("boundary_materialization_pending_rows"),
                delta_gate.get("boundary_materialization_decision_rows"),
                delta_gate.get("boundary_materialization_shadow_records"),
                delta_gate.get("boundary_materialization_issue_count"),
            ))
            out.write("- boundary materialization decision actions: `%s`\n\n" % delta_gate.get("boundary_materialization_decision_actions"))
        latest_pending_boundary = summary["inputs"].get("latest_pending_boundary_pack_summary")
        if latest_pending_boundary:
            delta_gate = summary["gates"]["char_token_deltas"]
            out.write("- latest pending-boundary review-pack summary: `%s`\n" % latest_pending_boundary)
            out.write("- pending boundary rows / chars / tokens: `%s` / `%s` / `%s`\n" % (
                delta_gate.get("pending_boundary_pack_rows"),
                delta_gate.get("pending_boundary_pack_total_chars"),
                delta_gate.get("pending_boundary_pack_total_tokens"),
            ))
            out.write("- pending boundary rows by action: `%s`\n\n" % delta_gate.get("pending_boundary_pack_by_action"))

        out.write("## Reviewed Issue Counts\n\n")
        out.write("| Error ID | Review rows | Unique docs |\n")
        out.write("| --- | ---: | ---: |\n")
        unique_counts = summary["reviewed_unique_doc_counts_by_error_type"]
        for error_id, count in summary["reviewed_true_error_type_counts"].items():
            out.write("| `%s` | %s | %s |\n" % (error_id, count, unique_counts.get(error_id, 0)))

        out.write("\n## Candidate Action Precision\n\n")
        out.write("These are mixed review records, not a final promotion sample. `Intervention warranted` means the candidate row should receive some non-keep action; `exact action` means the specific candidate action was correct.\n\n")
        out.write("| Candidate action | Reviewed rows | Intervention warranted | Intervention precision | Intervention Wilson 95% CI | Exact matches | Exact precision | Exact Wilson 95% CI | True FPs |\n")
        out.write("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |\n")
        for action, payload in summary["candidate_action_precision"].items():
            out.write(
                "| `%s` | %s | %s | %.2f%% | %.2f%% to %.2f%% | %s | %.2f%% | %.2f%% to %.2f%% | %s |\n"
                % (
                    action,
                    payload["reviewed_rows"],
                    payload["intervention_warranted_matches"],
                    percent(payload["intervention_warranted_precision"]),
                    payload["intervention_warranted_wilson_95_ci_percent"][0],
                    payload["intervention_warranted_wilson_95_ci_percent"][1],
                    payload["exact_matches"],
                    percent(payload["exact_precision"]),
                    payload["exact_wilson_95_ci_percent"][0],
                    payload["exact_wilson_95_ci_percent"][1],
                    payload["true_false_positives"],
                )
            )

        out.write("\n## Missing Deltas\n\n")
        out.write("Changing-action review rows missing `chars_removed` or `tokens_removed`: `%s` after `%s` rows were resolved by validated E001 shadow evidence or boundary materialization. Before those artifacts the missing count was `%s`.\n\n" % (
            summary["changing_review_rows_missing_char_or_token_delta"],
            summary.get("shadow_delta_rows_covered"),
            summary.get("changing_review_rows_missing_char_or_token_delta_before_shadow_evidence"),
        ))
        out.write("| Action | Rows missing deltas |\n")
        out.write("| --- | ---: |\n")
        for action, count in summary["changing_review_rows_missing_char_or_token_delta_by_action"].items():
            out.write("| `%s` | %s |\n" % (action, count))

        shadow_gate = summary["gates"]["shadow_overlay"]
        if shadow_gate.get("e001_shadow_summary"):
            out.write("\n## E001 Shadow Overlay Pilot\n\n")
            out.write("This pilot materializes only the 32 reviewed E001 replacement-character rows as duplicate/shadow records. It does not approve a full HPLT cleaning overlay.\n\n")
            out.write("- summary: `%s`\n" % shadow_gate.get("e001_shadow_summary"))
            out.write("- manifest: `%s`\n" % shadow_gate.get("e001_shadow_manifest"))
            out.write("- validation: `%s`\n" % shadow_gate.get("e001_shadow_validation"))
            out.write("- valid duplicate/shadow invariants: `%s`\n" % shadow_gate.get("e001_shadow_valid"))
            out.write("- materialized rows: `%s`\n" % shadow_gate.get("e001_shadow_materialized_rows"))
            out.write("- validation violations: `%s`\n" % shadow_gate.get("e001_shadow_validation_violations"))
            out.write("- chars removed: `%s`\n" % shadow_gate.get("e001_shadow_chars_removed"))
            out.write("- tokens removed: `%s`\n" % shadow_gate.get("e001_shadow_tokens_removed"))
        if shadow_gate.get("reviewed_overlay_summary"):
            out.write("\n## Reviewed Shadow Overlay\n\n")
            out.write("This artifact aggregates reviewed/materialized keep-out and duplicate/shadow decisions into one validated overlay candidate. It does not promote broad detector actions and does not override the destructive-precision or false-negative gates.\n\n")
            out.write("- summary: `%s`\n" % shadow_gate.get("reviewed_overlay_summary"))
            out.write("- manifest: `%s`\n" % shadow_gate.get("reviewed_overlay_manifest"))
            out.write("- shadow records: `%s`\n" % shadow_gate.get("reviewed_overlay_shadow_records"))
            out.write("- validation: `%s`\n" % shadow_gate.get("reviewed_overlay_validation"))
            out.write("- valid duplicate/shadow invariants: `%s`\n" % shadow_gate.get("reviewed_overlay_valid"))
            out.write("- decision rows: `%s`\n" % shadow_gate.get("reviewed_overlay_decision_rows"))
            out.write("- shadow records: `%s`\n" % shadow_gate.get("reviewed_overlay_shadow_records_count"))
            out.write("- unique source docs: `%s`\n" % shadow_gate.get("reviewed_overlay_unique_source_docs"))
            out.write("- validation violations: `%s`\n" % shadow_gate.get("reviewed_overlay_validation_violations"))
            out.write("- shadow-link issues: `%s`\n" % shadow_gate.get("reviewed_overlay_shadow_link_issues"))

        delta_gate = summary["gates"]["char_token_deltas"]
        if delta_gate.get("boundary_spec_summary"):
            out.write("\n## Boundary-Spec Review Pack\n\n")
            out.write("This pack materializes full text for unresolved reviewed changing actions so exact `span_ranges`, `split_parts`, whole-doc holdout rationales, and eventual char/token deltas can be filled. It does not apply a cleaning policy.\n\n")
            out.write("- summary: `%s`\n" % delta_gate.get("boundary_spec_summary"))
            out.write("- unresolved review rows represented: `%s`\n" % delta_gate.get("boundary_spec_source_review_rows_missing_deltas"))
            out.write("- unique source docs: `%s`\n" % delta_gate.get("boundary_spec_unique_source_docs"))
            out.write("- output rows: `%s`\n" % delta_gate.get("boundary_spec_output_rows"))
            out.write("- read errors: `%s`\n" % delta_gate.get("boundary_spec_read_errors"))
            out.write("- hash mismatches: `%s`\n" % delta_gate.get("boundary_spec_hash_mismatches"))
            out.write("- tokenized rows: `%s`\n" % delta_gate.get("boundary_spec_tokenized_rows"))
        if delta_gate.get("boundary_materialization_summary"):
            out.write("\n## Boundary Materialization\n\n")
            out.write("The review-to-overlay materializer records completed boundary annotations as decision manifests and emits shadow records only for accepted text-changing actions. Whole-document drop/quarantine decisions do not create shadow text.\n\n")
            out.write("- summary: `%s`\n" % delta_gate.get("boundary_materialization_summary"))
            out.write("- decision manifest: `%s`\n" % summary["inputs"].get("latest_boundary_materialization_decision_manifest"))
            out.write("- materialization runs: `%s`\n" % delta_gate.get("boundary_materialization_runs"))
            out.write("- pending rows in latest run: `%s`\n" % delta_gate.get("boundary_materialization_pending_rows"))
            out.write("- cumulative decision manifest rows: `%s`\n" % delta_gate.get("boundary_materialization_decision_rows"))
            out.write("- cumulative decision actions: `%s`\n" % delta_gate.get("boundary_materialization_decision_actions"))
            out.write("- shadow records: `%s`\n" % delta_gate.get("boundary_materialization_shadow_records"))
            out.write("- issues: `%s`\n" % delta_gate.get("boundary_materialization_issue_count"))
            out.write("- chars/tokens removed from derived stream: `%s` / `%s`\n" % (
                delta_gate.get("boundary_materialization_chars_removed"),
                delta_gate.get("boundary_materialization_tokens_removed"),
            ))
        if delta_gate.get("pending_boundary_pack_summary"):
            out.write("\n## Pending Boundary Review Pack\n\n")
            out.write("This pack narrows the remaining boundary workload after whole-document `drop_doc`/`quarantine` decisions. It is navigation evidence only: it does not infer spans, split parts, or mutate source rows.\n\n")
            out.write("- summary: `%s`\n" % delta_gate.get("pending_boundary_pack_summary"))
            out.write("- pending rows: `%s`\n" % delta_gate.get("pending_boundary_pack_rows"))
            out.write("- pending chars/tokens: `%s` / `%s`\n" % (
                delta_gate.get("pending_boundary_pack_total_chars"),
                delta_gate.get("pending_boundary_pack_total_tokens"),
            ))
            out.write("- pending rows by action: `%s`\n" % delta_gate.get("pending_boundary_pack_by_action"))

        out.write("\n## Next Required Evidence\n\n")
        for item in summary["next_required_evidence"]:
            out.write("- %s\n" % item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--error-catalog", default=str(DEFAULT_ERROR_CATALOG))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    if not output_prefix.name.endswith(args.timestamp):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    summary = make_summary(args)
    json_path = Path(str(output_prefix) + ".json")
    md_path = Path(str(output_prefix) + ".md")
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "reviewed_rows": summary["reviewed_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the current HPLT cleaning policy recommendation report.

This is a synthesis report only. It reads reviewed annotations, gate audits,
and materialized duplicate/shadow manifests. It does not mutate HPLT rows and
does not create a cleaning overlay.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import build_policy_gate_audit as gate_audit


DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_ERROR_CATALOG = Path("ERROR_CATALOG.md")
DEFAULT_OUTPUT_PREFIX = "reports/policy_recommendation"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def latest_json(reports_dir: Path, pattern: str) -> tuple[Path | None, dict[str, Any]]:
    paths = sorted(
        reports_dir.glob(pattern),
        key=lambda path: (read_json(path).get("created_at_utc") or path.name),
    )
    if not paths:
        return None, {}
    return paths[-1], read_json(paths[-1])


def true_error_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in (
        "review_true_error_type_ids",
        "true_error_type_ids",
        "review_false_negative_error_ids",
        "missed_true_error_type_ids",
        "error_type_ids",
        "candidate_error_type_ids_reviewed_as_true",
    ):
        value = row.get(key)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).startswith("E"))
    return sorted(set(ids))


def row_source_doc_id(row: dict[str, Any]) -> str:
    return compact_text(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("policy_evaluation_sample_id"))


def row_action(row: dict[str, Any]) -> str:
    return compact_text(row.get("review_correct_action") or row.get("candidate_action") or row.get("action"))


def collect_review_rows(reports_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in gate_audit.annotation_files(reports_dir):
        for row in read_jsonl(path):
            enriched = dict(row)
            enriched["_annotation_file"] = str(path)
            rows.append(enriched)
    return rows


def load_decision_rows(reports_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    for path in sorted(reports_dir.glob("boundary_*materialization_*_decision_manifest.jsonl")):
        paths.append(str(path))
        rows.extend(read_jsonl(path))
    for path in sorted(reports_dir.glob("e001_shadow_overlay_*_manifest.jsonl")):
        paths.append(str(path))
        rows.extend(read_jsonl(path))
    return rows, paths


def source_delta_rows(decision_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse split children so each source/action/hash delta is counted once."""
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in decision_rows:
        action = compact_text(row.get("action"))
        source_doc_id = row_source_doc_id(row)
        key = (source_doc_id, action, compact_text(row.get("text_sha256_before")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def add_counter(counter: collections.Counter[str], value: Any) -> None:
    text = compact_text(value) or "unknown"
    counter[text] += 1


def aggregate_decisions(decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    collapsed = source_delta_rows(decision_rows)
    by_action: dict[str, dict[str, Any]] = {}
    by_error_action: dict[str, dict[str, dict[str, int]]] = {}
    good_text_loss: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    for row in collapsed:
        action = compact_text(row.get("action")) or "unknown"
        action_bucket = by_action.setdefault(
            action,
            {
                "source_docs": 0,
                "chars_removed": 0,
                "tokens_removed": 0,
                "held_out_chars": 0,
                "held_out_tokens": 0,
                "shadow_records": 0,
            },
        )
        action_bucket["source_docs"] += 1
        action_bucket["chars_removed"] += int(row.get("chars_removed") or 0)
        action_bucket["tokens_removed"] += int(row.get("tokens_removed") or 0)
        action_bucket["held_out_chars"] += int(row.get("held_out_chars") or 0)
        action_bucket["held_out_tokens"] += int(row.get("held_out_tokens") or 0)
        if row.get("is_shadow_record"):
            action_bucket["shadow_records"] += 1
        add_counter(good_text_loss[action], row.get("good_text_loss_estimate"))

        for error_id in true_error_ids(row):
            error_bucket = by_error_action.setdefault(error_id, {}).setdefault(
                action,
                {
                    "source_docs": 0,
                    "chars_removed": 0,
                    "tokens_removed": 0,
                    "held_out_chars": 0,
                    "held_out_tokens": 0,
                },
            )
            error_bucket["source_docs"] += 1
            error_bucket["chars_removed"] += int(row.get("chars_removed") or 0)
            error_bucket["tokens_removed"] += int(row.get("tokens_removed") or 0)
            error_bucket["held_out_chars"] += int(row.get("held_out_chars") or 0)
            error_bucket["held_out_tokens"] += int(row.get("held_out_tokens") or 0)

    return {
        "source_level_decision_rows": len(collapsed),
        "by_action": by_action,
        "by_error_action": by_error_action,
        "good_text_loss_by_action": {key: dict(value.most_common()) for key, value in sorted(good_text_loss.items())},
    }


def false_positive_examples(review_rows: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in review_rows:
        candidate = compact_text(row.get("candidate_action"))
        reviewed = compact_text(row.get("review_correct_action"))
        if not candidate or not reviewed or candidate == "keep" or candidate == reviewed:
            continue
        examples.append(
            {
                "source_doc_id": row_source_doc_id(row),
                "host": row.get("host"),
                "candidate_action": candidate,
                "review_correct_action": reviewed,
                "review_label": row.get("review_label"),
                "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
                "review_true_error_type_ids": true_error_ids(row),
                "review_false_positive_reason": row.get("review_false_positive_reason") or "",
                "review_notes": row.get("review_notes") or row.get("review_span_or_split_notes") or "",
                "annotation_file": row.get("_annotation_file"),
            }
        )
    examples.sort(key=lambda row: (row["candidate_action"], row.get("host") or "", row["source_doc_id"]))
    return examples[:limit]


def serious_false_negative_examples(review_rows: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in review_rows:
        if row.get("review_label") != "false_negative_found":
            continue
        if row.get("review_severity") != "serious":
            continue
        examples.append(
            {
                "source_doc_id": row_source_doc_id(row),
                "host": row.get("host"),
                "review_correct_action": row_action(row),
                "review_true_error_type_ids": true_error_ids(row),
                "review_good_text_loss_risk": row.get("review_good_text_loss_risk"),
                "review_notes": row.get("review_notes") or row.get("review_span_or_split_notes") or "",
                "annotation_file": row.get("_annotation_file"),
            }
        )
    examples.sort(key=lambda row: (row.get("host") or "", row["source_doc_id"]))
    return examples[:limit]


def review_good_text_loss(review_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_action: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in review_rows:
        action = row_action(row) or "unknown"
        if action == "keep":
            continue
        add_counter(by_action[action], row.get("review_good_text_loss_risk"))
    return {key: dict(value.most_common()) for key, value in sorted(by_action.items())}


def markdown_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_error_catalog(path: Path) -> dict[str, dict[str, Any]]:
    issues: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return issues
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| E"):
            continue
        cells = markdown_table_cells(line)
        if not cells or not re.fullmatch(r"E\d{3}", cells[0]):
            continue
        error_id = cells[0]
        bucket = issues.setdefault(error_id, {"error_id": error_id})
        if len(cells) >= 6 and cells[1] in {"confirmed", "candidate", "rejected", "merged"}:
            bucket["catalog_status"] = cells[1]
            bucket["confirmed_examples"] = cells[2]
            bucket["main_reviewed_action"] = cells[3]
            bucket["fp_fn_warning"] = cells[4]
            bucket["retrieval"] = cells[5]
        elif len(cells) >= 4 and "candidate_type" not in bucket:
            bucket["candidate_type"] = cells[1]
            bucket["main_action_to_test"] = cells[2]
            bucket["catalog_notes"] = cells[3]
    return issues


def issue_type_rows(
    catalog: dict[str, dict[str, Any]],
    reviewed_counts: dict[str, Any] | None,
    reviewed_unique_counts: dict[str, Any] | None,
    deltas_by_error: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reviewed_counts = reviewed_counts or {}
    reviewed_unique_counts = reviewed_unique_counts or {}
    all_ids = sorted(set(catalog) | set(reviewed_counts) | set(deltas_by_error))
    for error_id in all_ids:
        entry = dict(catalog.get(error_id) or {})
        by_action = deltas_by_error.get(error_id) or {}
        materialized_docs = sum(int(payload.get("source_docs") or 0) for payload in by_action.values())
        materialized_actions = ", ".join(
            f"{action}:{payload.get('source_docs')}"
            for action, payload in sorted(by_action.items())
        )
        entry.update(
            {
                "error_id": error_id,
                "reviewed_records": int(reviewed_counts.get(error_id) or 0),
                "reviewed_unique_docs": int(reviewed_unique_counts.get(error_id) or 0),
                "materialized_source_docs": materialized_docs,
                "materialized_actions": materialized_actions,
                "safe_status": (
                    "reviewed_materialized_only"
                    if materialized_docs
                    else "review_evidence_only"
                ),
            }
        )
        rows.append(entry)
    return rows


def action_recommendations() -> list[dict[str, str]]:
    return [
        {
            "action": "keep",
            "recommendation": "Keep reviewed-clean rows, but do not treat candidate keep as proof of cleanliness.",
            "evidence": "The 200-row candidate-keep control review found 43 actionable FNs and 12 serious FNs.",
        },
        {
            "action": "normalize_or_trim_span",
            "recommendation": "Use reviewed exact-boundary duplicate/shadow rows only; do not promote the broad detector.",
            "evidence": "Candidate precision is below the 95% gate; the narrow reviewed E001 pilot is materialized.",
        },
        {
            "action": "trim_prefix",
            "recommendation": "Use reviewed exact-boundary duplicate/shadow rows only.",
            "evidence": "Prefix trims were materialized from reviewed boundary annotations, not from an automatic rule.",
        },
        {
            "action": "trim_suffix",
            "recommendation": "Use reviewed exact-boundary duplicate/shadow rows only.",
            "evidence": "Suffix/footer/comment trims have the largest reviewed materialized repair volume, but broad suffix rules still need precision gating.",
        },
        {
            "action": "trim_span",
            "recommendation": "Do not use the broad detector automatically; use reviewed exact spans only.",
            "evidence": "Candidate precision is 3/36 exact matches, far below the destructive-action gate.",
        },
        {
            "action": "split_doc",
            "recommendation": "Use reviewed exact split boundaries only; continue treating archive/topic/list pages as the main residual FN risk.",
            "evidence": "Reviewed split child records exist, but serious FNs are dominated by missed split/list-page cases.",
        },
        {
            "action": "drop_doc",
            "recommendation": "Exclude only reviewed whole-doc drops from a derived stream; do not drop from detector hits alone.",
            "evidence": "Candidate drop precision is 42/74 exact matches, with many coherent or salvageable false positives.",
        },
        {
            "action": "quarantine",
            "recommendation": "Use as a manual holdout/review state, not as automatic exclusion.",
            "evidence": "Candidate quarantine precision is 45/132 exact matches and has high/medium good-text-loss risk.",
        },
    ]


def make_summary(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    error_catalog_path = Path(args.error_catalog)
    audit_path, audit = latest_json(reports_dir, "policy_gate_audit_*.json")
    if not audit_path:
        raise RuntimeError("No policy_gate_audit_*.json found")
    boundary_pack_path, boundary_pack = latest_json(reports_dir, "boundary_spec_review_pack_*_summary.json")
    fn_path, fn_summary = latest_json(reports_dir, "false_negative_global200_review_*_summary.json")
    policy_eval_path, policy_eval_summary = latest_json(reports_dir, "policy_evaluation_sample_summary_*.json")
    candidate_summary_path, candidate_summary = latest_json(reports_dir, "candidate_issue_summary_*.json")
    prevalence_path, prevalence_summary = latest_json(reports_dir, "global_prevalence_review_pack_*_summary.json")
    prevalence_annotation_path, prevalence_annotation_summary = latest_json(reports_dir, "global_prevalence_annotation_summary_*_summary.json")
    review_rows = collect_review_rows(reports_dir)
    decision_rows, decision_paths = load_decision_rows(reports_dir)
    decision_aggregate = aggregate_decisions(decision_rows)
    catalog = parse_error_catalog(error_catalog_path)
    gates = {key: value.get("status") for key, value in audit.get("gates", {}).items()}
    promote_full_overlay = all(
        gates.get(key) == "met"
        for key in ("issue_discovery", "char_token_deltas", "destructive_precision", "remaining_false_negatives", "shadow_overlay")
    )
    failed_gates = [key for key, status in sorted(gates.items()) if status != "met"]
    recommendation = {
        "apply_full_hplt_cleaning_overlay": promote_full_overlay,
        "recommendation": "do_not_apply_full_overlay" if not promote_full_overlay else "apply_reviewed_overlay",
        "rationale": (
            "Do not apply a full HPLT cleaning overlay because these gates are still not met: "
            + ", ".join(failed_gates)
            + "."
            if not promote_full_overlay
            else "All configured gates are met."
        ),
        "safe_current_use": (
            "Use the reviewed materialized duplicate/shadow decisions as auditable evidence or a manually scoped derived-stream pilot only; "
            "do not promote broad detector actions to the full HPLT slice."
        ),
    }

    return {
        "schema_version": "hplt-policy-recommendation-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Synthesis only. This does not mutate source HPLT rows or approve an automatic full overlay.",
        "inputs": {
            "policy_gate_audit": str(audit_path),
            "error_catalog": str(error_catalog_path),
            "boundary_spec_summary": str(boundary_pack_path) if boundary_pack_path else None,
            "false_negative_global200_summary": str(fn_path) if fn_path else None,
            "policy_evaluation_sample_summary": str(policy_eval_path) if policy_eval_path else None,
            "latest_candidate_summary": str(candidate_summary_path) if candidate_summary_path else None,
            "global_prevalence_review_coverage_summary": str(prevalence_path) if prevalence_path else None,
            "global_prevalence_annotation_summary": str(prevalence_annotation_path) if prevalence_annotation_path else None,
            "decision_manifest_paths": decision_paths,
        },
        "reviewed_rows": audit.get("reviewed_rows"),
        "reviewed_unique_source_docs": audit.get("reviewed_unique_source_docs"),
        "reviewed_issue_ids": audit.get("reviewed_issue_ids"),
        "reviewed_true_error_type_counts": audit.get("reviewed_true_error_type_counts"),
        "reviewed_unique_doc_counts_by_error_type": audit.get("reviewed_unique_doc_counts_by_error_type"),
        "review_action_counts": audit.get("review_action_counts"),
        "review_label_counts": audit.get("review_label_counts"),
        "gates": gates,
        "policy_evaluation_sample_summary": policy_eval_summary,
        "latest_candidate_summary": candidate_summary,
        "global_prevalence_review_coverage": prevalence_summary,
        "global_prevalence_annotation_summary": prevalence_annotation_summary,
        "candidate_action_precision": audit.get("candidate_action_precision"),
        "false_negative_estimate": audit.get("gates", {}).get("remaining_false_negatives"),
        "false_negative_global200_summary": fn_summary,
        "boundary_spec_latest": boundary_pack,
        "decision_aggregate": decision_aggregate,
        "issue_type_rows": issue_type_rows(
            catalog,
            audit.get("reviewed_true_error_type_counts"),
            audit.get("reviewed_unique_doc_counts_by_error_type"),
            decision_aggregate.get("by_error_action") or {},
        ),
        "action_recommendations": action_recommendations(),
        "review_good_text_loss_by_action": review_good_text_loss(review_rows),
        "false_positive_examples": false_positive_examples(review_rows),
        "serious_false_negative_examples": serious_false_negative_examples(review_rows),
        "recommendation": recommendation,
    }


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return "%.2f%%" % (float(value) * 100.0)


def compact_counts(counts: dict[str, Any] | None, limit: int = 8) -> str:
    if not counts:
        return "none"
    items = sorted(counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))
    return ", ".join(f"`{key}` {value}" for key, value in items[:limit])


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = summary["recommendation"]
    with path.open("w", encoding="utf-8") as out:
        out.write("# HPLT Cleaning Policy Recommendation\n\n")
        out.write("This report is a synthesis of reviewed evidence. It does not mutate HPLT rows and does not create a full cleaning overlay.\n\n")
        out.write("## Recommendation\n\n")
        out.write(f"- apply full HPLT cleaning overlay: `{str(rec['apply_full_hplt_cleaning_overlay']).lower()}`\n")
        out.write(f"- recommendation: `{rec['recommendation']}`\n")
        out.write(f"- rationale: {rec['rationale']}\n")
        out.write(f"- safe current use: {rec['safe_current_use']}\n\n")

        out.write("## Gate Status\n\n| Gate | Status |\n| --- | --- |\n")
        for gate, status in sorted(summary["gates"].items()):
            out.write(f"| `{gate}` | `{status}` |\n")

        sample = summary.get("policy_evaluation_sample_summary") or {}
        selection = sample.get("selection") or {}
        rationale = sample.get("sample_rationale") or {}
        out.write("\n## Sampling Scope\n\n")
        out.write("- policy-evaluation worklist rows: `%s`\n" % sample.get("records"))
        out.write("- current reviewed records / unique source docs: `%s` / `%s`\n" % (
            summary.get("reviewed_rows"),
            summary.get("reviewed_unique_source_docs"),
        ))
        prevalence = summary.get("global_prevalence_review_coverage") or {}
        if prevalence:
            out.write("- global-random prevalence frame: `%s` rows; reviewed unique rows `%s`; unreviewed rows `%s`\n" % (
                prevalence.get("global_prevalence_rows"),
                prevalence.get("reviewed_unique_rows"),
                prevalence.get("unreviewed_rows"),
            ))
            out.write("- next prevalence review pack: `%s` rows, selected as `%s`\n" % (
                prevalence.get("next_pack_rows"),
                prevalence.get("next_pack_selection"),
            ))
            prevalence_annotations = summary.get("global_prevalence_annotation_summary") or {}
            if prevalence_annotations:
                combined = prevalence_annotations.get("combined_with_base") or {}
                out.write("- latest prevalence annotation slice: `%s` reviewed rows from the 200-row pack; combined reviewed rows `%s`\n" % (
                    prevalence_annotations.get("reviewed_unique_rows_in_pack"),
                    combined.get("combined_reviewed_unique_rows"),
                ))
        else:
            out.write("- global-random prevalence worklist target: `%s` rows; this full prevalence sample has not been promoted to a final automatic-cleaning estimate\n" % selection.get("global_random_n"))
        out.write("- closed representative false-negative estimate: `200` candidate-keep global-random controls\n")
        out.write("- WDS-bin worklist target: `%s` rows per bin for bins `%s`\n" % (
            selection.get("wds_bin_n"),
            ", ".join(str(item) for item in selection.get("wds_bins") or []),
        ))
        out.write("- action/error precision worklist target: up to `%s` rows per action and `%s` rows per error ID\n" % (
            selection.get("action_precision_n"),
            selection.get("error_precision_n"),
        ))
        out.write("- sample-size rationale: %s\n" % (rationale.get("global_prevalence") or "n/a"))
        out.write("- destructive-rule rationale: %s\n\n" % (rationale.get("action_precision") or "n/a"))

        fn = summary.get("false_negative_estimate") or {}
        out.write("\n## How Dirty The Evidence Looks\n\n")
        out.write(
            "- reviewed records / source docs: `%s` / `%s`\n"
            % (summary.get("reviewed_rows"), summary.get("reviewed_unique_source_docs"))
        )
        out.write("- reviewed issue IDs: `%s`\n" % "`, `".join(summary.get("reviewed_issue_ids") or []))
        out.write("- reviewed actions: %s\n" % compact_counts(summary.get("review_action_counts")))
        out.write("- reviewed labels: %s\n" % compact_counts(summary.get("review_label_counts")))
        out.write(
            "- representative candidate-keep control FNs: `%s` actionable and `%s` serious out of `200`; serious Wilson 95%% CI `%s`\n"
            % (
                fn.get("actionable_false_negatives"),
                fn.get("serious_false_negatives"),
                fn.get("serious_false_negative_wilson_95_ci_percent"),
            )
        )
        if prevalence:
            out.write(
                "- global-prevalence coverage so far: `%s` actionable and `%s` serious rows among `%s` reviewed unique rows; actionable Wilson 95%% CI `%s`\n"
                % (
                    prevalence.get("reviewed_actionable_rows"),
                    prevalence.get("reviewed_serious_rows"),
                    prevalence.get("reviewed_unique_rows"),
                    prevalence.get("reviewed_actionable_wilson_95_ci_percent"),
                )
            )
            out.write("- prevalence caveat: `%s`\n" % prevalence.get("prevalence_caveat"))
            prevalence_annotations = summary.get("global_prevalence_annotation_summary") or {}
            if prevalence_annotations:
                combined = prevalence_annotations.get("combined_with_base") or {}
                out.write(
                    "- after latest prevalence annotation slice: `%s` actionable and `%s` serious rows among `%s` reviewed unique rows; actionable Wilson 95%% CI `%s`\n"
                    % (
                        combined.get("combined_actionable_rows"),
                        combined.get("combined_serious_rows"),
                        combined.get("combined_reviewed_unique_rows"),
                        combined.get("combined_actionable_wilson_95_ci_percent"),
                    )
                )
        out.write(
            "- interpretation: residual dirt is real after HPLT/local filtering, especially attached archive/topic/category/list rows, footer/comment/RSS/SEO suffixes, encoding/OCR residue, and low-main-text rows. The current evidence supports a conservative no-overlay recommendation, not a final HPLT-wide prevalence percentage.\n\n"
        )

        out.write("## Issue Types Found\n\n")
        out.write("| ID | Issue type | Reviewed records | Unique docs | Materialized source docs/actions | Safe status |\n")
        out.write("| --- | --- | ---: | ---: | --- | --- |\n")
        for row in summary.get("issue_type_rows") or []:
            out.write(
                "| `%s` | %s | %s | %s | %s | `%s` |\n"
                % (
                    row.get("error_id"),
                    row.get("candidate_type") or row.get("catalog_notes") or "",
                    row.get("reviewed_records"),
                    row.get("reviewed_unique_docs"),
                    row.get("materialized_actions") or "none",
                    row.get("safe_status"),
                )
            )

        out.write("\n## Action Recommendation\n\n")
        out.write("| Action | Recommendation | Evidence |\n")
        out.write("| --- | --- | --- |\n")
        for row in summary.get("action_recommendations") or []:
            out.write(
                "| `%s` | %s | %s |\n"
                % (row.get("action"), row.get("recommendation"), row.get("evidence"))
            )

        out.write("\n## Fix Safety\n\n")
        out.write("- approved automatic full-slice destructive rules: `none`\n")
        out.write("- safe current use: reviewed/materialized duplicate/shadow decisions only, or manually scoped derived-stream pilots using their manifests and checksums\n")
        out.write("- narrow materialized pilot: E001 replacement/control/private-use character cleanup for 32 reviewed rows\n")
        out.write("- blocked actions: broad `drop_doc`, `quarantine`, `trim_span`, and `normalize_or_trim_span` detector actions, because reviewed intervention-warranted precision and/or exact-action precision remain below the 95% promotion gates\n")
        out.write("- remaining risk: serious false negatives remain in representative controls, mostly `split_doc`/archive/list-page failures, plus a small number of rows better excluded from a derived stream\n\n")

        out.write("\n## Destructive Precision\n\n")
        out.write("`Intervention warranted` means the row should receive some non-keep action. `Exact action` means the candidate action itself was correct.\n\n")
        out.write("| Candidate action | Reviewed | Intervention warranted | Intervention precision | Intervention Wilson 95% CI | Exact | Exact precision | Exact Wilson 95% CI | True FPs |\n| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |\n")
        for action, payload in sorted((summary.get("candidate_action_precision") or {}).items()):
            intervention_ci = payload.get("intervention_warranted_wilson_95_ci_percent") or [None, None]
            exact_ci = payload.get("exact_wilson_95_ci_percent") or payload.get("wilson_95_ci_percent") or [None, None]
            intervention_precision = payload.get("intervention_warranted_precision")
            if intervention_precision is None:
                intervention_precision = payload.get("precision")
            exact_precision = payload.get("exact_precision")
            if exact_precision is None:
                exact_precision = payload.get("precision")
            out.write(
                "| `%s` | %s | %s | %.2f%% | %.2f%%-%.2f%% | %s | %.2f%% | %.2f%%-%.2f%% | %s |\n"
                % (
                    action,
                    payload.get("reviewed_rows"),
                    payload.get("intervention_warranted_matches"),
                    100.0 * float(intervention_precision or 0.0),
                    float(intervention_ci[0] or 0.0),
                    float(intervention_ci[1] or 0.0),
                    payload.get("exact_matches"),
                    100.0 * float(exact_precision or 0.0),
                    float(exact_ci[0] or 0.0),
                    float(exact_ci[1] or 0.0),
                    payload.get("true_false_positives"),
                )
            )

        out.write("\n## Remaining False Negatives\n\n")
        out.write(f"- actionable false negatives: `{fn.get('actionable_false_negatives')}`\n")
        out.write(f"- serious false negatives: `{fn.get('serious_false_negatives')}`\n")
        out.write(f"- actionable Wilson 95% CI: `{fn.get('actionable_false_negative_wilson_95_ci_percent')}`\n")
        out.write(f"- serious Wilson 95% CI: `{fn.get('serious_false_negative_wilson_95_ci_percent')}`\n\n")

        out.write("## Materialized Deltas By Action\n\n")
        out.write("| Action | Source docs | Chars removed | Tokens removed | Held-out chars | Held-out tokens |\n| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for action, payload in sorted(summary["decision_aggregate"]["by_action"].items()):
            out.write(
                "| `%s` | %s | %s | %s | %s | %s |\n"
                % (
                    action,
                    payload.get("source_docs"),
                    payload.get("chars_removed"),
                    payload.get("tokens_removed"),
                    payload.get("held_out_chars"),
                    payload.get("held_out_tokens"),
                )
            )

        out.write("\n## Materialized Deltas By Error And Action\n\n")
        out.write("The same source-level delta is attributed to each reviewed error ID on a decision, so error totals are not disjoint.\n\n")
        out.write("| Error | Action | Source docs | Chars removed | Tokens removed | Held-out chars | Held-out tokens |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for error_id, by_action in sorted(summary["decision_aggregate"]["by_error_action"].items()):
            for action, payload in sorted(by_action.items()):
                out.write(
                    "| `%s` | `%s` | %s | %s | %s | %s | %s |\n"
                    % (
                        error_id,
                        action,
                        payload.get("source_docs"),
                        payload.get("chars_removed"),
                        payload.get("tokens_removed"),
                        payload.get("held_out_chars"),
                        payload.get("held_out_tokens"),
                    )
                )

        out.write("\n## Good-Text-Loss Evidence\n\n")
        out.write("### Reviewed Rows\n\n")
        for action, counts in sorted(summary["review_good_text_loss_by_action"].items()):
            out.write(f"- `{action}`: `{counts}`\n")
        out.write("\n### Materialized Decisions\n\n")
        for action, counts in sorted(summary["decision_aggregate"]["good_text_loss_by_action"].items()):
            out.write(f"- `{action}`: `{counts}`\n")

        out.write("\n## False-Positive Examples\n\n")
        for row in summary["false_positive_examples"]:
            out.write(
                "- `%s` `%s`: candidate `%s` -> reviewed `%s`; errors `%s`; %s\n"
                % (
                    row["source_doc_id"],
                    row.get("host") or "",
                    row["candidate_action"],
                    row["review_correct_action"],
                    ",".join(row.get("review_true_error_type_ids") or []),
                    row.get("review_false_positive_reason") or row.get("review_notes") or "",
                )
            )

        out.write("\n## Serious False-Negative Examples\n\n")
        for row in summary["serious_false_negative_examples"]:
            out.write(
                "- `%s` `%s`: reviewed `%s`; errors `%s`; %s\n"
                % (
                    row["source_doc_id"],
                    row.get("host") or "",
                    row["review_correct_action"],
                    ",".join(row.get("review_true_error_type_ids") or []),
                    row.get("review_notes") or "",
                )
            )


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
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "recommendation": summary["recommendation"]["recommendation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

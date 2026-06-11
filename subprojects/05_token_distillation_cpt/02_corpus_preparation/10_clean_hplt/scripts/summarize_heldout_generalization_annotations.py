#!/usr/bin/env python3
"""Score held-out generalization annotations.

This summarizes fresh held-out review labels separately from the older
development/policy evidence. It does not mutate source data or approve a
cleaning overlay.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_HELDOUT_PACK = Path("reports/full_rule_count_20260606T105018Z_heldout_review_pack.jsonl")
DEFAULT_OUTPUT_PREFIX = Path("reports/heldout_generalization_audit")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pct(value: float) -> float:
    return 100.0 * value


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("heldout_review_id") or "")


def index_pack(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in (str(row.get("heldout_review_id") or ""), row_id(row)):
            if key:
                indexed[key] = row
    return indexed


def merge_annotation(annotation: dict[str, Any], pack_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(annotation.get("heldout_review_id") or "")
    row = pack_index.get(key)
    if row is None:
        row = pack_index.get(row_id(annotation))
    if row is None:
        return dict(annotation)
    merged = dict(row)
    merged.update(annotation)
    return merged


def reviewed_rows(annotation_paths: list[Path], pack_index: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in annotation_paths:
        file_rows = read_jsonl(path)
        kept = 0
        for raw in file_rows:
            if not raw.get("review_correct_action") and not raw.get("review_label"):
                continue
            row = merge_annotation(raw, pack_index)
            key = str(row.get("heldout_review_id") or row_id(row))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            rows.append(row)
            kept += 1
        files.append({"path": str(path), "rows": len(file_rows), "reviewed_rows": kept})
    return rows, files


def add_counter(counter: collections.Counter[str], value: Any) -> None:
    counter[str(value or "unknown")] += 1


def precision_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        candidate = str(row.get("candidate_action") or "")
        reviewed = str(row.get("review_correct_action") or "")
        if not candidate or not reviewed or candidate == KEEP_ACTION:
            continue
        by_action[candidate].append(row)

    out: dict[str, dict[str, Any]] = {}
    for action, action_rows in sorted(by_action.items()):
        n = len(action_rows)
        exact = sum(1 for row in action_rows if row.get("candidate_action") == row.get("review_correct_action"))
        intervention = sum(1 for row in action_rows if str(row.get("review_correct_action") or "") in CHANGING_ACTIONS)
        false_positive = sum(
            1
            for row in action_rows
            if str(row.get("review_correct_action") or "") == KEEP_ACTION
            or str(row.get("review_label") or "") == "false_positive"
        )
        exact_ci = wilson(exact, n)
        intervention_ci = wilson(intervention, n)
        out[action] = {
            "reviewed_rows": n,
            "intervention_warranted": intervention,
            "intervention_warranted_precision": intervention / float(n or 1),
            "intervention_warranted_wilson_95_ci_percent": [pct(intervention_ci[0]), pct(intervention_ci[1])],
            "exact_matches": exact,
            "exact_precision": exact / float(n or 1),
            "exact_wilson_95_ci_percent": [pct(exact_ci[0]), pct(exact_ci[1])],
            "true_false_positives": false_positive,
            "review_correct_action_counts": dict(collections.Counter(str(row.get("review_correct_action") or "unknown") for row in action_rows).most_common()),
        }
    return out


def false_negative_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    control_rows = [
        row
        for row in rows
        if str(row.get("candidate_action") or "") == KEEP_ACTION
        or "global_random_unseen" in (row.get("sample_set") or [])
        or "nohit_keep_control" in (row.get("sample_set") or [])
    ]
    fn_rows = [
        row
        for row in control_rows
        if str(row.get("review_label") or "") == "false_negative_found"
        or str(row.get("review_correct_action") or "") in CHANGING_ACTIONS
    ]
    serious_rows = [row for row in fn_rows if str(row.get("review_severity") or "") == "serious"]
    n = len(control_rows)
    fn_ci = wilson(len(fn_rows), n)
    serious_ci = wilson(len(serious_rows), n)
    return {
        "control_reviewed_rows": n,
        "false_negative_rows": len(fn_rows),
        "false_negative_wilson_95_ci_percent": [pct(fn_ci[0]), pct(fn_ci[1])],
        "serious_false_negative_rows": len(serious_rows),
        "serious_false_negative_wilson_95_ci_percent": [pct(serious_ci[0]), pct(serious_ci[1])],
        "false_negative_error_counts": dict(collections.Counter(error for row in fn_rows for error in (row.get("review_false_negative_error_ids") or row.get("review_true_error_type_ids") or [])).most_common()),
        "false_negative_action_counts": dict(collections.Counter(str(row.get("review_correct_action") or "unknown") for row in fn_rows).most_common()),
    }


def by_sample_set(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        sample_sets = row.get("sample_set") or ["unknown"]
        for sample_set in sample_sets:
            grouped[str(sample_set)].append(row)
    out: dict[str, dict[str, Any]] = {}
    for sample_set, sample_rows in sorted(grouped.items()):
        labels: collections.Counter[str] = collections.Counter()
        actions: collections.Counter[str] = collections.Counter()
        severity: collections.Counter[str] = collections.Counter()
        for row in sample_rows:
            add_counter(labels, row.get("review_label"))
            add_counter(actions, row.get("review_correct_action"))
            add_counter(severity, row.get("review_severity"))
        out[sample_set] = {
            "rows": len(sample_rows),
            "review_label_counts": dict(labels.most_common()),
            "review_correct_action_counts": dict(actions.most_common()),
            "review_severity_counts": dict(severity.most_common()),
        }
    return out


def good_text_loss(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_action: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        action = str(row.get("review_correct_action") or row.get("candidate_action") or "unknown")
        risk = str(row.get("review_good_text_loss_risk") or "unknown")
        by_action[action][risk] += 1
    return {action: dict(counter.most_common()) for action, counter in sorted(by_action.items())}


def make_summary(args: argparse.Namespace) -> dict[str, Any]:
    pack_rows = read_jsonl(args.heldout_pack)
    pack_index = index_pack(pack_rows)
    rows, files = reviewed_rows(args.annotations, pack_index)
    label_counts = collections.Counter(str(row.get("review_label") or "unknown") for row in rows)
    review_action_counts = collections.Counter(str(row.get("review_correct_action") or "unknown") for row in rows)
    candidate_action_counts = collections.Counter(str(row.get("candidate_action") or "unknown") for row in rows)
    return {
        "schema_version": "hplt-heldout-generalization-audit-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Held-out review scoring only. This does not mutate source rows or approve an overlay.",
        "inputs": {
            "heldout_pack": str(args.heldout_pack),
            "annotation_files": files,
        },
        "heldout_pack_rows": len(pack_rows),
        "reviewed_rows": len(rows),
        "reviewed_unique_source_docs": len(set(row_id(row) for row in rows if row_id(row))),
        "candidate_action_counts": dict(candidate_action_counts.most_common()),
        "review_label_counts": dict(label_counts.most_common()),
        "review_correct_action_counts": dict(review_action_counts.most_common()),
        "candidate_action_precision": precision_summary(rows),
        "false_negative_estimate": false_negative_summary(rows),
        "by_sample_set": by_sample_set(rows),
        "good_text_loss_by_review_action": good_text_loss(rows),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Held-Out Generalization Audit\n\n")
        out.write("This report scores fresh held-out review labels only. It does not mutate source HPLT rows and does not approve an overlay.\n\n")
        out.write(f"- held-out pack rows: `{summary['heldout_pack_rows']}`\n")
        out.write(f"- reviewed rows: `{summary['reviewed_rows']}`\n")
        out.write(f"- reviewed unique source docs: `{summary['reviewed_unique_source_docs']}`\n")
        out.write(f"- candidate actions reviewed: `{summary['candidate_action_counts']}`\n")
        out.write(f"- reviewed actions: `{summary['review_correct_action_counts']}`\n\n")

        out.write("## Candidate Action Precision\n\n")
        out.write("| Candidate action | Reviewed | Intervention warranted | Intervention precision | Intervention Wilson 95% CI | Exact | Exact precision | Exact Wilson 95% CI | True FPs |\n")
        out.write("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |\n")
        for action, payload in sorted(summary["candidate_action_precision"].items()):
            ici = payload["intervention_warranted_wilson_95_ci_percent"]
            eci = payload["exact_wilson_95_ci_percent"]
            out.write(
                f"| `{action}` | {payload['reviewed_rows']} | {payload['intervention_warranted']} | "
                f"{pct(payload['intervention_warranted_precision']):.2f}% | {ici[0]:.2f}%-{ici[1]:.2f}% | "
                f"{payload['exact_matches']} | {pct(payload['exact_precision']):.2f}% | {eci[0]:.2f}%-{eci[1]:.2f}% | "
                f"{payload['true_false_positives']} |\n"
            )

        fn = summary["false_negative_estimate"]
        out.write("\n## False Negatives\n\n")
        out.write(f"- control reviewed rows: `{fn['control_reviewed_rows']}`\n")
        out.write(f"- false negatives: `{fn['false_negative_rows']}`\n")
        out.write(f"- false-negative Wilson 95% CI: `{fn['false_negative_wilson_95_ci_percent']}`\n")
        out.write(f"- serious false negatives: `{fn['serious_false_negative_rows']}`\n")
        out.write(f"- serious false-negative Wilson 95% CI: `{fn['serious_false_negative_wilson_95_ci_percent']}`\n")
        out.write(f"- FN error counts: `{fn['false_negative_error_counts']}`\n\n")

        out.write("## By Sample Set\n\n")
        out.write("| Sample set | Rows | Labels | Actions | Severity |\n")
        out.write("| --- | ---: | --- | --- | --- |\n")
        for sample_set, payload in sorted(summary["by_sample_set"].items()):
            out.write(
                f"| `{sample_set}` | {payload['rows']} | `{payload['review_label_counts']}` | "
                f"`{payload['review_correct_action_counts']}` | `{payload['review_severity_counts']}` |\n"
            )

        out.write("\n## Good-Text-Loss Risk\n\n")
        for action, counts in sorted(summary["good_text_loss_by_review_action"].items()):
            out.write(f"- `{action}`: `{counts}`\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-pack", type=Path, default=DEFAULT_HELDOUT_PACK)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    args = parser.parse_args()

    summary = make_summary(args)
    json_path = args.output_prefix.with_name(f"{args.output_prefix.name}_{args.timestamp}.json")
    md_path = args.output_prefix.with_name(f"{args.output_prefix.name}_{args.timestamp}.md")
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(json.dumps({"summary_json": str(json_path), "markdown": str(md_path), "reviewed_rows": summary["reviewed_rows"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

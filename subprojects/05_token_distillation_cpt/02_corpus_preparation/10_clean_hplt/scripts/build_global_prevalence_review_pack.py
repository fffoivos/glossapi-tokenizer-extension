#!/usr/bin/env python3
"""Build coverage and next-review artifacts for the global prevalence frame.

This is non-destructive. It reads the policy-evaluation sample plus existing
review annotations, reports how much of the 1,067-row global prevalence frame
has review evidence, and emits a deterministic next review pack from the
unreviewed portion. It does not infer labels or approve a cleaning policy.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import build_policy_gate_audit as gate_audit


DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_POLICY_SAMPLE = Path("reports/policy_evaluation_sample_20260605T210154Z.jsonl")
DEFAULT_OUTPUT_PREFIX = "reports/global_prevalence_review_pack"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def source_doc_id(row: dict[str, Any]) -> str:
    return str(row.get("source_doc_id") or row.get("parent_source_doc_id") or row.get("policy_evaluation_sample_id") or "")


def source_shard(row: dict[str, Any]) -> str:
    doc_id = source_doc_id(row)
    parts = doc_id.split("::")
    return parts[1] if len(parts) >= 3 else "unknown"


def stable_key(text: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{text}".encode("utf-8", "replace")).hexdigest()


def risk_score(row: dict[str, Any]) -> float:
    scores = row.get("detector_scores") or {}
    values = [row.get("confidence")]
    values.extend(scores.values())
    numeric = []
    for value in values:
        try:
            if value is not None:
                numeric.append(float(value))
        except Exception:
            continue
    return max(numeric or [0.0])


def length_bucket(row: dict[str, Any]) -> str:
    chars = int(row.get("chars_before") or 0)
    if chars < 1000:
        return "lt_1k"
    if chars < 3000:
        return "1k_3k"
    if chars < 10000:
        return "3k_10k"
    return "ge_10k"


def quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    ordered = sorted(values)
    thresholds: list[float] = []
    for frac in (0.25, 0.50, 0.75):
        idx = min(len(ordered) - 1, max(0, math.ceil(frac * len(ordered)) - 1))
        thresholds.append(ordered[idx])
    return thresholds


def risk_band(value: float, thresholds: list[float]) -> str:
    if value <= thresholds[0]:
        return "risk_q1"
    if value <= thresholds[1]:
        return "risk_q2"
    if value <= thresholds[2]:
        return "risk_q3"
    return "risk_q4"


def wilson(successes: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n))) / denom
    return [max(0.0, center - half) * 100.0, min(1.0, center + half) * 100.0]


def collect_reviews(reports_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in gate_audit.annotation_files(reports_dir):
        for row in read_jsonl(path):
            doc_id = source_doc_id(row)
            if not doc_id:
                continue
            item = dict(row)
            item["_annotation_file"] = str(path)
            by_doc[doc_id].append(item)
    return by_doc


def review_action(row: dict[str, Any]) -> str:
    return str(row.get("review_correct_action") or row.get("candidate_action") or row.get("action") or "")


def review_true_error_ids(row: dict[str, Any]) -> list[str]:
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


def consolidate_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = collections.Counter(review_action(row) or "unknown" for row in rows)
    labels = collections.Counter(str(row.get("review_label") or "unknown") for row in rows)
    severities = collections.Counter(str(row.get("review_severity") or "unknown") for row in rows)
    error_ids: set[str] = set()
    files: set[str] = set()
    for row in rows:
        error_ids.update(review_true_error_ids(row))
        if row.get("_annotation_file"):
            files.add(str(row["_annotation_file"]))
    actionable = any((review_action(row) and review_action(row) != "keep") or row.get("review_label") in {"true_positive", "partial_true_positive", "false_negative_found"} for row in rows)
    serious = any(row.get("review_severity") == "serious" for row in rows)
    return {
        "review_records": len(rows),
        "review_actions": dict(actions.most_common()),
        "review_labels": dict(labels.most_common()),
        "review_severities": dict(severities.most_common()),
        "review_true_error_type_ids": sorted(error_ids),
        "review_files": sorted(files),
        "doc_has_actionable_error_review": actionable,
        "doc_has_serious_error_review": serious,
    }


def has_global_prevalence(row: dict[str, Any]) -> bool:
    return "global_prevalence_review" in as_list(row.get("evaluation_tasks")) or "global_prevalence" in as_list(row.get("sampling_strata"))


def counter_dict(rows: list[dict[str, Any]], key_fn) -> dict[str, int]:
    return dict(collections.Counter(str(key_fn(row)) for row in rows).most_common())


def allocate_stratified(rows: list[dict[str, Any]], pack_size: int, seed: int) -> list[dict[str, Any]]:
    if pack_size <= 0 or not rows:
        return []
    if len(rows) <= pack_size:
        return sorted(rows, key=lambda row: stable_key(source_doc_id(row), seed))
    strata: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        strata[(str(row.get("_quality_bin") or row.get("quality_bin") or "unknown"), str(row.get("_risk_band") or "unknown"))].append(row)
    total = len(rows)
    allocation: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for key, items in strata.items():
        exact = pack_size * len(items) / total
        base = min(len(items), int(math.floor(exact)))
        allocation[key] = base
        remainders.append((exact - base, key))
    remaining = pack_size - sum(allocation.values())
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    selected: list[dict[str, Any]] = []
    for key, count in allocation.items():
        selected.extend(sorted(strata[key], key=lambda row: stable_key(source_doc_id(row), seed))[:count])
    return sorted(selected, key=lambda row: stable_key(source_doc_id(row), seed))[:pack_size]


def read_text(path_text: str, limit: int | None = None) -> tuple[str, str | None]:
    if not path_text:
        return "", "missing_path"
    path = Path(path_text)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if limit and limit > 0:
        return text[:limit], None
    return text, None


def make_pack_row(row: dict[str, Any], full_text_chars: int) -> dict[str, Any]:
    text, error = read_text(str(row.get("doc_text_path") or ""), full_text_chars)
    return {
        "schema_version": "hplt-global-prevalence-review-pack-v1",
        "policy_evaluation_sample_id": row.get("policy_evaluation_sample_id"),
        "source_doc_id": source_doc_id(row),
        "parent_source_doc_id": row.get("parent_source_doc_id") or source_doc_id(row),
        "source_shard": source_shard(row),
        "source_dataset": row.get("source_dataset") or "HPLT/ell_Grek_ge8_no_mt_clean60",
        "url": row.get("url"),
        "host": row.get("host"),
        "quality_bin": row.get("quality_bin"),
        "chars_before": row.get("chars_before"),
        "text_sha256_before": row.get("text_sha256_before"),
        "candidate_action": row.get("candidate_action"),
        "candidate_error_type_ids": row.get("candidate_error_type_ids") or [],
        "confidence": row.get("confidence"),
        "risk_score": row.get("_risk_score"),
        "risk_band": row.get("_risk_band"),
        "length_bucket": row.get("_length_bucket"),
        "evaluation_tasks": row.get("evaluation_tasks") or [],
        "sampling_strata": row.get("sampling_strata") or [],
        "doc_text_path": row.get("doc_text_path"),
        "doc_text_read_error": error,
        "document_text": text,
        "review_label": None,
        "review_true_error_type_ids": [],
        "review_correct_action": None,
        "review_severity": None,
        "review_good_text_loss_risk": None,
        "review_notes": None,
    }


def write_markdown(path: Path, summary: dict[str, Any], pack_rows: list[dict[str, Any]], markdown_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        out.write("# Global Prevalence Review Coverage\n\n")
        out.write("This is a non-destructive coverage report and next-review pack for the 1,067-row `global_prevalence` frame. It does not label rows automatically and does not approve a cleaning overlay.\n\n")
        out.write("## Coverage\n\n")
        out.write("- global prevalence frame rows: `%s`\n" % summary["global_prevalence_rows"])
        out.write("- reviewed unique rows: `%s`\n" % summary["reviewed_unique_rows"])
        out.write("- unreviewed rows: `%s`\n" % summary["unreviewed_rows"])
        out.write("- reviewed actionable rows: `%s`\n" % summary["reviewed_actionable_rows"])
        out.write("- reviewed serious rows: `%s`\n" % summary["reviewed_serious_rows"])
        out.write("- reviewed actionable rate among covered rows: `%.2f%%`, Wilson 95%% CI `%s`\n" % (
            summary["reviewed_actionable_rate_percent"],
            summary["reviewed_actionable_wilson_95_ci_percent"],
        ))
        out.write("- caveat: `%s`\n\n" % summary["prevalence_caveat"])

        out.write("## Remaining Review Pack\n\n")
        out.write("- selected rows: `%s`\n" % summary["next_pack_rows"])
        out.write("- selection: `%s`\n" % summary["next_pack_selection"])
        out.write("- JSONL: `%s`\n" % summary["outputs"]["review_pack_jsonl"])
        out.write("- annotation template: `%s`\n\n" % summary["outputs"]["annotation_template_jsonl"])

        out.write("## Coverage By Stratum\n\n")
        for name, counts in summary["coverage_by"].items():
            out.write(f"### {name}\n\n")
            out.write("| Stratum | Total | Reviewed | Unreviewed |\n")
            out.write("| --- | ---: | ---: | ---: |\n")
            for key, payload in sorted(counts.items()):
                out.write("| `%s` | %s | %s | %s |\n" % (
                    key,
                    payload.get("total"),
                    payload.get("reviewed"),
                    payload.get("unreviewed"),
                ))
            out.write("\n")

        out.write("## Rows To Review\n\n")
        for index, row in enumerate(pack_rows, 1):
            text = (row.get("document_text") or row.get("evidence_excerpt") or "")[:markdown_chars]
            out.write("### %03d `%s`\n\n" % (index, row["source_doc_id"]))
            out.write("- host: `%s`\n" % (row.get("host") or ""))
            out.write("- url: `%s`\n" % (row.get("url") or ""))
            out.write("- quality/risk/length: `%s` / `%s` / `%s`\n" % (
                row.get("quality_bin"),
                row.get("risk_band"),
                row.get("length_bucket"),
            ))
            out.write("- candidate action/errors: `%s` / `%s`\n" % (
                row.get("candidate_action"),
                row.get("candidate_error_type_ids"),
            ))
            out.write("- review fields: `review_label`, `review_true_error_type_ids`, `review_correct_action`, `review_severity`, `review_good_text_loss_risk`, `review_notes`\n\n")
            out.write("```text\n%s\n```\n\n" % text)


def coverage_table(rows: list[dict[str, Any]], reviewed_doc_ids: set[str], key_fn) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(key_fn(row))
        bucket = result.setdefault(key, {"total": 0, "reviewed": 0, "unreviewed": 0})
        bucket["total"] += 1
        if source_doc_id(row) in reviewed_doc_ids:
            bucket["reviewed"] += 1
        else:
            bucket["unreviewed"] += 1
    return result


def make_summary_and_pack(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reports_dir = Path(args.reports_dir)
    policy_sample = Path(args.policy_sample)
    rows = [row for row in read_jsonl(policy_sample) if has_global_prevalence(row)]
    thresholds = quantile_thresholds([risk_score(row) for row in rows])
    for row in rows:
        score = risk_score(row)
        row["_risk_score"] = score
        row["_risk_band"] = risk_band(score, thresholds)
        row["_length_bucket"] = length_bucket(row)
        row["_source_shard"] = source_shard(row)
        row["_quality_bin"] = str(row.get("quality_bin") or "unknown")

    reviews_by_doc = collect_reviews(reports_dir)
    reviewed_doc_ids = set(reviews_by_doc)
    reviewed_rows = [row for row in rows if source_doc_id(row) in reviewed_doc_ids]
    unreviewed_rows = [row for row in rows if source_doc_id(row) not in reviewed_doc_ids]
    consolidated = {doc_id: consolidate_review(items) for doc_id, items in reviews_by_doc.items()}
    reviewed_actionable = sum(1 for row in reviewed_rows if consolidated[source_doc_id(row)]["doc_has_actionable_error_review"])
    reviewed_serious = sum(1 for row in reviewed_rows if consolidated[source_doc_id(row)]["doc_has_serious_error_review"])

    selected = allocate_stratified(unreviewed_rows, int(args.pack_size), int(args.seed))
    pack_rows = [make_pack_row(row, int(args.full_text_chars)) for row in selected]
    template_rows = [{key: value for key, value in row.items() if key != "document_text"} for row in pack_rows]
    n_reviewed = len(reviewed_rows)
    rate = reviewed_actionable / n_reviewed if n_reviewed else 0.0

    summary = {
        "schema_version": "hplt-global-prevalence-review-coverage-v1",
        "created_at_utc": args.timestamp,
        "policy_note": "Coverage and review-pack artifact only. It does not label rows automatically, mutate source rows, or approve a cleaning overlay.",
        "inputs": {
            "reports_dir": str(reports_dir),
            "policy_sample": str(policy_sample),
            "annotation_files": [str(path) for path in gate_audit.annotation_files(reports_dir)],
        },
        "global_prevalence_rows": len(rows),
        "reviewed_unique_rows": len(reviewed_rows),
        "unreviewed_rows": len(unreviewed_rows),
        "reviewed_actionable_rows": reviewed_actionable,
        "reviewed_serious_rows": reviewed_serious,
        "reviewed_actionable_rate_percent": rate * 100.0,
        "reviewed_actionable_wilson_95_ci_percent": wilson(reviewed_actionable, n_reviewed),
        "prevalence_caveat": "Reviewed coverage is a mixed overlap of prior pilot, action-precision, quarantine, and false-negative reviews; it is useful coverage evidence but not yet the final unbiased prevalence estimate for all 1,067 global-prevalence rows.",
        "risk_quantile_thresholds": thresholds,
        "coverage_by": {
            "quality_bin": coverage_table(rows, reviewed_doc_ids, lambda row: row.get("_quality_bin")),
            "risk_band": coverage_table(rows, reviewed_doc_ids, lambda row: row.get("_risk_band")),
            "length_bucket": coverage_table(rows, reviewed_doc_ids, lambda row: row.get("_length_bucket")),
            "candidate_action": coverage_table(rows, reviewed_doc_ids, lambda row: row.get("candidate_action") or "unknown"),
            "source_shard": coverage_table(rows, reviewed_doc_ids, lambda row: row.get("_source_shard")),
        },
        "next_pack_rows": len(pack_rows),
        "next_pack_selection": "deterministic proportional stratified sample over unreviewed global_prevalence rows by quality_bin x risk_band",
        "next_pack_by_quality_bin": counter_dict(pack_rows, lambda row: row.get("quality_bin")),
        "next_pack_by_risk_band": counter_dict(pack_rows, lambda row: row.get("risk_band")),
        "next_pack_by_length_bucket": counter_dict(pack_rows, lambda row: row.get("length_bucket")),
        "next_pack_by_candidate_action": counter_dict(pack_rows, lambda row: row.get("candidate_action")),
        "next_pack_by_source_shard": counter_dict(pack_rows, lambda row: row.get("source_shard")),
        "outputs": {},
    }
    return summary, pack_rows, template_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--policy-sample", default=str(DEFAULT_POLICY_SAMPLE))
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--pack-size", type=int, default=200)
    parser.add_argument("--full-text-chars", type=int, default=0, help="0 means include full text when doc_text_path is readable")
    parser.add_argument("--markdown-chars", type=int, default=3000)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    if not output_prefix.name.endswith(args.timestamp):
        output_prefix = Path(str(output_prefix) + "_" + args.timestamp)

    summary, pack_rows, template_rows = make_summary_and_pack(args)
    json_path = Path(str(output_prefix) + "_summary.json")
    pack_path = Path(str(output_prefix) + ".jsonl")
    template_path = Path(str(output_prefix) + "_annotation_template.jsonl")
    md_path = Path(str(output_prefix) + ".md")
    summary["outputs"] = {
        "summary_json": str(json_path),
        "review_pack_jsonl": str(pack_path),
        "annotation_template_jsonl": str(template_path),
        "markdown": str(md_path),
    }
    write_json(json_path, summary)
    write_jsonl(pack_path, pack_rows)
    write_jsonl(template_path, template_rows)
    write_markdown(md_path, summary, pack_rows, int(args.markdown_chars))
    print(json.dumps({"summary": str(json_path), "review_pack": str(pack_path), "markdown": str(md_path), "unreviewed_rows": summary["unreviewed_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

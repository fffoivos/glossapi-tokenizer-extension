#!/usr/bin/env python3
"""Leak-free LLM-silver comparison report for structural line predictions."""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import GoldDocument, read_gold, sha256_file, validate_silver

PREDICTIONS_SCHEMA = "academic-structure-predictions-v1"
CLASSES = ("O", "BIB", "TOC")


def read_predictions(path: str | Path, documents: Sequence[GoldDocument]) -> dict[str, list[str]]:
    expected = {document.document_id: document for document in documents}
    predictions: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != PREDICTIONS_SCHEMA:
                raise ValueError(f"prediction row {row_number}: unsupported schema")
            document_id = row.get("document_id")
            if document_id not in expected or document_id in predictions:
                raise ValueError(f"prediction row {row_number}: unknown/duplicate document_id")
            document = expected[document_id]
            lines = row.get("lines", [])
            if len(lines) != len(document.lines):
                raise ValueError(f"prediction row {row_number}: incomplete line coverage")
            guesses: list[str] = []
            for gold, predicted in zip(document.lines, lines):
                if predicted.get("line_id") != gold.line_id or predicted.get("abs_idx") != gold.abs_idx:
                    raise ValueError(f"prediction row {row_number}: line identity/order mismatch")
                guess = predicted.get("prediction")
                if guess not in CLASSES:
                    raise ValueError(f"prediction row {row_number}: invalid prediction {guess!r}")
                guesses.append(guess)
            predictions[document_id] = guesses
    missing = set(expected) - set(predictions)
    if missing:
        raise ValueError(f"predictions omit {len(missing)} documents")
    return predictions


def _safe_ratio(numerator: float, denominator: float, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _runs(
    labels: Sequence[str], target: str, known: Sequence[bool], coordinates: Sequence[int]
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, (label, is_known) in enumerate(zip(labels, known)):
        if is_known and label == target:
            if start is None:
                start = index
        elif start is not None:
            spans.append((coordinates[start], coordinates[index - 1]))
            start = None
    if start is not None:
        spans.append((coordinates[start], coordinates[-1]))
    return spans


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union


def _empty_counts() -> collections.Counter[str]:
    return collections.Counter()


def document_counts(
    document: GoldDocument,
    guesses: Sequence[str],
    *,
    maximum_false_fraction: float = 0.01,
    maximum_contiguous_false_tokens: int = 256,
) -> collections.Counter[str]:
    counts = _empty_counts()
    known = [line.label != "UNKNOWN" for line in document.lines]
    counts["represented_lines"] = len(document.lines)
    counts["present_lines"] = document.n_present_lines
    false_run_tokens = 0
    maximum_false_run_tokens = 0
    for line, guess, is_known in zip(document.lines, guesses, known):
        if not is_known:
            counts["unknown_lines"] += 1
            false_run_tokens = 0
            continue
        weight = line.token_count
        counts["known_lines"] += 1
        counts["known_tokens"] += weight
        gold_action = line.label != "O"
        predicted_action = guess != "O"
        counts["line_action_tp"] += int(gold_action and predicted_action)
        counts["line_action_fp"] += int(not gold_action and predicted_action)
        counts["line_action_fn"] += int(gold_action and not predicted_action)
        counts["token_action_tp"] += weight * int(gold_action and predicted_action)
        counts["token_action_fp"] += weight * int(not gold_action and predicted_action)
        counts["token_action_fn"] += weight * int(gold_action and not predicted_action)
        if line.label == "O":
            if line.is_running_prose:
                counts["prose_tokens"] += weight
                counts["prose_tokens_retained"] += weight * int(not predicted_action)
                counts["prose_false_deletion_tokens"] += weight * int(predicted_action)
                false_run_tokens = false_run_tokens + weight if predicted_action else 0
            else:
                false_run_tokens = 0
            maximum_false_run_tokens = max(maximum_false_run_tokens, false_run_tokens)
        else:
            false_run_tokens = 0
        for target in ("BIB", "TOC"):
            counts[f"{target.lower()}_gold_lines"] += int(line.label == target)
            counts[f"{target.lower()}_tp_lines"] += int(line.label == target and guess == target)
            counts[f"{target.lower()}_gold_tokens"] += weight * int(line.label == target)
            counts[f"{target.lower()}_tp_tokens"] += weight * int(line.label == target and guess == target)
    false_fraction = _safe_ratio(counts["prose_false_deletion_tokens"], counts["prose_tokens"])
    counts["catastrophic_fraction"] = int(false_fraction > maximum_false_fraction)
    counts["catastrophic_contiguous"] = int(
        maximum_false_run_tokens > maximum_contiguous_false_tokens
    )
    counts["catastrophic_document"] = int(
        counts["catastrophic_fraction"] or counts["catastrophic_contiguous"]
    )
    counts["documents_with_false_deletion"] = int(counts["token_action_fp"] > 0)
    counts["documents_with_any_gold_action"] = int(
        counts["token_action_tp"] + counts["token_action_fn"] > 0
    )
    counts["documents_with_any_predicted_action"] = int(
        counts["token_action_tp"] + counts["token_action_fp"] > 0
    )
    counts["documents_action_detected_tp"] = int(
        counts["documents_with_any_gold_action"] and counts["documents_with_any_predicted_action"]
    )
    counts["maximum_contiguous_false_tokens"] = maximum_false_run_tokens

    for target in ("BIB", "TOC"):
        coordinates = [line.abs_idx for line in document.lines]
        gold_spans = _runs([line.label for line in document.lines], target, known, coordinates)
        predicted_spans = _runs(guesses, target, known, coordinates)
        counts[f"{target.lower()}_gold_spans"] += len(gold_spans)
        counts[f"{target.lower()}_pred_spans"] += len(predicted_spans)
        counts[f"{target.lower()}_exact_tp_spans"] += sum(span in gold_spans for span in predicted_spans)
        counts[f"{target.lower()}_iou50_tp_pred"] += sum(
            any(_iou(predicted, gold) >= 0.5 for gold in gold_spans) for predicted in predicted_spans
        )
        counts[f"{target.lower()}_iou50_tp_gold"] += sum(
            any(_iou(gold, predicted) >= 0.5 for predicted in predicted_spans) for gold in gold_spans
        )
    return counts


def _sum_counts(rows: Sequence[Mapping[str, float]]) -> collections.Counter[str]:
    total = _empty_counts()
    for row in rows:
        for key, value in row.items():
            if key != "maximum_contiguous_false_tokens":
                total[key] += value
        total["maximum_contiguous_false_tokens"] = max(
            total["maximum_contiguous_false_tokens"], row["maximum_contiguous_false_tokens"]
        )
    return total


def metrics_from_counts(counts: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "line": {
            "action_precision": _safe_ratio(counts["line_action_tp"], counts["line_action_tp"] + counts["line_action_fp"], 1.0),
            "action_recall": _safe_ratio(counts["line_action_tp"], counts["line_action_tp"] + counts["line_action_fn"]),
            "bib_recall": _safe_ratio(counts["bib_tp_lines"], counts["bib_gold_lines"]),
            "toc_recall": _safe_ratio(counts["toc_tp_lines"], counts["toc_gold_lines"]),
        },
        "token": {
            "action_precision": _safe_ratio(counts["token_action_tp"], counts["token_action_tp"] + counts["token_action_fp"], 1.0),
            "action_recall": _safe_ratio(counts["token_action_tp"], counts["token_action_tp"] + counts["token_action_fn"]),
            "bib_recall": _safe_ratio(counts["bib_tp_tokens"], counts["bib_gold_tokens"]),
            "toc_recall": _safe_ratio(counts["toc_tp_tokens"], counts["toc_gold_tokens"]),
            "prose_contamination": _safe_ratio(counts["prose_false_deletion_tokens"], counts["prose_tokens"]),
            "true_main_text_retention": _safe_ratio(counts["prose_tokens_retained"], counts["prose_tokens"], 1.0),
        },
        "span": {},
        "document": {
            "count": int(counts["document_count"]),
            "false_deletion_free_fraction": 1.0 - _safe_ratio(
                counts["documents_with_false_deletion"], counts["document_count"]
            ),
            "structure_detection_precision": _safe_ratio(
                counts["documents_action_detected_tp"],
                counts["documents_with_any_predicted_action"],
                1.0,
            ),
            "structure_detection_recall": _safe_ratio(
                counts["documents_action_detected_tp"],
                counts["documents_with_any_gold_action"],
            ),
            "catastrophic_prose_deletions": int(counts["catastrophic_document"]),
            "maximum_contiguous_false_deletion_tokens": int(counts["maximum_contiguous_false_tokens"]),
        },
        "coverage": {
            "represented_present_line_fraction": _safe_ratio(counts["represented_lines"], counts["present_lines"]),
            "known_label_present_line_fraction": _safe_ratio(counts["known_lines"], counts["present_lines"]),
            "unknown_lines": int(counts["unknown_lines"]),
        },
    }
    for target in ("bib", "toc"):
        result["span"][target] = {
            "exact_precision": _safe_ratio(counts[f"{target}_exact_tp_spans"], counts[f"{target}_pred_spans"], 1.0),
            "exact_recall": _safe_ratio(counts[f"{target}_exact_tp_spans"], counts[f"{target}_gold_spans"]),
            "iou50_precision": _safe_ratio(counts[f"{target}_iou50_tp_pred"], counts[f"{target}_pred_spans"], 1.0),
            "iou50_recall": _safe_ratio(counts[f"{target}_iou50_tp_gold"], counts[f"{target}_gold_spans"]),
        }
    return result


def evaluate(
    documents: Sequence[GoldDocument],
    predictions: Mapping[str, Sequence[str]],
    *,
    split: str = "test",
    maximum_false_fraction: float = 0.01,
    maximum_contiguous_false_tokens: int = 256,
) -> tuple[dict[str, Any], dict[str, collections.Counter[str]]]:
    selected = [document for document in documents if document.split == split]
    per_document: dict[str, collections.Counter[str]] = {}
    for document in selected:
        counts = document_counts(
            document,
            predictions[document.document_id],
            maximum_false_fraction=maximum_false_fraction,
            maximum_contiguous_false_tokens=maximum_contiguous_false_tokens,
        )
        counts["document_count"] = 1
        per_document[document.document_id] = counts
    total = _sum_counts(list(per_document.values()))
    overall = metrics_from_counts(total)
    overall["by_source"] = {}
    for source in sorted({document.source for document in selected}):
        ids = [document.document_id for document in selected if document.source == source]
        overall["by_source"][source] = metrics_from_counts(
            _sum_counts([per_document[document_id] for document_id in ids])
        )
    return overall, per_document


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def work_bootstrap(
    documents: Sequence[GoldDocument],
    candidate: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    *, replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    test_documents = [document for document in documents if document.split == "test"]
    ids_by_work: dict[str, list[str]] = collections.defaultdict(list)
    for document in test_documents:
        ids_by_work[document.work_id].append(document.document_id)
    works = sorted(ids_by_work)
    rng = random.Random(seed)
    values: dict[str, list[float]] = collections.defaultdict(list)
    for _ in range(replicates):
        sampled = [rng.choice(works) for _ in works]
        candidate_rows = [candidate[document_id] for work in sampled for document_id in ids_by_work[work]]
        baseline_rows = [baseline[document_id] for work in sampled for document_id in ids_by_work[work]]
        cm = metrics_from_counts(_sum_counts(candidate_rows))["token"]
        bm = metrics_from_counts(_sum_counts(baseline_rows))["token"]
        values["action_precision"].append(cm["action_precision"])
        values["prose_contamination"].append(cm["prose_contamination"])
        values["bib_recall_gain"].append(cm["bib_recall"] - bm["bib_recall"])
        values["toc_recall_gain"].append(cm["toc_recall"] - bm["toc_recall"])
    return {
        name: [_percentile(samples, 0.025), _percentile(samples, 0.975)]
        for name, samples in values.items()
    }


def promotion_report(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], confidence: Mapping[str, Sequence[float]],
    gates: Mapping[str, Any],
    *,
    artifact_bytes: int | None = None,
    runtime_parity: Mapping[str, Any] | None = None,
    candidate_benchmark: Mapping[str, Any] | None = None,
    baseline_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    token = candidate["token"]
    baseline_token = baseline["token"]
    checks = {
        "action_token_precision": token["action_precision"] >= gates["minimum_deletion_token_precision"],
        "action_token_precision_ci95_lower": confidence["action_precision"][0] >= gates["minimum_deletion_token_precision_ci95_lower"],
        "each_source_action_token_precision": all(
            row["token"]["action_precision"] >= gates["minimum_per_source_deletion_token_precision"]
            for row in candidate["by_source"].values()
        ),
        "prose_contamination": token["prose_contamination"] <= gates["maximum_prose_token_contamination"],
        "prose_contamination_ci95_upper": confidence["prose_contamination"][1] <= gates["maximum_prose_token_contamination_ci95_upper"],
        "zero_catastrophic_prose_deletions": candidate["document"]["catastrophic_prose_deletions"] == 0,
        "bib_recall_gain_at_matched_safety": token["bib_recall"] - baseline_token["bib_recall"] >= gates["minimum_bib_recall_gain"],
        "toc_recall_gain_at_matched_safety": token["toc_recall"] - baseline_token["toc_recall"] >= gates["minimum_toc_recall_gain"],
        "per_head_recall_nonregression": all(
            token[f"{head}_recall"] - baseline_token[f"{head}_recall"]
            >= -gates["maximum_per_head_recall_regression"]
            for head in ("bib", "toc")
        ),
        "per_source_recall_nonregression": all(
            candidate["by_source"][source]["token"][f"{head}_recall"]
            - baseline["by_source"][source]["token"][f"{head}_recall"]
            >= -gates["maximum_per_source_recall_regression"]
            for source in candidate["by_source"] for head in ("bib", "toc")
        ),
        "pergamos_recall_nonregression": (
            gates.get("pergamos_allows_recall_regression", False)
            or "Apothetirio_Pergamos" not in candidate["by_source"]
            or all(
                candidate["by_source"]["Apothetirio_Pergamos"]["token"][f"{head}_recall"]
                >= baseline["by_source"]["Apothetirio_Pergamos"]["token"][f"{head}_recall"]
                for head in ("bib", "toc")
            )
        ),
    }
    operational = {
        "artifact_size": artifact_bytes is not None and artifact_bytes <= gates["maximum_artifact_bytes"],
        "cpu_runtime_parity": (
            not gates.get("require_python_runtime_parity", True)
            or runtime_parity is not None and runtime_parity.get("status") == "pass"
        ),
        "peak_rss": (
            candidate_benchmark is not None
            and candidate_benchmark.get("peak_rss_bytes", candidate_benchmark.get("peak_rss_platform_units", float("inf")))
            <= gates["maximum_peak_rss_bytes"]
        ),
        "cpu_cost_relative_to_c0": (
            candidate_benchmark is not None and baseline_benchmark is not None
            and candidate_benchmark["best_seconds"]
            <= baseline_benchmark["best_seconds"] * gates["maximum_cpu_hours_relative_to_c0"]
        ),
        "one_cpu_node_wall_time": (
            candidate_benchmark is not None
            and candidate_benchmark["best_seconds"] / 3600.0
            <= gates["maximum_allowlist_wall_hours_one_cpu_node"]
        ),
    }
    statistical_status = "pass" if all(checks.values()) else "fail"
    if statistical_status == "fail":
        status = "fail"
    elif all(operational.values()):
        status = "pass"
    else:
        status = "blocked"
    return {
        "status": status,
        "statistical_status": statistical_status,
        "checks": checks,
        "operational_checks": operational,
        "recall_gain": {
            "bib": token["bib_recall"] - baseline_token["bib_recall"],
            "toc": token["toc_recall"] - baseline_token["toc_recall"],
        },
        "note": "The locked test is comparison-only; no threshold, feature, or architecture selection is permitted from this report.",
    }


def _mark_silver_safety_unavailable(metrics: dict[str, Any]) -> None:
    """Remove misleading safety values that require independent prose gold."""
    for row in [metrics, *metrics.get("by_source", {}).values()]:
        row["token"]["prose_contamination"] = None
        row["token"]["true_main_text_retention"] = None
        row["document"]["catastrophic_prose_deletions"] = None
        row["document"]["maximum_contiguous_false_deletion_tokens"] = None
    metrics["metric_availability"] = {
        "silver_action_and_recall_metrics": True,
        "independent_running_prose_safety_metrics": False,
        "reason": "LLM silver has no independent is_running_prose judgments",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    documents = read_gold(args.silver)
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    contract_receipt = validate_silver(
        documents, config["silver_contract"], split_manifest=split_manifest
    )
    candidate_predictions = read_predictions(args.candidate, documents)
    baseline_predictions = read_predictions(args.baseline, documents)
    gates = config["deployment_gates"]
    metric_options = {
        "maximum_false_fraction": float(gates["maximum_false_deletion_fraction_per_document"]),
        "maximum_contiguous_false_tokens": int(gates["maximum_contiguous_false_deletion_tokens"]),
    }
    candidate, candidate_docs = evaluate(documents, candidate_predictions, **metric_options)
    baseline, baseline_docs = evaluate(documents, baseline_predictions, **metric_options)
    confidence = work_bootstrap(
        documents, candidate_docs, baseline_docs,
        replicates=int(gates["bootstrap_replicates"]), seed=int(gates["bootstrap_seed"]),
    )
    _mark_silver_safety_unavailable(candidate)
    _mark_silver_safety_unavailable(baseline)
    confidence["prose_contamination"] = None
    promotion = {
        "status": "no_op",
        "production_eligible": False,
        "reason": (
            "LLM_silver authorizes research comparison only; production requires the separate "
            "receipt-bound 100-case high-risk false-deletion audit and all deployment gates"
        ),
        "required_manual_review_count": int(
            config["deployment_safety_audit"]["required_review_count"]
        ),
        "production_fallback": "no_op",
    }
    report = {
        "schema_version": "academic-structure-evaluation-v1",
        "receipts": {
            "silver_sha256": sha256_file(args.silver),
            "candidate_sha256": sha256_file(args.candidate),
            "baseline_sha256": sha256_file(args.baseline),
            "config_sha256": sha256_file(args.config),
            "split_manifest_sha256": sha256_file(args.split_manifest),
        },
        "contract": contract_receipt,
        "candidate": candidate,
        "baseline": baseline,
        "work_clustered_bootstrap_ci95": confidence,
        "evidence_tier": "LLM_silver",
        "promotion": promotion,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["promotion"], ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

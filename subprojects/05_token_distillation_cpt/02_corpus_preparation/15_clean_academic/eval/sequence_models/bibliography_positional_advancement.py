#!/usr/bin/env python3
"""Gate positional entry arms with paired work bootstrap evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import load_table
from .bibliography_positional_models import ALL_ARMS, load_targets
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-positional-advancement-v1"


def threshold_at_precision(
    targets: np.ndarray, probability: np.ndarray, precision_target: float,
) -> dict[str, float | int]:
    if targets.shape != probability.shape or targets.ndim != 1 or not len(targets):
        raise ValueError("threshold selection requires equal non-empty vectors")
    if not 0 < precision_target <= 1:
        raise ValueError("precision target must be in (0, 1]")
    truth = targets == 1
    positive_count = int(np.count_nonzero(truth))
    if not positive_count:
        raise ValueError("threshold selection requires positives")
    order = np.argsort(-probability, kind="stable")
    sorted_probability, sorted_truth = probability[order], truth[order]
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    )
    tp = cumulative_tp[group_ends]
    predicted = group_ends + 1
    precision = tp / predicted
    recall = tp / positive_count
    eligible = np.flatnonzero(precision >= precision_target)
    if not len(eligible):
        threshold = float(np.nextafter(np.max(probability), np.inf))
        return {
            "threshold": threshold, "precision": 1.0, "recall": 0.0,
            "tp": 0, "fp": 0, "fn": positive_count,
        }
    best = max(
        eligible,
        key=lambda index: (recall[index], precision[index], sorted_probability[group_ends[index]]),
    )
    selected_tp, selected_predicted = int(tp[best]), int(predicted[best])
    return {
        "threshold": float(sorted_probability[group_ends[best]]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "tp": selected_tp,
        "fp": selected_predicted - selected_tp,
        "fn": positive_count - selected_tp,
    }


def metrics_at_threshold(
    targets: np.ndarray, probability: np.ndarray, threshold: float,
) -> dict[str, float | int]:
    truth, predicted = targets == 1, probability >= threshold
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    return {
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "line_count": len(targets),
    }


def paired_work_recall_interval(
    targets: np.ndarray, baseline_probability: np.ndarray, candidate_probability: np.ndarray,
    work_codes: np.ndarray, *, baseline_threshold: float, candidate_threshold: float,
    replicates: int, seed: int,
) -> dict[str, float | int]:
    if replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    if not (
        targets.shape == baseline_probability.shape == candidate_probability.shape == work_codes.shape
    ):
        raise ValueError("paired bootstrap vectors are misaligned")
    n_works = int(work_codes.max()) + 1
    truth = targets == 1
    positives = np.bincount(work_codes, weights=truth, minlength=n_works)
    baseline_tp = np.bincount(
        work_codes, weights=truth & (baseline_probability >= baseline_threshold), minlength=n_works
    )
    candidate_tp = np.bincount(
        work_codes, weights=truth & (candidate_probability >= candidate_threshold), minlength=n_works
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        weights = np.bincount(rng.integers(0, n_works, n_works), minlength=n_works)
        denominator = float(np.dot(weights, positives))
        if denominator == 0:
            deltas[replicate] = np.nan
            continue
        deltas[replicate] = (
            float(np.dot(weights, candidate_tp)) - float(np.dot(weights, baseline_tp))
        ) / denominator
    valid = deltas[np.isfinite(deltas)]
    if len(valid) < int(0.95 * replicates):
        raise ValueError("too many paired bootstrap replicates lack positive lines")
    return {
        "replicates": replicates,
        "valid_replicates": len(valid),
        "mean_delta": float(np.mean(valid)),
        "lower_95": float(np.quantile(valid, 0.025)),
        "upper_95": float(np.quantile(valid, 0.975)),
    }


def _source_masks(table: Any, labelled: np.ndarray) -> dict[str, np.ndarray]:
    source_by_document = np.asarray([str(row["source"]) for row in table.documents], dtype=object)
    line_sources = source_by_document[np.asarray(table.document_indices)]
    return {
        source: labelled & (line_sources == source)
        for source in sorted(set(source_by_document.tolist()))
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.base_table_dir, expected_split="train")
    base_sha = sha256_file(table.root / "manifest.json")
    targets, target_manifest = load_targets(
        args.role_target_dir, base_sha, len(table.targets)
    )
    labelled = targets != -1
    model_root = Path(args.model_dir).resolve()
    model_report = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
    if (
        model_report.get("schema_version") != "bibliography-positional-entry-ladder-v1"
        or model_report.get("validation_opened") is not False
    ):
        raise ValueError("positional ladder is not a sealed train-OOF report")
    probabilities = {
        arm: np.load(model_root / f"{arm}.oof_probability.npy", mmap_mode="r", allow_pickle=False)
        for arm in ALL_ARMS
    }
    if any(value.shape != targets.shape or np.any(~np.isfinite(value[labelled])) for value in probabilities.values()):
        raise ValueError("OOF probability coverage is incomplete")
    thresholds = {
        arm: threshold_at_precision(
            np.asarray(targets[labelled]), np.asarray(probability[labelled]), args.precision_target
        )
        for arm, probability in probabilities.items()
    }
    source_masks = _source_masks(table, labelled)
    by_source = {
        arm: {
            source: metrics_at_threshold(
                np.asarray(targets[mask]), np.asarray(probability[mask]),
                float(thresholds[arm]["threshold"]),
            )
            for source, mask in source_masks.items() if np.any(mask)
        }
        for arm, probability in probabilities.items()
    }
    work_by_document = [str(row["work_id"]) for row in table.documents]
    unique_works = {work: index for index, work in enumerate(sorted(set(work_by_document)))}
    work_codes = np.asarray(
        [unique_works[work_by_document[int(document)]] for document in table.document_indices[labelled]],
        dtype=np.int32,
    )
    comparisons: dict[str, Any] = {}
    for arm in ("P0D", "P1", "P1G"):
        interval = paired_work_recall_interval(
            np.asarray(targets[labelled]), np.asarray(probabilities["P0"][labelled]),
            np.asarray(probabilities[arm][labelled]), work_codes,
            baseline_threshold=float(thresholds["P0"]["threshold"]),
            candidate_threshold=float(thresholds[arm]["threshold"]),
            replicates=args.bootstrap_replicates, seed=args.seed + ALL_ARMS.index(arm),
        )
        source_deltas = {
            source: by_source[arm][source]["recall"] - by_source["P0"][source]["recall"]
            for source in by_source["P0"]
        }
        positional_candidate = arm in {"P1", "P1G"}
        passed = (
            positional_candidate
            and thresholds[arm]["precision"] >= args.precision_target
            and interval["lower_95"] > 0
            and min(source_deltas.values()) >= -args.maximum_source_recall_loss
        )
        comparisons[arm] = {
            "paired_recall_delta": interval,
            "source_recall_deltas": source_deltas,
            "advancement_passed": passed,
            "diagnostic_only": not positional_candidate,
        }
    passing = [arm for arm in ("P1", "P1G") if comparisons[arm]["advancement_passed"]]
    selected = max(
        passing,
        key=lambda arm: (
            thresholds[arm]["recall"],
            comparisons[arm]["paired_recall_delta"]["lower_95"],
            -(("P1", "P1G").index(arm)),
        ),
    ) if passing else "P0"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_advancement_validation_unopened",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "precision_target": args.precision_target,
        "maximum_source_recall_loss": args.maximum_source_recall_loss,
        "work_count": len(unique_works),
        "labelled_line_count": int(np.count_nonzero(labelled)),
        "thresholds": thresholds,
        "by_source": by_source,
        "comparisons_to_p0": comparisons,
        "selected_summary_arm": selected,
        "next_stage": "run_predeclared_sparse_p2_controls",
        "validation_opened": False,
        "inputs": {
            "base_table_manifest_sha256": base_sha,
            "role_target_manifest_sha256": sha256_file(
                Path(args.role_target_dir).resolve() / "manifest.json"
            ),
            "overlay_sha256": target_manifest["overlay"]["sha256"],
            "model_report_sha256": sha256_file(model_root / "report.json"),
            "probability_sha256": {
                arm: sha256_file(model_root / f"{arm}.oof_probability.npy") for arm in ALL_ARMS
            },
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--role-target-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision-target", type=float, default=0.99)
    parser.add_argument("--maximum-source-recall-loss", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

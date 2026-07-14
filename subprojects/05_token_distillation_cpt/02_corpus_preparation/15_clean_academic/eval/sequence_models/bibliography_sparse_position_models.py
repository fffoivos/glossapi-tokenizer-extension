#!/usr/bin/env python3
"""Fit 8/16/32-bin sparse P2 entry arms and gate them against P0D."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_positional_advancement import (
    metrics_at_threshold, paired_work_recall_interval, threshold_at_precision,
)
from .bibliography_positional_features import FEATURE_NAMES, NONMATCH_CATEGORIES
from .bibliography_positional_models import load_targets
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-sparse-position-ladder-v1"


def _libraries() -> tuple[Any, Any, Any]:
    import scipy.sparse as sparse
    import sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import average_precision_score
    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}")
    return sparse, SGDClassifier, average_precision_score


def pool_bins(matrix: Any, bins: int) -> Any:
    if bins not in (8, 16, 32):
        raise ValueError("bins must be 8, 16, or 32")
    if bins == 32:
        return matrix
    sparse = _libraries()[0]
    coo = matrix.tocoo(copy=False)
    factor = 32 // bins
    channels = coo.col // 32
    columns = channels * bins + (coo.col % 32) // factor
    pooled = sparse.coo_matrix(
        (coo.data / factor, (coo.row, columns)),
        shape=(matrix.shape[0], (len(FEATURE_NAMES) + len(NONMATCH_CATEGORIES)) * bins),
        dtype=np.float32,
    ).tocsr()
    pooled.sum_duplicates()
    pooled.eliminate_zeros()
    return pooled


def _fit_arm(
    x: Any, targets: np.ndarray, folds: np.ndarray, *, alpha_grid: Sequence[float],
    seed: int, parallel_folds: int, model_dir: Path, arm: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _, classifier, average_precision = _libraries()
    n_folds = int(folds.max()) + 1

    def model(alpha: float, local_seed: int) -> Any:
        return classifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.25,
            alpha=float(alpha), max_iter=30, tol=1.0e-4, shuffle=True,
            random_state=local_seed, average=True, n_jobs=1,
        )

    def outer_task(outer: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        inner = (outer + 1) % n_folds
        inner_fit, inner_holdout = folds != outer, folds == inner
        inner_fit &= folds != inner
        candidates = []
        for offset, alpha in enumerate(alpha_grid):
            fitted = model(alpha, seed + outer * 100 + offset)
            fitted.fit(x[inner_fit], targets[inner_fit])
            probability = fitted.predict_proba(x[inner_holdout])[:, 1]
            candidates.append({
                "alpha": float(alpha),
                "inner_pr_auc": float(average_precision(targets[inner_holdout], probability)),
            })
        selected = max(candidates, key=lambda row: (row["inner_pr_auc"], -row["alpha"]))
        fit, holdout = folds != outer, folds == outer
        fitted = model(float(selected["alpha"]), seed + outer)
        fitted.fit(x[fit], targets[fit])
        probability = fitted.predict_proba(x[holdout])[:, 1]
        with (model_dir / f"{arm}.fold{outer}.pkl").open("xb") as handle:
            pickle.dump(fitted, handle, protocol=5)
        return np.flatnonzero(holdout), probability, {
            "outer_fold": outer, "inner_holdout_fold": inner,
            "selected_alpha": selected["alpha"], "candidates": candidates,
            "iterations": int(fitted.n_iter_), "feature_count": x.shape[1],
        }

    probability = np.full(len(targets), np.nan, dtype=np.float32)
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_folds, parallel_folds)) as executor:
        for rows, values, report in executor.map(outer_task, range(n_folds)):
            probability[rows] = values
            reports.append(report)
    reports.sort(key=lambda row: int(row["outer_fold"]))
    if np.any(~np.isfinite(probability)):
        raise ValueError("P2 OOF coverage is incomplete")
    return probability, reports


def _source_masks(table: Any, labelled_indices: np.ndarray) -> dict[str, np.ndarray]:
    source_by_document = np.asarray([str(row["source"]) for row in table.documents], dtype=object)
    sources = source_by_document[np.asarray(table.document_indices[labelled_indices])]
    return {source: sources == source for source in sorted(set(sources.tolist()))}


def _write_json_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse, _, average_precision = _libraries()
    table = load_table(args.base_table_dir, expected_split="train")
    base_sha = sha256_file(table.root / "manifest.json")
    full_targets, target_manifest = load_targets(args.role_target_dir, base_sha, len(table.targets))
    root = Path(args.sparse_table_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "bibliography-sparse-position-table-v1" or not manifest.get("raster_integral_parity"):
        raise ValueError("sparse table lacks verified raster parity")
    labelled_indices = np.load(root / "labelled_indices.npy", allow_pickle=False)
    targets = np.asarray(full_targets[labelled_indices], dtype=np.int8)
    folds = np.asarray(table.folds[labelled_indices], dtype=np.uint8)
    maps32 = sparse.load_npz(root / "position_map_32.npz").tocsr()
    scalars = sparse.load_npz(root / "count_gap_scalars.npz").tocsr()
    if maps32.shape[0] != len(targets) or scalars.shape != (len(targets), 77):
        raise ValueError("sparse model inputs are misaligned")
    baseline_root = Path(args.baseline_model_dir).resolve()
    baseline_full = np.load(baseline_root / "P0D.oof_probability.npy", mmap_mode="r", allow_pickle=False)
    baseline = np.asarray(baseline_full[labelled_indices])
    baseline_threshold = threshold_at_precision(targets, baseline, args.precision_target)
    work_by_document = [str(row["work_id"]) for row in table.documents]
    unique_works = {work: index for index, work in enumerate(sorted(set(work_by_document)))}
    work_codes = np.asarray([
        unique_works[work_by_document[int(document)]]
        for document in table.document_indices[labelled_indices]
    ], dtype=np.int32)
    source_masks = _source_masks(table, labelled_indices)
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_dir = output / "models"
    model_dir.mkdir()
    results: dict[str, Any] = {}
    for bins in (8, 16, 32):
        arm = f"P2_{bins}"
        maps = pool_bins(maps32, bins)
        x = sparse.hstack((scalars, maps), format="csr", dtype=np.float32)
        probability, fold_reports = _fit_arm(
            x, targets, folds, alpha_grid=args.alpha_grid, seed=args.seed + bins,
            parallel_folds=args.parallel_folds, model_dir=model_dir, arm=arm,
        )
        threshold = threshold_at_precision(targets, probability, args.precision_target)
        interval = paired_work_recall_interval(
            targets, baseline, probability, work_codes,
            baseline_threshold=float(baseline_threshold["threshold"]),
            candidate_threshold=float(threshold["threshold"]),
            replicates=args.bootstrap_replicates, seed=args.seed + bins,
        )
        source_deltas = {}
        for source, mask in source_masks.items():
            source_deltas[source] = (
                metrics_at_threshold(targets[mask], probability[mask], float(threshold["threshold"]))["recall"]
                - metrics_at_threshold(targets[mask], baseline[mask], float(baseline_threshold["threshold"]))["recall"]
            )
        passed = (
            threshold["precision"] >= args.precision_target
            and interval["lower_95"] > 0
            and min(source_deltas.values()) >= -args.maximum_source_recall_loss
        )
        results[arm] = {
            "bins": bins, "folds": fold_reports,
            "pr_auc": float(average_precision(targets, probability)),
            "threshold_at_precision": threshold,
            "paired_recall_delta_vs_p0d": interval,
            "source_recall_deltas_vs_p0d": source_deltas,
            "advancement_passed": passed,
        }
        with (output / f"{arm}.oof_probability.npy").open("xb") as handle:
            np.save(handle, probability, allow_pickle=False)
    passing = [arm for arm, row in results.items() if row["advancement_passed"]]
    selected = max(
        passing,
        key=lambda arm: (results[arm]["threshold_at_precision"]["recall"], -results[arm]["bins"]),
    ) if passing else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "p2_candidate_passed_run_controls" if selected else "p2_stopped_no_gain_over_p0d",
        "code_commit": args.code_commit, "slurm_job_id": args.slurm_job_id,
        "sklearn_version": PINNED_SKLEARN_VERSION,
        "row_count": len(targets), "positive_count": int(np.count_nonzero(targets == 1)),
        "baseline": {"arm": "P0D", "threshold_at_precision": baseline_threshold},
        "arms": results, "selected_p2_arm": selected,
        "next_stage": "run_location_shuffle_controls" if selected else "retain_count_only_p0d_and_skip_p3",
        "validation_opened": False,
        "inputs": {
            "sparse_table_manifest_sha256": sha256_file(root / "manifest.json"),
            "baseline_report_sha256": sha256_file(baseline_root / "report.json"),
            "role_target_manifest_sha256": sha256_file(Path(args.role_target_dir).resolve() / "manifest.json"),
            "overlay_sha256": target_manifest["overlay"]["sha256"],
        },
    }
    _write_json_new(output / "report.json", report)
    receipt = {**report, "outputs": {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.rglob("*")) if path.is_file()
    }}
    _write_json_new(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--role-target-dir", required=True)
    parser.add_argument("--sparse-table-dir", required=True)
    parser.add_argument("--baseline-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=(1e-6, 3e-6, 1e-5, 3e-5))
    parser.add_argument("--precision-target", type=float, default=0.99)
    parser.add_argument("--maximum-source-recall-loss", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

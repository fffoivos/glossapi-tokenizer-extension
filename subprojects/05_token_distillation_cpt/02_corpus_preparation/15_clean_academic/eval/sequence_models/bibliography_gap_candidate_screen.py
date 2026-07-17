#!/usr/bin/env python3
"""Grouped pooled-model screen across gap preparation regimes and sizes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import PINNED_SKLEARN_VERSION
from .bibliography_gap_candidate_table import SCHEMA_VERSION as TABLE_SCHEMA_VERSION
from .bibliography_gap_candidates import REGIME_ORDER
from .bibliography_gap_connect_models import best_safety_threshold, binary_metrics
from .bibliography_gap_sampling import (
    available_negative_group_count,
    fit_weights,
    select_training_rows,
    size_rungs,
)
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-candidate-screen-oof-v1"
ARMS = ("pooled_logistic", "pooled_hist")


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    return {
        "hist": HistGradientBoostingClassifier,
        "logistic": LogisticRegression,
        "average_precision": average_precision_score,
        "scaler": StandardScaler,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


class CandidateTable:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != TABLE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate table schema")
        if self.manifest.get("validation_opened") is not False:
            raise ValueError("candidate screen requires validation-isolated inputs")
        self.features = np.load(self.root / "features.npy", mmap_mode="r", allow_pickle=False)
        self.line_targets = np.load(self.root / "line_targets.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.root / "gap_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(self.root / "targets.npy", mmap_mode="r", allow_pickle=False)
        self.folds = np.load(self.root / "folds.npy", mmap_mode="r", allow_pickle=False)
        self.lengths = np.load(self.root / "gap_lengths.npy", mmap_mode="r", allow_pickle=False)
        self.metadata = tuple(
            json.loads(line)
            for line in (self.root / "gaps.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        n = len(self.targets)
        if not (
            len(self.folds) == len(self.lengths) == len(self.metadata) == n
            and len(self.offsets) == n + 1
            and int(self.offsets[-1]) == len(self.features) == len(self.line_targets)
            and np.all(np.diff(self.offsets.astype(np.int64)) == self.lengths)
        ):
            raise ValueError("candidate table arrays are not aligned")
        for index, row in enumerate(self.metadata):
            if int(row["target_connect"]) != int(self.targets[index]):
                raise ValueError("candidate metadata target mismatch")

    def sequence(self, index: int) -> np.ndarray:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return np.asarray(self.features[start:end], dtype=np.float32)


def pooled_features(table: CandidateTable) -> np.ndarray:
    rows = []
    for index in range(len(table.targets)):
        values = table.sequence(index)
        rows.append(np.concatenate((
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
            np.asarray((math.log1p(len(values)),), dtype=np.float32),
        )))
    result = np.stack(rows).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("pooled candidate features are non-finite")
    return result


def _fit(
    arm: str, features: np.ndarray, targets: np.ndarray, rows: np.ndarray,
    metadata: Sequence[Mapping[str, Any]], *, seed: int, c_value: float = 0.1,
) -> Any:
    tools = _sklearn()
    weights = fit_weights(metadata, targets, rows)
    if arm == "pooled_hist":
        return tools["hist"](
            learning_rate=0.05,
            max_iter=180,
            max_depth=3,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ).fit(features[rows], targets[rows], sample_weight=weights)
    if arm == "pooled_logistic":
        scaler = tools["scaler"]().fit(features[rows])
        model = tools["logistic"](
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        ).fit(scaler.transform(features[rows]), targets[rows], sample_weight=weights)
        return scaler, model
    raise ValueError(arm)


def _predict(model: Any, arm: str, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if arm == "pooled_logistic":
        scaler, estimator = model
        return estimator.predict_proba(scaler.transform(features[rows]))[:, 1]
    return model.predict_proba(features[rows])[:, 1]


def _break_ap(targets: np.ndarray, probability: np.ndarray) -> float:
    return float(_sklearn()["average_precision"](targets == 0, 1.0 - probability))


def _work_bootstrap_se(
    targets: np.ndarray, probability: np.ndarray, works: np.ndarray, *,
    seed: int, replicates: int,
) -> float:
    unique = np.unique(works)
    rng = np.random.default_rng(seed)
    values = []
    by_work = {work: np.flatnonzero(works == work) for work in unique}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_work[work] for work in sampled])
        if len(np.unique(targets[rows])) < 2:
            continue
        values.append(_break_ap(targets[rows], probability[rows]))
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def _source_metrics(
    targets: np.ndarray, probability: np.ndarray, thresholds: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for source in sorted(set(sources.tolist())):
        rows = np.flatnonzero(sources == source)
        if len(np.unique(targets[rows])) < 2:
            continue
        result[str(source)] = binary_metrics(
            targets[rows], probability[rows], thresholds[rows]
        )
    return result


def _run_configuration(
    *, table: CandidateTable, pooled: np.ndarray, genuine_rows: np.ndarray,
    train_rows: np.ndarray, arm: str, regime: str, size_label: str,
    seed: int, maximum_false_connect_rate: float, bootstrap_replicates: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n_folds = int(table.folds.max()) + 1
    probability = np.full(len(genuine_rows), np.nan, dtype=np.float32)
    threshold = np.full(len(genuine_rows), np.nan, dtype=np.float32)
    genuine_fold = table.folds[genuine_rows]
    fold_reports = []
    c_grid = (0.01, 0.1, 1.0)

    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        outer_eval_local = np.flatnonzero(genuine_fold == outer)
        inner_eval = genuine_rows[genuine_fold == inner]
        outer_eval = genuine_rows[outer_eval_local]
        inner_fit = train_rows[
            (table.folds[train_rows] != outer) & (table.folds[train_rows] != inner)
        ]
        outer_fit = train_rows[table.folds[train_rows] != outer]
        for name, rows in (
            ("inner fit", inner_fit),
            ("outer fit", outer_fit),
            ("inner evaluation", inner_eval),
            ("outer evaluation", outer_eval),
        ):
            if {int(value) for value in np.unique(table.targets[rows])} != {0, 1}:
                raise ValueError(f"{regime}/{size_label}/{arm}/fold{outer} {name} lacks a class")

        selected_c = 0.1
        c_reports = []
        if arm == "pooled_logistic":
            best = (-1.0, -1.0)
            for c_value in c_grid:
                candidate = _fit(
                    arm, pooled, table.targets, inner_fit, table.metadata,
                    seed=seed + outer, c_value=c_value,
                )
                local_probability = _predict(candidate, arm, pooled, inner_eval)
                row = {
                    "c": c_value,
                    "break_pr_auc": _break_ap(table.targets[inner_eval], local_probability),
                    "connect_pr_auc": float(_sklearn()["average_precision"](
                        table.targets[inner_eval], local_probability
                    )),
                }
                c_reports.append(row)
                if (row["break_pr_auc"], row["connect_pr_auc"]) > best:
                    best = (row["break_pr_auc"], row["connect_pr_auc"])
                    selected_c = c_value

        inner_model = _fit(
            arm, pooled, table.targets, inner_fit, table.metadata,
            seed=seed + 100 + outer, c_value=selected_c,
        )
        inner_probability = _predict(inner_model, arm, pooled, inner_eval)
        selection = best_safety_threshold(
            table.targets[inner_eval], inner_probability,
            max_false_connect_rate=maximum_false_connect_rate,
        )
        model = _fit(
            arm, pooled, table.targets, outer_fit, table.metadata,
            seed=seed + 200 + outer, c_value=selected_c,
        )
        probability[outer_eval_local] = _predict(model, arm, pooled, outer_eval)
        threshold[outer_eval_local] = selection["threshold"]
        fold_reports.append({
            "outer_fold": outer,
            "inner_fold": inner,
            "inner_fit_count": len(inner_fit),
            "outer_fit_count": len(outer_fit),
            "selected_c": selected_c if arm == "pooled_logistic" else None,
            "candidates": c_reports,
            "threshold_selection": selection,
        })

    if not np.isfinite(probability).all() or not np.isfinite(threshold).all():
        raise RuntimeError("pooled OOF predictions are incomplete")
    targets = table.targets[genuine_rows]
    works = np.asarray([table.metadata[index]["work_id"] for index in genuine_rows])
    sources = np.asarray([table.metadata[index]["source"] for index in genuine_rows])
    metrics = binary_metrics(targets, probability, threshold)
    metrics["break_pr_auc_bootstrap_se"] = _work_bootstrap_se(
        targets, probability, works,
        seed=seed + 999,
        replicates=bootstrap_replicates,
    )
    report = {
        "arm": arm,
        "regime": regime,
        "size": size_label,
        "train_row_count": len(train_rows),
        "train_boundary_group_count": len({
            table.metadata[index]["boundary_group_id"] for index in train_rows
        }),
        "train_negative_boundary_group_count": len({
            table.metadata[index]["boundary_group_id"]
            for index in train_rows if table.targets[index] == 0
        }),
        "train_work_count": len({table.metadata[index]["work_id"] for index in train_rows}),
        "oof_metrics": metrics,
        "by_source": _source_metrics(targets, probability, threshold, sources),
        "folds": fold_reports,
    }
    return report, probability, threshold


def _selection_key(report: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = report["oof_metrics"]
    return (
        int(metrics["false_connect_count"]),
        int(report["train_negative_boundary_group_count"]),
        -float(metrics["connect_recall"]),
        -float(metrics["break_pr_auc"]),
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    table = CandidateTable(Path(args.table_dir))
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    pooled = pooled_features(table)
    genuine_rows = np.asarray([
        index for index, row in enumerate(table.metadata)
        if bool(row["genuine_deployment_candidate"])
    ], dtype=np.int64)
    if {int(value) for value in np.unique(table.targets[genuine_rows])} != {0, 1}:
        raise ValueError("fixed genuine evaluation rows require both classes")

    reports = []
    predictions: dict[str, np.ndarray] = {}
    thresholds: dict[str, np.ndarray] = {}
    for regime in REGIME_ORDER:
        available = available_negative_group_count(table.metadata, table.targets, regime)
        for limit in size_rungs(available):
            size_label = "all" if limit is None else str(limit)
            train_rows = select_training_rows(
                table.metadata,
                table.targets,
                regime=regime,
                negative_group_limit=limit,
                seed=args.seed,
            )
            for arm in ARMS:
                report, probability, threshold = _run_configuration(
                    table=table,
                    pooled=pooled,
                    genuine_rows=genuine_rows,
                    train_rows=train_rows,
                    arm=arm,
                    regime=regime,
                    size_label=size_label,
                    seed=args.seed,
                    maximum_false_connect_rate=args.maximum_false_connect_rate,
                    bootstrap_replicates=args.bootstrap_replicates,
                )
                reports.append(report)
                key = f"{regime}__{size_label}__{arm}"
                predictions[key] = probability
                thresholds[key] = threshold

    best_break = max(float(row["oof_metrics"]["break_pr_auc"]) for row in reports)
    best_report = max(reports, key=lambda row: float(row["oof_metrics"]["break_pr_auc"]))
    one_se_floor = best_break - float(best_report["oof_metrics"]["break_pr_auc_bootstrap_se"])
    baseline = next(
        row for row in reports
        if row["regime"] == "deployment_real" and row["size"] == "all" and row["arm"] == "pooled_hist"
    )

    def source_safe(row: Mapping[str, Any]) -> bool:
        return all(
            int(row["by_source"].get(source, {}).get("false_connect_count", 0))
            <= int(metrics["false_connect_count"])
            for source, metrics in baseline["by_source"].items()
        )

    eligible = [
        row for row in reports
        if float(row["oof_metrics"]["break_pr_auc"]) >= one_se_floor and source_safe(row)
    ]
    if not eligible:
        eligible = [row for row in reports if float(row["oof_metrics"]["break_pr_auc"]) >= one_se_floor]
    selected_pooled = min(eligible, key=_selection_key)
    best_by_configuration: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in eligible:
        key = (str(row["regime"]), str(row["size"]))
        previous = best_by_configuration.get(key)
        if previous is None or _selection_key(row) < _selection_key(previous):
            best_by_configuration[key] = row
    selected_config = (str(selected_pooled["regime"]), str(selected_pooled["size"]))
    sequence_candidates = [best_by_configuration[selected_config]]
    sequence_candidates.extend(sorted(
        (
            row for key, row in best_by_configuration.items()
            if key != selected_config and str(row["regime"]) != str(selected_pooled["regime"])
        ),
        key=_selection_key,
    )[:1])
    selected = {
        "selected_pooled": {
            key: selected_pooled[key] for key in ("arm", "regime", "size")
        },
        "selected_for_sequence": [
            {key: row[key] for key in ("arm", "regime", "size")}
            for row in sequence_candidates
        ],
        "selection": {
            "best_break_pr_auc": best_break,
            "best_break_pr_auc_se": best_report["oof_metrics"]["break_pr_auc_bootstrap_se"],
            "one_se_floor": one_se_floor,
            "source_safety_reference": {
                source: metrics["false_connect_count"]
                for source, metrics in baseline["by_source"].items()
            },
        },
    }
    np.savez_compressed(output / "oof_probabilities.npz", **predictions)
    np.savez_compressed(output / "oof_thresholds.npz", **thresholds)
    _write_json_new(output / "selected.json", selected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_pooled_regime_size_screen",
        "validation_opened": False,
        "deployment_approved": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "genuine_evaluation_count": len(genuine_rows),
        "genuine_break_count": int(np.count_nonzero(table.targets[genuine_rows] == 0)),
        "genuine_connect_count": int(np.count_nonzero(table.targets[genuine_rows] == 1)),
        "configurations": reports,
        "selected": selected,
        "interpretation_guard": (
            "Only genuine deployment-threshold candidates are scored. Synthetic regimes "
            "are training-only; the external dual-reviewed safety audit remains required."
        ),
        "inputs": {
            "table_manifest_sha256": sha256_file(table.root / "manifest.json"),
            "table_receipt_sha256": sha256_file(table.root / "receipt.json"),
        },
    }
    _write_json_new(output / "report.json", report)
    descriptor = os.open(output / "README.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        metrics = selected_pooled["oof_metrics"]
        handle.write(
            "# Bibliography gap regime/size screen\n\n"
            f"Selected pooled arm: `{selected_pooled['arm']}` / `{selected_pooled['regime']}` / "
            f"`{selected_pooled['size']}` negative-boundary rung.\n\n"
            f"- break PR-AUC: {metrics['break_pr_auc']:.6f}\n"
            f"- connect precision: {metrics['connect_precision']:.6f}\n"
            f"- connect recall: {metrics['connect_recall']:.6f}\n"
            f"- false connects: {metrics['false_connect_count']}\n\n"
            "This is grouped train-OOF silver research, not deployment approval.\n"
        )
    _write_json_new(output / "receipt.json", {**report, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-false-connect-rate", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

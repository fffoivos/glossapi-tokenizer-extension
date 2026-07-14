#!/usr/bin/env python3
"""Nested work-grouped P0/P0D/P1/P1G entry-anchor experiment."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import (
    PINNED_SKLEARN_VERSION,
    _best_threshold,
    binary_metrics,
    load_table,
)
from .bibliography_positional_features import FEATURE_NAMES, GAP_SUMMARY_NAMES, POSITION_SUMMARY_NAMES
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-positional-entry-ladder-v1"
LINEAR_ARMS = ("P0", "P1", "P1G")
ALL_ARMS = ("P0", "P0D", "P1", "P1G")


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}")
    return {
        "version": sklearn.__version__,
        "logistic": LogisticRegression,
        "hist": HistGradientBoostingClassifier,
        "average_precision": average_precision_score,
    }


@dataclass(frozen=True)
class PositionalTable:
    root: Path
    manifest: Mapping[str, Any]
    position_summaries: np.ndarray
    gap_summaries: np.ndarray


def load_positional_table(root: str | Path, base_manifest_sha256: str, n: int) -> PositionalTable:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "bibliography-positional-feature-table-v2":
        raise ValueError("unsupported positional table")
    if manifest.get("base_table", {}).get("manifest_sha256") != base_manifest_sha256:
        raise ValueError("positional/base table provenance mismatch")
    position = np.load(root / "position_summaries.npy", mmap_mode="r", allow_pickle=False)
    gaps = np.load(root / "gap_summaries.npy", mmap_mode="r", allow_pickle=False)
    if position.shape != (n, len(FEATURE_NAMES), len(POSITION_SUMMARY_NAMES)):
        raise ValueError("positional summary shape mismatch")
    if gaps.shape != (n, len(GAP_SUMMARY_NAMES)):
        raise ValueError("gap summary shape mismatch")
    return PositionalTable(root, manifest, position, gaps)


def load_targets(root: str | Path, base_manifest_sha256: str, n: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "bibliography-role-target-view-v1":
        raise ValueError("unsupported role target view")
    if manifest.get("base_table_manifest_sha256") != base_manifest_sha256:
        raise ValueError("role/base table provenance mismatch")
    targets = np.load(root / "entry_targets.npy", mmap_mode="r", allow_pickle=False)
    if targets.shape != (n,) or not set(np.unique(targets)) <= {-1, 0, 1}:
        raise ValueError("entry target array is malformed")
    return targets, manifest


@dataclass
class Transform:
    arm: str
    mean: np.ndarray
    scale: np.ndarray

    @staticmethod
    def continuous(arm: str, counts: np.ndarray, position: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        parts = [np.log1p(counts.astype(np.float32))]
        if arm in {"P1", "P1G"}:
            parts.append(position.reshape(len(position), -1).astype(np.float32))
        if arm == "P1G":
            parts.append(gaps.astype(np.float32))
        return np.concatenate(parts, axis=1)

    @classmethod
    def fit(
        cls, arm: str, counts: np.ndarray, position: np.ndarray, gaps: np.ndarray,
        indices: np.ndarray,
    ) -> "Transform":
        continuous = cls.continuous(arm, counts[indices], position[indices], gaps[indices])
        mean, scale = continuous.mean(axis=0), continuous.std(axis=0)
        scale[scale < 1.0e-12] = 1.0
        return cls(arm, mean, scale)

    def apply(self, counts: np.ndarray, position: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        presence = (counts > 0).astype(np.float32)
        continuous = self.continuous(self.arm, counts, position, gaps)
        scaled = ((continuous - self.mean) / self.scale).astype(np.float32)
        return np.concatenate((presence, scaled), axis=1)


def feature_names(arm: str) -> list[str]:
    names = [f"presence:{name}" for name in FEATURE_NAMES] + [f"log1p:{name}" for name in FEATURE_NAMES]
    if arm in {"P1", "P1G"}:
        names.extend(
            f"position:{feature}:{summary}"
            for feature in FEATURE_NAMES for summary in POSITION_SUMMARY_NAMES
        )
    if arm == "P1G":
        names.extend(f"gap:{name}" for name in GAP_SUMMARY_NAMES)
    return names


def _fit_linear(
    arm: str, counts: np.ndarray, position: np.ndarray, gaps: np.ndarray,
    targets: np.ndarray, fit_indices: np.ndarray, *, c_value: float, seed: int,
) -> tuple[Any, Transform]:
    transform = Transform.fit(arm, counts, position, gaps, fit_indices)
    x = transform.apply(counts[fit_indices], position[fit_indices], gaps[fit_indices])
    model = _sklearn()["logistic"](
        C=float(c_value), fit_intercept=True, l1_ratio=0.25, solver="saga",
        max_iter=250, random_state=int(seed), tol=1.0e-5,
    )
    model.fit(x, targets[fit_indices])
    return model, transform


def _predict(
    model: Any, transform: Transform, counts: np.ndarray, position: np.ndarray,
    gaps: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(transform.apply(counts[indices], position[indices], gaps[indices]))[:, 1]


def _fit_nested_linear(
    arm: str, counts: np.ndarray, position: np.ndarray, gaps: np.ndarray,
    targets: np.ndarray, folds: np.ndarray, labelled: np.ndarray, *, n_folds: int,
    c_grid: Sequence[float], seed: int, model_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.full(len(targets), np.nan, dtype=np.float32)
    reports: list[dict[str, Any]] = []
    average_precision = _sklearn()["average_precision"]
    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        inner_fit = np.flatnonzero(labelled & (folds != outer) & (folds != inner))
        inner_holdout = np.flatnonzero(labelled & (folds == inner))
        candidates = []
        for index, c_value in enumerate(c_grid):
            model, transform = _fit_linear(
                arm, counts, position, gaps, targets, inner_fit,
                c_value=float(c_value), seed=seed + outer * 100 + index,
            )
            prediction = _predict(model, transform, counts, position, gaps, inner_holdout)
            candidates.append(
                {
                    "C": float(c_value),
                    "inner_pr_auc": float(average_precision(targets[inner_holdout] == 1, prediction)),
                }
            )
        selected = max(candidates, key=lambda row: (row["inner_pr_auc"], -row["C"]))
        fit = np.flatnonzero(labelled & (folds != outer))
        holdout = np.flatnonzero(folds == outer)
        model, transform = _fit_linear(
            arm, counts, position, gaps, targets, fit,
            c_value=float(selected["C"]), seed=seed + outer,
        )
        probability[holdout] = _predict(model, transform, counts, position, gaps, holdout)
        with (model_dir / f"{arm}.fold{outer}.pkl").open("xb") as handle:
            pickle.dump((model, transform), handle, protocol=5)
        reports.append(
            {
                "outer_fold": outer,
                "inner_holdout_fold": inner,
                "fit_labelled_lines": len(fit),
                "outer_holdout_all_lines": len(holdout),
                "selected_C": selected["C"],
                "candidates": candidates,
                "iterations": int(np.max(model.n_iter_)),
                "feature_count": len(feature_names(arm)),
            }
        )
    return probability, reports


def _fit_p0d(
    counts: np.ndarray, targets: np.ndarray, folds: np.ndarray, labelled: np.ndarray,
    *, n_folds: int, seed: int, model_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.full(len(targets), np.nan, dtype=np.float32)
    reports = []
    features = np.concatenate(((counts > 0).astype(np.float32), np.log1p(counts.astype(np.float32))), axis=1)
    for fold in range(n_folds):
        fit = labelled & (folds != fold)
        holdout = folds == fold
        model = _sklearn()["hist"](
            learning_rate=0.08, max_iter=75, max_depth=3, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=seed + fold,
        )
        model.fit(features[fit], targets[fit])
        probability[holdout] = model.predict_proba(features[holdout])[:, 1]
        with (model_dir / f"P0D.fold{fold}.pkl").open("xb") as handle:
            pickle.dump(model, handle, protocol=5)
        reports.append(
            {"outer_fold": fold, "fit_labelled_lines": int(np.count_nonzero(fit)),
             "outer_holdout_all_lines": int(np.count_nonzero(holdout)), "feature_count": features.shape[1]}
        )
    return probability, reports


def _source_metrics(table: Any, targets: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    result = {}
    for source in sorted({str(row["source"]) for row in table.documents}):
        documents = [index for index, row in enumerate(table.documents) if row["source"] == source]
        mask = (targets != -1) & np.isin(table.document_indices, documents)
        result[source] = binary_metrics(targets[mask], probability[mask], threshold) if np.any(mask) else {"line_count": 0}
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.base_table_dir, expected_split="train")
    base_sha = sha256_file(table.root / "manifest.json")
    positional = load_positional_table(args.positional_table_dir, base_sha, len(table.targets))
    targets, target_manifest = load_targets(args.role_target_dir, base_sha, len(table.targets))
    labelled = targets != -1
    if not np.any(targets == 1) or not np.any(targets == 0):
        raise ValueError("entry ladder needs positive and negative targets")
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model_dir = output / "models"
    model_dir.mkdir()
    n_folds = int(table.manifest["n_folds"])
    results: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    c_grid = tuple(float(value) for value in args.c_grid)
    for index, arm in enumerate(LINEAR_ARMS):
        probability, folds = _fit_nested_linear(
            arm, table.counts, positional.position_summaries, positional.gap_summaries,
            targets, table.folds, labelled, n_folds=n_folds, c_grid=c_grid,
            seed=int(args.seed) + index * 1000, model_dir=model_dir,
        )
        probabilities[arm], results[arm] = probability, {"folds": folds}
    probability, folds = _fit_p0d(
        table.counts, targets, table.folds, labelled, n_folds=n_folds,
        seed=int(args.seed) + 9000, model_dir=model_dir,
    )
    probabilities["P0D"], results["P0D"] = probability, {"folds": folds}
    for arm in ALL_ARMS:
        probability = probabilities[arm]
        threshold = _best_threshold(targets[labelled], probability[labelled])
        results[arm].update(
            {
                "oof_metrics": binary_metrics(targets[labelled], probability[labelled], threshold["threshold"]),
                "selected_oof_threshold": threshold,
                "by_source": _source_metrics(table, targets, probability, threshold["threshold"]),
            }
        )
        _save(output / f"{arm}.oof_probability.npy", probability)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_nested_train_oof_validation_unopened",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "sklearn_version": _sklearn()["version"],
        "base_table_manifest_sha256": base_sha,
        "positional_table_manifest_sha256": sha256_file(positional.root / "manifest.json"),
        "role_target_manifest_sha256": sha256_file(Path(args.role_target_dir).resolve() / "manifest.json"),
        "overlay_sha256": target_manifest["overlay"]["sha256"],
        "labelled_line_count": int(np.count_nonzero(labelled)),
        "positive_line_count": int(np.count_nonzero(targets == 1)),
        "negative_line_count": int(np.count_nonzero(targets == 0)),
        "masked_line_count": int(np.count_nonzero(targets == -1)),
        "c_grid": list(c_grid),
        "selection": "one work-grouped inner fold per outer fold; identical grid for each linear arm",
        "arms": results,
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output / "report.json", report)
    outputs = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    receipt = {**report, "outputs": outputs}
    _write_json(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--positional-table-dir", required=True)
    parser.add_argument("--role-target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--c-grid", nargs="+", type=float, default=(0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

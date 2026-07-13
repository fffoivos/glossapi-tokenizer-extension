#!/usr/bin/env python3
"""Out-of-fold bibliography entry-line model ladder.

The ladder consumes only the 35 explicit bibliography feature counts.  It does
not receive line length, document position, neighbouring features, text, or
silver block identity.  Validation data is not accepted by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import FEATURE_NAMES, TARGET_ENTRY, TARGET_MASK


SCHEMA_VERSION = "bibliography-entry-line-ladder-v1"
PINNED_SKLEARN_VERSION = "1.9.0"
LINEAR_ARMS = ("L1", "L2", "L3", "L4")
ALL_ARMS = ("L0",) + LINEAR_ARMS + ("D1",)


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    return {
        "sklearn": sklearn,
        "LogisticRegression": LogisticRegression,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "average_precision_score": average_precision_score,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Table:
    root: Path
    manifest: Mapping[str, Any]
    documents: tuple[Mapping[str, Any], ...]
    counts: np.ndarray
    targets: np.ndarray
    original_labels: np.ndarray
    header_kinds: np.ndarray
    abs_indices: np.ndarray
    token_counts: np.ndarray
    char_lengths: np.ndarray
    block_indices: np.ndarray
    document_indices: np.ndarray
    folds: np.ndarray


def load_table(root: str | Path) -> Table:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "bibliography-entry-feature-table-v1":
        raise ValueError("unsupported feature-table schema")
    if manifest.get("split") != "train":
        raise ValueError("line ladder accepts a train-only feature table")
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "counts",
            "targets",
            "original_labels",
            "header_kinds",
            "abs_indices",
            "token_counts",
            "char_lengths",
            "block_indices",
            "document_indices",
            "folds",
        )
    }
    n_lines = int(manifest["line_count"])
    if arrays["counts"].shape != (n_lines, len(FEATURE_NAMES)):
        raise ValueError("feature matrix shape does not match manifest")
    if any(len(value) != n_lines for name, value in arrays.items() if name != "counts"):
        raise ValueError("line-array length mismatch")
    return Table(
        root=root,
        manifest=manifest,
        documents=tuple(_json_rows(root / "documents.jsonl")),
        **arrays,
    )


@dataclass
class Transform:
    arm: str
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arm: str, counts: np.ndarray) -> "Transform":
        if arm not in (*LINEAR_ARMS, "D1"):
            raise ValueError(f"unsupported transform arm {arm}")
        if arm == "L1":
            return cls(arm, np.zeros(counts.shape[1]), np.ones(counts.shape[1]))
        log_counts = np.log1p(counts.astype(np.float64))
        basis = counts.astype(np.float64) if arm == "L2" else log_counts
        mean = basis.mean(axis=0)
        scale = basis.std(axis=0)
        scale[scale < 1.0e-12] = 1.0
        return cls(arm, mean, scale)

    def apply(self, counts: np.ndarray) -> np.ndarray:
        presence = (counts > 0).astype(np.float32)
        if self.arm == "L1":
            return presence
        log_counts = np.log1p(counts.astype(np.float32))
        if self.arm == "L2":
            return ((counts.astype(np.float32) - self.mean) / self.scale).astype(np.float32)
        if self.arm == "L3":
            return ((log_counts - self.mean) / self.scale).astype(np.float32)
        if self.arm == "L4":
            scaled = ((log_counts - self.mean) / self.scale).astype(np.float32)
            return np.concatenate((presence, scaled), axis=1)
        if self.arm == "D1":
            return np.concatenate((presence, log_counts), axis=1)
        raise AssertionError(self.arm)

    def metadata(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _best_threshold(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    candidates = np.unique(
        np.concatenate((np.linspace(0.05, 0.95, 19), np.quantile(probability, np.linspace(0.02, 0.98, 49))))
    )
    for threshold in candidates:
        metrics = binary_metrics(y, probability, float(threshold), include_ap=False)
        row = {"threshold": float(threshold), **metrics}
        if best is None or (
            row["f1"], row["precision"], -abs(row["threshold"] - 0.5)
        ) > (
            best["f1"], best["precision"], -abs(best["threshold"] - 0.5)
        ):
            best = row
    assert best is not None
    return best


def binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
    *,
    include_ap: bool = True,
) -> dict[str, float]:
    if len(y) != len(probability) or not len(y):
        raise ValueError("metrics need equal non-empty arrays")
    guess = probability >= threshold
    truth = y == TARGET_ENTRY
    tp = int(np.count_nonzero(guess & truth))
    fp = int(np.count_nonzero(guess & ~truth))
    fn = int(np.count_nonzero(~guess & truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = 0.25
    f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if beta2 * precision + recall else 0.0
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f05,
        "brier": float(np.mean((probability - truth.astype(float)) ** 2)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    if include_ap:
        result["pr_auc"] = float(_sklearn()["average_precision_score"](truth, probability))
    return result


def _fit_linear(
    arm: str,
    counts: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[Any, Transform]:
    transform = Transform.fit(arm, counts[fit_mask])
    x = transform.apply(counts[fit_mask])
    kwargs: dict[str, Any] = {
        "C": float(c_value),
        "fit_intercept": True,
        "max_iter": 250,
        "random_state": int(seed),
        "tol": 1.0e-5,
    }
    if arm == "L4":
        kwargs.update(l1_ratio=0.25, solver="saga")
    else:
        kwargs.update(l1_ratio=0.0, solver="lbfgs")
    model = _sklearn()["LogisticRegression"](**kwargs)
    model.fit(x, targets[fit_mask])
    return model, transform


def _linear_original_units(model: Any, transform: Transform) -> dict[str, Any]:
    coefficient = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])
    if transform.arm == "L1":
        names = [f"presence:{name}" for name in FEATURE_NAMES]
        original = coefficient
        original_intercept = intercept
    elif transform.arm in {"L2", "L3"}:
        prefix = "count" if transform.arm == "L2" else "log1p"
        names = [f"{prefix}:{name}" for name in FEATURE_NAMES]
        original = coefficient / transform.scale
        original_intercept = intercept - float(np.sum(coefficient * transform.mean / transform.scale))
    elif transform.arm == "L4":
        split = len(FEATURE_NAMES)
        names = [f"presence:{name}" for name in FEATURE_NAMES] + [f"log1p:{name}" for name in FEATURE_NAMES]
        original = np.concatenate((coefficient[:split], coefficient[split:] / transform.scale))
        original_intercept = intercept - float(np.sum(coefficient[split:] * transform.mean / transform.scale))
    else:
        raise ValueError("D1 has no linear coefficient export")
    return {
        "feature_names": names,
        "coefficients": original.tolist(),
        "bias": original_intercept,
    }


def _fit_d1(
    counts: np.ndarray, targets: np.ndarray, fit_mask: np.ndarray, *, seed: int
) -> tuple[Any, Transform]:
    transform = Transform.fit("D1", counts[fit_mask])
    model = _sklearn()["HistGradientBoostingClassifier"](
        learning_rate=0.08,
        max_iter=75,
        max_depth=3,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(transform.apply(counts[fit_mask]), targets[fit_mask])
    return model, transform


def _pickle(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        pickle.dump(value, handle, protocol=5)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _l0_oof(table: Table, labelled: np.ndarray, n_folds: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = np.count_nonzero(table.counts, axis=1).astype(np.float32)
    probability = np.full(len(table.targets), np.nan, dtype=np.float32)
    fold_rows = []
    for fold in range(n_folds):
        fit = labelled & (table.folds != fold)
        holdout = table.folds == fold
        best = None
        for threshold in range(1, len(FEATURE_NAMES) + 1):
            guess = points[fit] >= threshold
            truth = table.targets[fit] == TARGET_ENTRY
            tp = int(np.count_nonzero(guess & truth))
            fp = int(np.count_nonzero(guess & ~truth))
            fn = int(np.count_nonzero(~guess & truth))
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidate = (f1, precision, -threshold, threshold)
            if best is None or candidate > best:
                best = candidate
        threshold = int(best[-1])
        probability[holdout] = _sigmoid(points[holdout] - threshold)
        fold_rows.append({"fold": fold, "point_threshold": threshold})
    return probability, fold_rows


def _fit_candidate_oof(
    table: Table,
    arm: str,
    labelled: np.ndarray,
    n_folds: int,
    *,
    c_value: float | None,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[Any, Transform]]]:
    probability = np.full(len(table.targets), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    models: list[tuple[Any, Transform]] = []
    for fold in range(n_folds):
        fit = labelled & (table.folds != fold)
        holdout = table.folds == fold
        if arm == "D1":
            model, transform = _fit_d1(table.counts, table.targets, fit, seed=seed + fold)
        else:
            assert c_value is not None
            model, transform = _fit_linear(
                arm,
                table.counts,
                table.targets,
                fit,
                c_value=c_value,
                seed=seed + fold,
            )
        probability[holdout] = model.predict_proba(transform.apply(table.counts[holdout]))[:, 1]
        fold_row: dict[str, Any] = {
            "fold": fold,
            "fit_lines": int(np.count_nonzero(fit)),
            "holdout_lines": int(np.count_nonzero(holdout)),
            "iterations": int(np.max(getattr(model, "n_iter_", [0]))),
        }
        if arm in LINEAR_ARMS:
            fold_row["original_units"] = _linear_original_units(model, transform)
        fold_rows.append(fold_row)
        models.append((model, transform))
    return probability, fold_rows, models


def _error_diagnostics(table: Table, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    guess = probability >= threshold
    negative = table.targets == 0
    false = guess & negative
    false_documents = set(int(value) for value in table.document_indices[false])
    longest = 0
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        run = 0
        for value in false[start:end]:
            run = run + 1 if value else 0
            longest = max(longest, run)
    masked_headers = table.header_kinds > 0
    return {
        "false_positive_document_count": len(false_documents),
        "longest_consecutive_false_positive_lines": longest,
        "masked_header_fire_rate": float(np.mean(guess[masked_headers])) if masked_headers.any() else 0.0,
    }


def _group_metrics(
    table: Table, probability: np.ndarray, threshold: float, key: str
) -> dict[str, Any]:
    result = {}
    doc_values = [str(document[key]) for document in table.documents]
    for value in sorted(set(doc_values)):
        docs = {index for index, item in enumerate(doc_values) if item == value}
        mask = (table.targets != TARGET_MASK) & np.isin(table.document_indices, list(docs))
        result[value] = binary_metrics(table.targets[mask], probability[mask], threshold)
    return result


def _length_metrics(table: Table, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    bands = (("<=110", 0, 110), ("111-220", 111, 220), ("221-330", 221, 330), (">330", 331, np.iinfo(np.uint32).max))
    result = {}
    for name, low, high in bands:
        mask = (table.targets != TARGET_MASK) & (table.char_lengths >= low) & (table.char_lengths <= high)
        result[name] = (
            {"line_count": 0}
            if not np.any(mask)
            else {
                "line_count": int(np.count_nonzero(mask)),
                **binary_metrics(table.targets[mask], probability[mask], threshold),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    sklearn_modules = _sklearn()
    table = load_table(args.table_dir)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    n_folds = int(table.manifest["n_folds"])
    labelled = table.targets != TARGET_MASK
    if set(np.unique(table.folds)) != set(range(n_folds)):
        raise ValueError("feature table does not contain every declared fold")

    arm_results: dict[str, Any] = {}
    selected_probabilities: dict[str, np.ndarray] = {}
    l0_probability, l0_folds = _l0_oof(table, labelled, n_folds)
    l0_threshold = _best_threshold(table.targets[labelled], l0_probability[labelled])
    arm_results["L0"] = {"selected": {"kind": "equal_presence", **l0_threshold}, "folds": l0_folds}
    selected_probabilities["L0"] = l0_probability

    c_grid = tuple(float(value) for value in args.c_grid)
    for arm_index, arm in enumerate(LINEAR_ARMS):
        candidates = []
        cached: dict[float, tuple[np.ndarray, list[dict[str, Any]], list[tuple[Any, Transform]]]] = {}
        for c_value in c_grid:
            probability, folds, models = _fit_candidate_oof(
                table,
                arm,
                labelled,
                n_folds,
                c_value=c_value,
                seed=int(args.seed) + arm_index * 100,
            )
            metrics = binary_metrics(table.targets[labelled], probability[labelled])
            candidates.append({"C": c_value, **metrics})
            cached[c_value] = (probability, folds, models)
        selected = max(candidates, key=lambda row: (row["pr_auc"], row["f0_5"], -row["C"]))
        probability, folds, models = cached[float(selected["C"])]
        threshold = _best_threshold(table.targets[labelled], probability[labelled])
        for fold, bundle in enumerate(models):
            _pickle(model_dir / f"{arm}.fold{fold}.pkl", bundle)
        arm_results[arm] = {"candidates": candidates, "selected": {**selected, **threshold}, "folds": folds}
        selected_probabilities[arm] = probability

    d1_probability, d1_folds, d1_models = _fit_candidate_oof(
        table,
        "D1",
        labelled,
        n_folds,
        c_value=None,
        seed=int(args.seed) + 500,
    )
    d1_threshold = _best_threshold(table.targets[labelled], d1_probability[labelled])
    for fold, bundle in enumerate(d1_models):
        _pickle(model_dir / f"D1.fold{fold}.pkl", bundle)
    arm_results["D1"] = {
        "selected": {"kind": "hist_gradient_boosting_depth3", **binary_metrics(table.targets[labelled], d1_probability[labelled]), **d1_threshold},
        "folds": d1_folds,
    }
    selected_probabilities["D1"] = d1_probability

    report_arms = {}
    for arm in ALL_ARMS:
        probability = selected_probabilities[arm]
        threshold = float(arm_results[arm]["selected"]["threshold"])
        _save_array(output_dir / f"{arm}.oof_probability.npy", probability)
        report_arms[arm] = {
            **arm_results[arm],
            "oof_metrics": binary_metrics(table.targets[labelled], probability[labelled], threshold),
            "diagnostics": _error_diagnostics(table, probability, threshold),
            "by_source": _group_metrics(table, probability, threshold, "source"),
            "by_coverage": _group_metrics(table, probability, threshold, "coverage"),
            "by_length": _length_metrics(table, probability, threshold),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_only_validation_unopened",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "feature_table_manifest_sha256": _sha256(table.root / "manifest.json"),
        "sklearn_version": sklearn_modules["sklearn"].__version__,
        "natural_prevalence": float(np.mean(table.targets[labelled] == TARGET_ENTRY)),
        "labelled_line_count": int(np.count_nonzero(labelled)),
        "masked_line_count": int(np.count_nonzero(~labelled)),
        "c_grid": list(c_grid),
        "elastic_net_l1_ratio": 0.25,
        "arms": report_arms,
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "line_oof_report.json", report)
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            outputs[str(path.relative_to(output_dir))] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--c-grid", type=float, nargs="+", default=(0.1, 1.0))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

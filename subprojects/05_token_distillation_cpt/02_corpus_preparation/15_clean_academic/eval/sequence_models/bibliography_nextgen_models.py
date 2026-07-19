#!/usr/bin/env python3
"""Train grouped out-of-fold full-document bibliography line models.

These models do not replace the frozen entry detector.  They consume its OOF
probability together with role, deterministic, shape, and local-context
features to estimate whether each line belongs to a bibliography region.  A
separate constrained decoder remains responsible for emitting blocks.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-line-oof-v1"
CONTEXT_SIGNALS = (
    "probability:entry",
    "probability:signal_tcn",
    "probability:continuation_specialist",
    "probability:continuation",
    "probability:filler",
    "probability:bib_header",
    "probability:bib_subheader",
    "probability:non_bib_header",
)
CONTEXT_RADII = (1, 3, 8)


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
        "version": sklearn.__version__,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def context_feature_names(base_names: Sequence[str]) -> tuple[str, ...]:
    result = list(base_names)
    for signal in CONTEXT_SIGNALS:
        for direction in ("above", "below"):
            for radius in CONTEXT_RADII:
                result.extend(
                    (
                        f"context:{signal}:{direction}:r{radius}:max",
                        f"context:{signal}:{direction}:r{radius}:mean",
                    )
                )
    result.extend(
        (
            "context:nearest_entry_anchor_above",
            "context:nearest_entry_anchor_below",
            "context:markdown_heading_above_3",
            "context:markdown_heading_below_3",
        )
    )
    return tuple(result)


def _physical_slices(abs_indices: np.ndarray) -> list[tuple[int, int]]:
    starts = [0]
    starts.extend(
        int(index)
        for index in np.flatnonzero(
            np.diff(abs_indices.astype(np.int64)) > MAX_PHYSICAL_GAP
        )
        + 1
    )
    starts.append(len(abs_indices))
    return list(zip(starts[:-1], starts[1:], strict=True))


def _directional_summary(
    values: np.ndarray, start: int, end: int, radius: int, direction: int
) -> tuple[float, float]:
    if direction < 0:
        low, high = max(start, end - radius), end
    else:
        low, high = start, min(end, start + radius)
    selected = values[low:high]
    return (
        float(selected.max(initial=0.0)),
        float(selected.mean()) if len(selected) else 0.0,
    )


def build_context_features(
    features: np.ndarray,
    names: Sequence[str],
    table: Any,
) -> np.ndarray:
    """Append bounded bidirectional summaries without crossing physical gaps."""

    name_to_index = {name: index for index, name in enumerate(names)}
    missing = set(CONTEXT_SIGNALS) - set(name_to_index)
    if missing:
        raise ValueError(f"nextgen table lacks context signals: {sorted(missing)}")
    markdown_index = name_to_index["structure:markdown_heading"]
    extra_count = len(context_feature_names(names)) - len(names)
    result = np.empty((len(features), len(names) + extra_count), dtype=np.float32)
    result[:, : len(names)] = features
    for document in table.documents:
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local_abs = table.abs_indices[doc_start:doc_end]
        for segment_start, segment_end in _physical_slices(local_abs):
            absolute_start, absolute_end = doc_start + segment_start, doc_start + segment_end
            local = features[absolute_start:absolute_end]
            entry = local[:, name_to_index["probability:entry"]]
            markdown = local[:, markdown_index]
            previous_anchor: int | None = None
            above_distance = np.full(len(local), 31.0, dtype=np.float32)
            for index in range(len(local)):
                if entry[index] >= 0.25:
                    previous_anchor = index
                if previous_anchor is not None:
                    above_distance[index] = min(31, index - previous_anchor)
            following_anchor: int | None = None
            below_distance = np.full(len(local), 31.0, dtype=np.float32)
            for index in range(len(local) - 1, -1, -1):
                if entry[index] >= 0.25:
                    following_anchor = index
                if following_anchor is not None:
                    below_distance[index] = min(31, following_anchor - index)
            for index in range(len(local)):
                values: list[float] = []
                for signal in CONTEXT_SIGNALS:
                    signal_values = local[:, name_to_index[signal]]
                    for direction in (-1, 1):
                        for radius in CONTEXT_RADII:
                            if direction < 0:
                                low, high = max(0, index - radius), index
                            else:
                                low, high = index + 1, min(len(local), index + radius + 1)
                            selected = signal_values[low:high]
                            values.extend(
                                (
                                    float(selected.max(initial=0.0)),
                                    float(selected.mean()) if len(selected) else 0.0,
                                )
                            )
                values.extend(
                    (
                        float(above_distance[index]),
                        float(below_distance[index]),
                        float(markdown[max(0, index - 3) : index].max(initial=0.0)),
                        float(markdown[index + 1 : min(len(local), index + 4)].max(initial=0.0)),
                    )
                )
                result[absolute_start + index, len(names) :] = np.asarray(
                    values, dtype=np.float32
                )
    if not np.isfinite(result).all():
        raise RuntimeError("context feature matrix contains non-finite values")
    return result


def _fit_inventory(
    table: Any,
    features: np.ndarray,
    names: Sequence[str],
    *,
    maximum_negative_ratio: float,
    seed: int,
) -> np.ndarray:
    target = table.original_labels == LABEL_TO_ID["BIB"]
    trusted = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    hard = (
        (features[:, names.index("probability:entry")] >= 0.05)
        | (features[:, names.index("probability:signal_tcn")] >= 0.05)
        | (features[:, names.index("structure:markdown_heading")] > 0)
        | (features[:, names.index("structure:table_row")] > 0)
    )
    near = np.zeros(len(target), dtype=bool)
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        positive = np.flatnonzero(target[start:end])
        for index in positive:
            near[start + max(0, int(index) - 30) : start + min(end - start, int(index) + 31)] = True
    eligible = trusted & (target | hard | near)
    positive_count = int(np.count_nonzero(eligible & target))
    allowed_negative = int(maximum_negative_ratio * positive_count)
    negative = np.flatnonzero(eligible & ~target)
    if len(negative) > allowed_negative:
        rng = np.random.default_rng(seed)
        keep = rng.choice(negative, size=allowed_negative, replace=False)
        eligible[negative] = False
        eligible[keep] = True
    return eligible


def _weights(indices: np.ndarray, target: np.ndarray, documents: np.ndarray) -> np.ndarray:
    doc_counts = collections.Counter(int(documents[index]) for index in indices)
    class_counts = np.bincount(target[indices].astype(np.uint8), minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("training partition needs both classes")
    result = np.asarray(
        [
            1.0
            / doc_counts[int(documents[index])]
            * len(indices)
            / (2.0 * class_counts[int(target[index])])
            for index in indices
        ],
        dtype=np.float64,
    )
    return result / result.mean()


@dataclass
class ModelBundle:
    kind: str
    scaler: Any
    model: Any

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (
            self.scaler.transform(features).astype(np.float32)
            if self.scaler is not None
            else features
        )
        return self.model.predict_proba(values)[:, 1].astype(np.float32)


def _fit(
    kind: str,
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    documents: np.ndarray,
    *,
    seed: int,
) -> ModelBundle:
    tools = _sklearn()
    weights = _weights(indices, target, documents)
    if kind == "linear":
        scaler = tools["scaler"]().fit(features[indices], sample_weight=weights)
        transformed = scaler.transform(features[indices]).astype(np.float32)
        model = tools["logistic"](
            C=0.1,
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        ).fit(transformed, target[indices], sample_weight=weights)
        return ModelBundle(kind, scaler, model)
    if kind == "hist":
        model = tools["hist"](
            learning_rate=0.05,
            max_iter=250,
            max_depth=4,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ).fit(features[indices], target[indices], sample_weight=weights)
        return ModelBundle(kind, None, model)
    raise ValueError(f"unknown model kind: {kind}")


def _raw_metrics(target: np.ndarray, probability: np.ndarray, trusted: np.ndarray) -> dict[str, Any]:
    selected = trusted.astype(bool)
    truth = target[selected]
    values = probability[selected]
    prediction = values >= 0.5
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    return {
        "pr_auc": float(_sklearn()["average_precision"](truth, values)),
        "brier": float(np.mean((values - truth.astype(float)) ** 2)),
        "precision_at_0_5": tp / (tp + fp) if tp + fp else 1.0,
        "recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
        "tp_at_0_5": tp,
        "fp_at_0_5": fp,
        "fn_at_0_5": fn,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root = Path(args.table_dir).resolve()
    base_root = Path(args.base_table_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TABLE_SCHEMA or manifest.get("test_opened") is not False:
        raise ValueError("training requires the sealed full-development feature table")
    table = load_table(base_root, expected_split="train")
    base_features = np.load(table_root / "features.npy", mmap_mode="r", allow_pickle=False)
    names = tuple(manifest["feature_names"])
    if base_features.shape != (len(table.targets), len(names)):
        raise ValueError("nextgen feature table shape mismatch")
    features = build_context_features(np.asarray(base_features), names, table)
    context_names = context_feature_names(names)
    target = table.original_labels == LABEL_TO_ID["BIB"]
    trusted = table.original_labels != LABEL_TO_ID["UNKNOWN"]
    eligible = _fit_inventory(
        table,
        features,
        context_names,
        maximum_negative_ratio=float(args.maximum_negative_ratio),
        seed=int(args.seed),
    )
    output.mkdir(parents=True)
    model_root = output / "models"
    model_root.mkdir()
    probability = np.full(len(target), np.nan, dtype=np.float32)
    fold_reports = []
    n_folds = int(table.manifest["n_folds"])
    for fold in range(n_folds):
        fit = np.flatnonzero(eligible & (table.folds != fold))
        holdout = np.flatnonzero(table.folds == fold)
        bundle = _fit(
            args.kind,
            features,
            target,
            fit,
            table.document_indices,
            seed=int(args.seed) + fold,
        )
        probability[holdout] = bundle.predict(features[holdout])
        with (model_root / f"fold{fold}.pkl").open("xb") as handle:
            # A built-in container remains loadable when this module was run
            # through ``python -m``; pickling the local dataclass would bind it
            # to ``__main__.ModelBundle``.
            pickle.dump(
                {"kind": bundle.kind, "scaler": bundle.scaler, "model": bundle.model},
                handle,
                protocol=5,
            )
        fold_reports.append(
            {
                "fold": fold,
                "fit_line_count": len(fit),
                "holdout_line_count": len(holdout),
                "fit_positive_count": int(np.count_nonzero(target[fit])),
                "holdout_positive_count": int(np.count_nonzero(target[holdout])),
            }
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("OOF probabilities are incomplete")
    _save(output / "oof_probability.npy", probability)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_grouped_oof_full_document_line_training",
        "kind": args.kind,
        "validation_opened": False,
        "test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "feature_count": len(context_names),
        "feature_names": context_names,
        "eligible_fit_line_count": int(np.count_nonzero(eligible)),
        "eligible_positive_count": int(np.count_nonzero(eligible & target)),
        "maximum_negative_ratio": float(args.maximum_negative_ratio),
        "folds": fold_reports,
        "oof_metrics_before_block_decoding": _raw_metrics(target, probability, trusted),
        "inputs": {
            "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
            "table_features_sha256": sha256_file(table_root / "features.npy"),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
        },
        "sklearn_version": _sklearn()["version"],
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                str(path.relative_to(output)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=("linear", "hist"), required=True)
    parser.add_argument("--maximum-negative-ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

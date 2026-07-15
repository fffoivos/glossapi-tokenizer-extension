#!/usr/bin/env python3
"""Grouped OOF heading and continuation/filler expert models."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import PINNED_SKLEARN_VERSION
from .bibliography_role_v2 import ID_TO_ROLE, ROLE_TO_ID
from .contract import sha256_file


HEADING_SCHEMA = "bibliography-heading-expert-oof-v1"
CONNECTOR_SCHEMA = "bibliography-connector-expert-oof-v1"
HEADING_TYPES = ("BIB_HEADER", "BIB_SUBHEADER", "NON_BIB_HEADER")
HEADING_TYPE_TO_INDEX = {name: index for index, name in enumerate(HEADING_TYPES)}
HEADING_PROBABILITY_COLUMNS = (
    "any_header", "bib_header", "bib_subheader", "non_bib_header",
)
CONNECTOR_PROBABILITY_COLUMNS = (
    "connector", "continuation", "filler", "other",
)


def _sklearn() -> dict[str, Any]:
    import sklearn
    from scipy import sparse
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    return {
        "sparse": sparse,
        "tfidf": TfidfVectorizer,
        "logistic": LogisticRegression,
        "hist": HistGradientBoostingClassifier,
        "average_precision": average_precision_score,
        "scaler": StandardScaler,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _macro_ap(y: np.ndarray, probability: np.ndarray, classes: Sequence[int]) -> float:
    scores = []
    average_precision = _sklearn()["average_precision"]
    for class_id in classes:
        truth = y == class_id
        if not truth.any() or truth.all():
            continue
        scores.append(float(average_precision(truth, probability[:, class_id])))
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class HeadingTransform:
    char_vectorizer: Any
    word_vectorizer: Any
    numeric_scaler: Any

    @classmethod
    def fit(cls, texts: Sequence[str], numeric: np.ndarray) -> "HeadingTransform":
        tools = _sklearn()
        char = tools["tfidf"](
            analyzer="char_wb", ngram_range=(2, 5), min_df=2,
            sublinear_tf=True, lowercase=True, dtype=np.float32,
        )
        word = tools["tfidf"](
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, lowercase=True, dtype=np.float32,
            token_pattern=r"(?u)\b[^\W_]+(?:[’'\-][^\W_]+)*\b",
        )
        char.fit(texts)
        word.fit(texts)
        scaler = tools["scaler"](with_mean=False).fit(numeric)
        return cls(char, word, scaler)

    def apply(self, texts: Sequence[str], numeric: np.ndarray) -> Any:
        sparse = _sklearn()["sparse"]
        return sparse.hstack(
            (
                self.char_vectorizer.transform(texts),
                self.word_vectorizer.transform(texts),
                sparse.csr_matrix(self.numeric_scaler.transform(numeric).astype(np.float32)),
            ),
            format="csr",
            dtype=np.float32,
        )


@dataclass
class HeadingBundle:
    transform: HeadingTransform
    any_model: Any
    type_model: Any

    def predict(self, texts: Sequence[str], numeric: np.ndarray) -> np.ndarray:
        x = self.transform.apply(texts, numeric)
        any_probability = self.any_model.predict_proba(x)[:, 1]
        conditional = np.zeros((len(texts), len(HEADING_TYPES)), dtype=np.float64)
        raw = self.type_model.predict_proba(x)
        for column, class_id in enumerate(self.type_model.classes_):
            conditional[:, int(class_id)] = raw[:, column]
        typed = conditional * any_probability[:, None]
        return np.column_stack((any_probability, typed)).astype(np.float32)


def _fit_heading_bundle(
    texts: Sequence[str], numeric: np.ndarray, roles: np.ndarray,
    indices: np.ndarray, *, c_value: float, seed: int,
) -> HeadingBundle:
    tools = _sklearn()
    local_texts = [texts[int(index)] for index in indices]
    transform = HeadingTransform.fit(local_texts, numeric[indices])
    x = transform.apply(local_texts, numeric[indices])
    local_roles = roles[indices]
    heading_ids = np.asarray([ROLE_TO_ID[name] for name in HEADING_TYPES])
    any_target = np.isin(local_roles, heading_ids).astype(np.uint8)
    if len(np.unique(any_target)) != 2:
        raise ValueError("heading fit needs positive and negative candidates")
    any_model = tools["logistic"](
        C=float(c_value), solver="saga", l1_ratio=0.0, class_weight="balanced",
        max_iter=500, random_state=int(seed),
    ).fit(x, any_target)
    header_mask = np.isin(local_roles, heading_ids)
    type_target = np.asarray(
        [HEADING_TYPE_TO_INDEX[ID_TO_ROLE[int(role)]] for role in local_roles[header_mask]],
        dtype=np.uint8,
    )
    if len(np.unique(type_target)) != len(HEADING_TYPES):
        raise ValueError("heading type fit needs all three heading classes")
    type_model = tools["logistic"](
        C=float(c_value), solver="saga", l1_ratio=0.0, class_weight="balanced",
        max_iter=500, random_state=int(seed) + 1,
    ).fit(x[header_mask], type_target)
    return HeadingBundle(transform, any_model, type_model)


def fit_heading_oof(
    texts: Sequence[str], numeric: np.ndarray, roles: np.ndarray,
    trusted: np.ndarray, folds: np.ndarray, *, c_grid: Sequence[float], seed: int,
) -> tuple[np.ndarray, list[HeadingBundle], list[dict[str, Any]]]:
    n = len(texts)
    if not (numeric.shape[0] == roles.shape[0] == trusted.shape[0] == folds.shape[0] == n):
        raise ValueError("heading arrays are not aligned")
    n_folds = int(folds.max()) + 1
    probability = np.full((n, len(HEADING_PROBABILITY_COLUMNS)), np.nan, dtype=np.float32)
    bundles: list[HeadingBundle] = []
    reports: list[dict[str, Any]] = []
    heading_ids = np.asarray([ROLE_TO_ID[name] for name in HEADING_TYPES])
    labelled = trusted.astype(bool) & (roles != ROLE_TO_ID["UNKNOWN"])
    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        inner_fit = np.flatnonzero(labelled & (folds != outer) & (folds != inner))
        inner_holdout = np.flatnonzero(labelled & (folds == inner))
        candidates = []
        for offset, c_value in enumerate(c_grid):
            bundle = _fit_heading_bundle(
                texts, numeric, roles, inner_fit, c_value=float(c_value),
                seed=seed + outer * 100 + offset,
            )
            prediction = bundle.predict(
                [texts[int(index)] for index in inner_holdout], numeric[inner_holdout]
            )
            any_truth = np.isin(roles[inner_holdout], heading_ids)
            any_ap = float(_sklearn()["average_precision"](any_truth, prediction[:, 0]))
            type_rows = np.flatnonzero(any_truth)
            type_truth = np.asarray(
                [HEADING_TYPE_TO_INDEX[ID_TO_ROLE[int(role)]] for role in roles[inner_holdout][type_rows]]
            )
            type_ap = _macro_ap(type_truth, prediction[type_rows, 1:], range(len(HEADING_TYPES)))
            candidates.append({"C": float(c_value), "any_header_pr_auc": any_ap, "type_macro_pr_auc": type_ap})
        selected = max(candidates, key=lambda row: (row["any_header_pr_auc"], row["type_macro_pr_auc"], -row["C"]))
        fit = np.flatnonzero(labelled & (folds != outer))
        holdout = np.flatnonzero(folds == outer)
        bundle = _fit_heading_bundle(
            texts, numeric, roles, fit, c_value=float(selected["C"]), seed=seed + outer,
        )
        probability[holdout] = bundle.predict(
            [texts[int(index)] for index in holdout], numeric[holdout]
        )
        bundles.append(bundle)
        reports.append({
            "outer_fold": outer, "inner_fold": inner, "selected_C": selected["C"],
            "fit_labelled_candidates": len(fit), "holdout_candidates": len(holdout),
            "candidates": candidates,
        })
    if not np.isfinite(probability).all():
        raise RuntimeError("heading OOF probabilities are incomplete")
    return probability, bundles, reports


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValueError("binary evaluation requires positive and negative examples")
    return {
        "pr_auc": float(_sklearn()["average_precision"](y, p)),
        "brier": float(np.mean((p - y.astype(float)) ** 2)),
    }


def _conditional_continuation(probability: np.ndarray) -> np.ndarray:
    return np.divide(
        probability[:, 1], probability[:, 0],
        out=np.full(len(probability), 0.5, dtype=np.float32),
        where=probability[:, 0] > 1.0e-7,
    )


def _connector_targets(roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    continuation = roles == ROLE_TO_ID["CONTINUATION"]
    filler = roles == ROLE_TO_ID["FILLER"]
    connector = continuation | filler
    subtype_mask = connector
    subtype = continuation.astype(np.uint8)
    other = (roles == ROLE_TO_ID["OTHER"]).astype(np.uint8)
    return connector.astype(np.uint8), subtype, subtype_mask, other


@dataclass
class ConnectorBundle:
    arm: str
    connector_model: Any
    subtype_model: Any
    other_model: Any
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return features
        return ((features - self.mean) / self.scale).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        transformed = self.transform(features)
        connector = self.connector_model.predict_proba(transformed)[:, 1]
        continuation_conditional = self.subtype_model.predict_proba(transformed)[:, 1]
        continuation = connector * continuation_conditional
        filler = connector * (1.0 - continuation_conditional)
        other = self.other_model.predict_proba(transformed)[:, 1]
        return np.column_stack((connector, continuation, filler, other)).astype(np.float32)


def _fit_binary_model(arm: str, x: np.ndarray, y: np.ndarray, *, seed: int, c_value: float = 1.0,
                      max_depth: int = 3, min_samples_leaf: int = 20) -> Any:
    tools = _sklearn()
    if len(np.unique(y)) != 2:
        raise ValueError("binary expert fit requires two classes")
    if arm == "logistic":
        return tools["logistic"](
            C=float(c_value), solver="saga", l1_ratio=0.25,
            class_weight="balanced", max_iter=500, random_state=int(seed),
        ).fit(x, y)
    if arm == "hist":
        positives = max(1, int(np.count_nonzero(y)))
        negatives = max(1, len(y) - positives)
        weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
        return tools["hist"](
            learning_rate=0.05, max_iter=150, max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf), l2_regularization=1.0,
            early_stopping=False, random_state=int(seed),
        ).fit(x, y, sample_weight=weights)
    raise ValueError(f"unsupported connector arm: {arm}")


def _fit_connector_bundle(
    features: np.ndarray, connector_target: np.ndarray, subtype_target: np.ndarray,
    subtype_trusted: np.ndarray, other_target: np.ndarray, other_trusted: np.ndarray,
    indices: np.ndarray, *, arm: str,
    seed: int, c_value: float = 1.0, max_depth: int = 3, min_samples_leaf: int = 20,
) -> ConnectorBundle:
    kwargs = {"arm": arm, "c_value": c_value, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    transformed = features
    if arm == "logistic":
        mean = features[indices].mean(axis=0).astype(np.float32)
        scale = features[indices].std(axis=0).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        transformed = ((features - mean) / scale).astype(np.float32)
    connector_model = _fit_binary_model(
        x=transformed[indices], y=connector_target[indices], seed=seed, **kwargs,
    )
    subtype_indices = indices[subtype_trusted[indices].astype(bool)]
    subtype_model = _fit_binary_model(
        x=transformed[subtype_indices], y=subtype_target[subtype_indices], seed=seed + 1, **kwargs,
    )
    other_indices = indices[other_trusted[indices].astype(bool)]
    other_model = _fit_binary_model(
        x=transformed[other_indices], y=other_target[other_indices], seed=seed + 2, **kwargs,
    )
    return ConnectorBundle(arm, connector_model, subtype_model, other_model, mean, scale)


def fit_connector_oof(
    features: np.ndarray, roles: np.ndarray, trusted: np.ndarray, folds: np.ndarray,
    *, seed: int, c_grid: Sequence[float] = (0.1, 1.0, 10.0),
    connector_target: np.ndarray | None = None,
    connector_trusted: np.ndarray | None = None,
    subtype_target: np.ndarray | None = None,
    subtype_trusted: np.ndarray | None = None,
    other_target: np.ndarray | None = None,
    other_trusted: np.ndarray | None = None,
) -> tuple[np.ndarray, list[ConnectorBundle], list[dict[str, Any]]]:
    if not (features.shape[0] == roles.shape[0] == trusted.shape[0] == folds.shape[0]):
        raise ValueError("connector arrays are not aligned")
    n_folds = int(folds.max()) + 1
    probability = np.full((len(roles), len(CONNECTOR_PROBABILITY_COLUMNS)), np.nan, dtype=np.float32)
    derived_connector, derived_subtype, derived_subtype_mask, derived_other = _connector_targets(roles)
    base_trusted = trusted.astype(bool) & (roles != ROLE_TO_ID["UNKNOWN"])
    connector_target = derived_connector if connector_target is None else connector_target
    connector_trusted = base_trusted if connector_trusted is None else connector_trusted.astype(bool)
    subtype_target = derived_subtype if subtype_target is None else subtype_target
    subtype_trusted = (base_trusted & derived_subtype_mask) if subtype_trusted is None else subtype_trusted.astype(bool)
    other_target = derived_other if other_target is None else other_target
    other_trusted = base_trusted if other_trusted is None else other_trusted.astype(bool)
    for name, value in (
        ("connector_target", connector_target), ("connector_trusted", connector_trusted),
        ("subtype_target", subtype_target), ("subtype_trusted", subtype_trusted),
        ("other_target", other_target), ("other_trusted", other_trusted),
    ):
        if value.shape != roles.shape:
            raise ValueError(f"{name} is not aligned")
    bundles: list[ConnectorBundle] = []
    reports: list[dict[str, Any]] = []
    arms = [
        *(dict(arm="logistic", c_value=float(c)) for c in c_grid),
        *(dict(arm="hist", max_depth=depth, min_samples_leaf=leaf) for depth in (2, 3) for leaf in (20, 40)),
    ]
    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        inner_fit = np.flatnonzero(connector_trusted & (folds != outer) & (folds != inner))
        inner_holdout = np.flatnonzero(connector_trusted & (folds == inner))
        candidates = []
        for offset, config in enumerate(arms):
            bundle = _fit_connector_bundle(
                features, connector_target, subtype_target, subtype_trusted, other_target,
                other_trusted, inner_fit, seed=seed + outer * 100 + offset, **config,
            )
            prediction = bundle.predict(features[inner_holdout])
            subtype_rows = subtype_trusted[inner_holdout].astype(bool)
            other_rows = other_trusted[inner_holdout].astype(bool)
            connector_metrics = _binary_metrics(
                connector_target[inner_holdout], prediction[:, 0],
            )
            subtype_metrics = _binary_metrics(
                subtype_target[inner_holdout][subtype_rows],
                _conditional_continuation(prediction)[subtype_rows],
            )
            other_metrics = _binary_metrics(
                other_target[inner_holdout][other_rows], prediction[other_rows, 3],
            )
            candidates.append({
                **config,
                "connector_pr_auc": connector_metrics["pr_auc"],
                "connector_brier": connector_metrics["brier"],
                "subtype_pr_auc": subtype_metrics["pr_auc"],
                "subtype_brier": subtype_metrics["brier"],
                "other_pr_auc": other_metrics["pr_auc"],
                "other_brier": other_metrics["brier"],
                "mean_pr_auc": float(np.mean([
                    connector_metrics["pr_auc"], subtype_metrics["pr_auc"],
                    other_metrics["pr_auc"],
                ])),
            })
        selected = max(candidates, key=lambda row: (
            row["mean_pr_auc"], row["connector_pr_auc"], row["subtype_pr_auc"],
            row["arm"] == "logistic",
        ))
        config = {key: selected[key] for key in ("arm", "c_value", "max_depth", "min_samples_leaf") if key in selected}
        fit = np.flatnonzero(connector_trusted & (folds != outer))
        holdout = np.flatnonzero(folds == outer)
        bundle = _fit_connector_bundle(
            features, connector_target, subtype_target, subtype_trusted, other_target,
            other_trusted, fit, seed=seed + outer, **config,
        )
        probability[holdout] = bundle.predict(features[holdout])
        bundles.append(bundle)
        reports.append({
            "outer_fold": outer, "inner_fold": inner, "selected": config,
            "fit_labelled_candidates": len(fit), "holdout_candidates": len(holdout),
            "candidates": candidates,
        })
    if not np.isfinite(probability).all():
        raise RuntimeError("connector OOF probabilities are incomplete")
    return probability, bundles, reports


def _load_texts(path: Path) -> list[str]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
                raise ValueError(f"{path}:{number}: malformed text row")
            result.append(str(value["text"]))
    return result


def _load_table(root: Path) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "features", "roles", "trusted", "folds", "row_indices",
            "connector_targets", "connector_trusted", "subtype_targets",
            "subtype_trusted", "other_targets", "other_trusted",
        )
        if (root / f"{name}.npy").exists()
    }
    return manifest, arrays


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root, output = Path(args.table_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest, arrays = _load_table(table_root)
    models = output / "models"
    models.mkdir()
    if args.kind == "heading":
        texts = _load_texts(table_root / "texts.jsonl")
        probability, bundles, reports = fit_heading_oof(
            texts, arrays["features"], arrays["roles"], arrays["trusted"], arrays["folds"],
            c_grid=(0.1, 1.0, 10.0), seed=args.seed,
        )
        schema, columns = HEADING_SCHEMA, HEADING_PROBABILITY_COLUMNS
        labelled = arrays["trusted"].astype(bool) & (arrays["roles"] != ROLE_TO_ID["UNKNOWN"])
        heading_ids = np.asarray([ROLE_TO_ID[name] for name in HEADING_TYPES])
        any_truth = np.isin(arrays["roles"][labelled], heading_ids)
        heading_rows = np.flatnonzero(labelled & np.isin(arrays["roles"], heading_ids))
        type_truth = np.asarray([
            HEADING_TYPE_TO_INDEX[ID_TO_ROLE[int(role)]] for role in arrays["roles"][heading_rows]
        ])
        oof_metrics = {
            "any_header": _binary_metrics(any_truth.astype(np.uint8), probability[labelled, 0]),
            "heading_type_macro_pr_auc": _macro_ap(
                type_truth, probability[heading_rows, 1:], range(len(HEADING_TYPES)),
            ),
            "trusted_candidate_count": int(np.count_nonzero(labelled)),
        }
    else:
        probability, bundles, reports = fit_connector_oof(
            arrays["features"], arrays["roles"], arrays["trusted"], arrays["folds"], seed=args.seed,
            connector_target=arrays.get("connector_targets"),
            connector_trusted=arrays.get("connector_trusted"),
            subtype_target=arrays.get("subtype_targets"),
            subtype_trusted=arrays.get("subtype_trusted"),
            other_target=arrays.get("other_targets"),
            other_trusted=arrays.get("other_trusted"),
        )
        schema, columns = CONNECTOR_SCHEMA, CONNECTOR_PROBABILITY_COLUMNS
        connector_trusted = arrays["connector_trusted"].astype(bool)
        subtype_trusted = arrays["subtype_trusted"].astype(bool)
        other_trusted = arrays["other_trusted"].astype(bool)
        oof_metrics = {
            "connector": _binary_metrics(
                arrays["connector_targets"][connector_trusted], probability[connector_trusted, 0],
            ),
            "continuation_given_connector": _binary_metrics(
                arrays["subtype_targets"][subtype_trusted],
                _conditional_continuation(probability)[subtype_trusted],
            ),
            "other": _binary_metrics(
                arrays["other_targets"][other_trusted], probability[other_trusted, 3],
            ),
            "trusted_connector_head_count": int(np.count_nonzero(connector_trusted)),
            "trusted_subtype_head_count": int(np.count_nonzero(subtype_trusted)),
            "trusted_other_head_count": int(np.count_nonzero(other_trusted)),
        }
    for fold, bundle in enumerate(bundles):
        with (models / f"fold{fold}.pkl").open("xb") as handle:
            pickle.dump(bundle, handle, protocol=5)
    _save_array(output / "oof_probability.npy", probability)
    _save_array(output / "row_indices.npy", np.asarray(arrays["row_indices"]))
    report = {
        "schema_version": schema, "status": "passed_grouped_oof_expert_training",
        "kind": args.kind, "code_commit": args.code_commit, "slurm_job_id": args.slurm_job_id,
        "validation_opened": False, "probability_columns": columns,
        "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
        "folds": reports, "oof_metrics": oof_metrics,
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(output / "receipt.json", {
        **report,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(output.iterdir()) if path.is_file()
        },
    })
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("heading", "connector"), required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

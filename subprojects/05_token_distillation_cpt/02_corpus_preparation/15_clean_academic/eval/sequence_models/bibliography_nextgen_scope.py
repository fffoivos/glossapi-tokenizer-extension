#!/usr/bin/env python3
"""Train a grouped OOF scope classifier over decoded bibliography components."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_nextgen_decode import evaluate
from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-component-scope-oof-v1"
SIGNALS = (
    "probability:entry",
    "probability:signal_tcn",
    "probability:continuation_specialist",
    "probability:continuation",
    "probability:filler",
    "probability:bib_header",
    "probability:bib_subheader",
    "probability:non_bib_header",
    "presence:numbered_entry_count",
    "presence:year_count",
    "presence:url_count",
    "presence:doi_count",
    "presence:page_range_count",
    "presence:prose_lead_count",
    "structure:markdown_heading",
    "structure:table_row",
)
THRESHOLDS = (
    0.02,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.70,
    0.85,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
)


def _sklearn() -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    return {
        "hist": HistGradientBoostingClassifier,
        "linear": LogisticRegression,
        "scaler": StandardScaler,
        "version": sklearn.__version__,
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def component_feature_names() -> tuple[str, ...]:
    result = [
        "component:relative_start",
        "component:relative_end",
        "component:relative_middle",
        "component:relative_remaining",
        "component:line_count",
        "component:log1p_line_count",
        "component:char_count",
        "component:mean_char_count",
        "component:ordinal_fraction",
        "component:component_count",
        "component:previous_gap",
        "component:following_gap",
        "line_probability:min",
        "line_probability:q10",
        "line_probability:q25",
        "line_probability:mean",
        "line_probability:median",
        "line_probability:q75",
        "line_probability:q90",
        "line_probability:max",
        "line_probability:fraction_ge_0_5",
        "line_probability:fraction_ge_0_75",
        "line_probability:fraction_ge_0_9",
    ]
    for signal in SIGNALS:
        result.extend(
            (
                f"signal:{signal}:mean",
                f"signal:{signal}:max",
                f"signal:{signal}:active_fraction",
            )
        )
    for side in ("before", "after"):
        result.extend(
            (
                f"neighbor:{side}:line_probability:max10",
                f"neighbor:{side}:line_probability:mean10",
                f"neighbor:{side}:bib_header:max10",
                f"neighbor:{side}:non_bib_header:max10",
                f"neighbor:{side}:markdown_present10",
            )
        )
    return tuple(result)


def _summary(values: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90))
    return (
        float(values.min()),
        float(quantiles[0]),
        float(quantiles[1]),
        float(values.mean()),
        float(quantiles[2]),
        float(quantiles[3]),
        float(quantiles[4]),
        float(values.max()),
        float(np.mean(values >= 0.50)),
        float(np.mean(values >= 0.75)),
        float(np.mean(values >= 0.90)),
    )


def build_component_table(
    table: Any,
    prediction: np.ndarray,
    probability: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return component features, document indices, inclusive bounds, and targets."""

    if not (
        prediction.shape == probability.shape == (len(table.targets),)
        and features.shape[0] == len(prediction)
    ):
        raise ValueError("component inputs are not line aligned")
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    missing = set(SIGNALS) - set(name_to_index)
    if missing:
        raise ValueError(f"component table lacks signals: {sorted(missing)}")
    rows: list[list[float]] = []
    document_indices: list[int] = []
    bounds: list[tuple[int, int]] = []
    targets: list[bool] = []
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    for document_index, document in enumerate(table.documents):
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        doc_length = doc_end - doc_start
        local_blocks = blocks_from_mask(
            prediction[doc_start:doc_end], table.abs_indices[doc_start:doc_end]
        )
        for ordinal, (local_start, local_end) in enumerate(local_blocks):
            start, end = doc_start + local_start, doc_start + local_end
            length = end - start + 1
            previous_end = local_blocks[ordinal - 1][1] if ordinal else -1
            next_start = (
                local_blocks[ordinal + 1][0]
                if ordinal + 1 < len(local_blocks)
                else doc_length
            )
            values = [
                local_start / max(1, doc_length - 1),
                local_end / max(1, doc_length - 1),
                ((local_start + local_end) / 2) / max(1, doc_length - 1),
                (doc_length - local_end - 1) / max(1, doc_length - 1),
                float(length),
                math.log1p(length),
                float(table.char_lengths[start : end + 1].astype(np.int64).sum()),
                float(table.char_lengths[start : end + 1].mean()),
                ordinal / max(1, len(local_blocks) - 1),
                float(len(local_blocks)),
                float(local_start - previous_end - 1),
                float(next_start - local_end - 1),
                *_summary(probability[start : end + 1]),
            ]
            for signal in SIGNALS:
                local = features[start : end + 1, name_to_index[signal]]
                active_threshold = 0.5 if signal.startswith("probability:") else 0.0
                values.extend(
                    (
                        float(local.mean()),
                        float(local.max(initial=0.0)),
                        float(np.mean(local > active_threshold)),
                    )
                )
            for neighbor_start, neighbor_end in (
                (max(doc_start, start - 10), start),
                (end + 1, min(doc_end, end + 11)),
            ):
                neighbor_probability = probability[neighbor_start:neighbor_end]
                neighbor_features = features[neighbor_start:neighbor_end]
                values.extend(
                    (
                        float(neighbor_probability.max(initial=0.0)),
                        float(neighbor_probability.mean())
                        if len(neighbor_probability)
                        else 0.0,
                        float(
                            neighbor_features[
                                :, name_to_index["probability:bib_header"]
                            ].max(initial=0.0)
                        ),
                        float(
                            neighbor_features[
                                :, name_to_index["probability:non_bib_header"]
                            ].max(initial=0.0)
                        ),
                        float(
                            neighbor_features[
                                :, name_to_index["structure:markdown_heading"]
                            ].max(initial=0.0)
                        ),
                    )
                )
            rows.append(values)
            document_indices.append(document_index)
            bounds.append((start, end))
            targets.append(bool(gold[start : end + 1].any()))
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(rows), len(component_feature_names())):
        raise RuntimeError("component feature schema mismatch")
    if not np.isfinite(matrix).all():
        raise RuntimeError("component features contain non-finite values")
    return (
        matrix,
        np.asarray(document_indices, dtype=np.uint32),
        np.asarray(bounds, dtype=np.uint32),
        np.asarray(targets, dtype=bool),
    )


def _weights(target: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(target), dtype=np.float64)
    if mode != "balanced":
        raise ValueError(f"unknown weighting: {mode}")
    counts = np.bincount(target.astype(np.uint8), minlength=2)
    if np.any(counts == 0):
        raise ValueError("component fit requires both classes")
    return np.asarray([len(target) / (2 * counts[int(value)]) for value in target])


def _fit(kind: str, x: np.ndarray, y: np.ndarray, weighting: str, seed: int) -> dict[str, Any]:
    tools = _sklearn()
    weights = _weights(y, weighting)
    if kind == "hist":
        model = tools["hist"](
            learning_rate=0.05,
            max_iter=200,
            max_depth=3,
            min_samples_leaf=12,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=seed,
        ).fit(x, y, sample_weight=weights)
        return {"kind": kind, "scaler": None, "model": model}
    if kind == "linear":
        scaler = tools["scaler"]().fit(x, sample_weight=weights)
        model = tools["linear"](
            C=0.1, solver="lbfgs", max_iter=500, random_state=seed
        ).fit(scaler.transform(x), y, sample_weight=weights)
        return {"kind": kind, "scaler": scaler, "model": model}
    raise ValueError(f"unknown component model: {kind}")


def predict_component_probability(models: Sequence[Mapping[str, Any]], x: np.ndarray) -> np.ndarray:
    result = np.zeros(len(x), dtype=np.float64)
    for bundle in models:
        values = bundle["scaler"].transform(x) if bundle["scaler"] is not None else x
        result += bundle["model"].predict_proba(values)[:, 1]
    return (result / len(models)).astype(np.float32)


def apply_component_threshold(
    prediction: np.ndarray,
    bounds: np.ndarray,
    component_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    result = prediction.copy()
    for (start, end), value in zip(bounds, component_probability, strict=True):
        if value < threshold:
            result[int(start) : int(end) + 1] = False
    return result


def _eligible(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics["line_precision"] >= 0.98
        and metrics["char_precision"] >= 0.98
        and metrics["non_bib_markdown_heading_crossings"] == 0
        and metrics["spurious_blocks_per_zero_block_document"] <= 0.02
    )


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        min(metrics["line_recall"], metrics["char_recall"]),
        metrics["char_recall"],
        metrics["line_recall"],
        metrics["line_precision"],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_root = Path(args.base_table_dir).resolve()
    table_root = Path(args.table_dir).resolve()
    model_root = Path(args.model_dir).resolve()
    decoder_root = Path(args.decoder_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    model_report = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
    decoder_report = json.loads((decoder_root / "report.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TABLE_SCHEMA or model_report.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("unsupported component-scope inputs")
    selected_decoder = decoder_report.get("deployment_near_miss") or decoder_report.get("selected")
    if not isinstance(selected_decoder, Mapping):
        raise ValueError("decoder has no frozen development candidate")
    prediction_path = decoder_root / "deployment_near_miss_oof_prediction.npy"
    if not prediction_path.exists():
        prediction_path = decoder_root / "selected_oof_prediction.npy"
    table = load_table(base_root, expected_split="train")
    features = np.load(table_root / "features.npy", mmap_mode="r", allow_pickle=False)
    probability = np.load(model_root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False).astype(bool)
    x, component_documents, bounds, target = build_component_table(
        table, prediction, probability, features, manifest["feature_names"]
    )
    component_folds = table.folds[bounds[:, 0].astype(np.int64)].astype(np.uint8)
    output.mkdir(parents=True)
    (output / "models").mkdir()
    oof = np.full(len(x), np.nan, dtype=np.float32)
    folds = []
    for fold in range(int(table.manifest["n_folds"])):
        fit = component_folds != fold
        holdout = component_folds == fold
        bundle = _fit(args.kind, x[fit], target[fit], args.weighting, int(args.seed) + fold)
        oof[holdout] = predict_component_probability((bundle,), x[holdout])
        with (output / "models" / f"fold{fold}.pkl").open("xb") as handle:
            pickle.dump(bundle, handle, protocol=5)
        folds.append(
            {
                "fold": fold,
                "fit_component_count": int(np.count_nonzero(fit)),
                "holdout_component_count": int(np.count_nonzero(holdout)),
                "fit_positive_count": int(np.count_nonzero(target[fit])),
                "holdout_positive_count": int(np.count_nonzero(target[holdout])),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("component OOF predictions are incomplete")
    rows = []
    for threshold in THRESHOLDS:
        scoped = apply_component_threshold(prediction, bounds, oof, threshold)
        rows.append(
            {
                "threshold": threshold,
                "kept_component_count": int(np.count_nonzero(oof >= threshold)),
                "metrics": evaluate(table, scoped, features, manifest["feature_names"]),
            }
        )
    eligible = [row for row in rows if _eligible(row["metrics"])]
    selected = max(eligible, key=_selection_key) if eligible else None
    near = [
        row
        for row in rows
        if row["metrics"]["line_precision"] >= 0.98
        and row["metrics"]["char_precision"] >= 0.98
        and row["metrics"]["non_bib_markdown_heading_crossings"] == 0
    ]
    near_miss = max(near, key=_selection_key) if near else None
    with (output / "component_features.npy").open("xb") as handle:
        np.save(handle, x, allow_pickle=False)
    with (output / "component_bounds.npy").open("xb") as handle:
        np.save(handle, bounds, allow_pickle=False)
    with (output / "component_oof_probability.npy").open("xb") as handle:
        np.save(handle, oof, allow_pickle=False)
    with (output / "component_target.npy").open("xb") as handle:
        np.save(handle, target, allow_pickle=False)
    chosen = selected or near_miss
    if chosen is not None:
        scoped = apply_component_threshold(prediction, bounds, oof, chosen["threshold"])
        with (output / "selected_oof_prediction.npy").open("xb") as handle:
            np.save(handle, scoped, allow_pickle=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_component_scope_gate" if selected else "research_only_component_scope",
        "test_opened": False,
        "validation_opened": False,
        "kind": args.kind,
        "weighting": args.weighting,
        "feature_names": component_feature_names(),
        "component_count": len(x),
        "positive_component_count": int(np.count_nonzero(target)),
        "negative_component_count": int(np.count_nonzero(~target)),
        "folds": folds,
        "threshold_candidates": rows,
        "selected": selected,
        "deployment_near_miss": near_miss,
        "decoder_config": selected_decoder["config"],
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "sklearn_version": _sklearn()["version"],
        "inputs": {
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
            "line_model_receipt_sha256": sha256_file(model_root / "receipt.json"),
            "decoder_receipt_sha256": sha256_file(decoder_root / "receipt.json"),
            "line_prediction_sha256": sha256_file(prediction_path),
        },
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(
        output / "receipt.json",
        {
            **report,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=("hist", "linear"), required=True)
    parser.add_argument("--weighting", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

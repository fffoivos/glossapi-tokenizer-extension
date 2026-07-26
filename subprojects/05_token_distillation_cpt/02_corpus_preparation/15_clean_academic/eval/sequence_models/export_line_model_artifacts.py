#!/usr/bin/env python3
"""Export the deployed bibliography line-model stack to a language-neutral format.

The Rust port must reproduce the Python line mask at threshold 0.9
(decision-equivalent). To do that without reimplementing scikit-learn it needs
the fitted parameters, not the pickles. This writes one directory holding every
model in the chain as JSON + .npy, with a receipt binding it to the source
artifacts.

The chain (v3, no citation-grammar features):

    P0D entry      HistGradientBoosting x5 folds  -> probability:entry
    signal TCN     torch SignalTCN x5 folds       -> probability:signal_tcn
    heading bundle TF-IDF(char_wb 2-5 + word 1-2) + LogisticRegression x2
                                                  -> bib_header / bib_subheader / non_bib_header
    connector      StandardScaler + 3 binary models
                                                  -> connector / continuation / filler / other
    line model     HistGradientBoosting x5 folds  -> the line probability we threshold

Run on Clariden inside the pinned runtime (scikit-learn 1.9.0 is asserted by the
source modules; unpickling requires it).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "bibliography-line-model-export-v1"


# --------------------------------------------------------------------------
# tree ensembles
# --------------------------------------------------------------------------
def export_hist_gradient_boosting(model: Any) -> dict[str, Any]:
    """Flatten a fitted HistGradientBoostingClassifier into plain arrays.

    sklearn stores each boosting iteration as a TreePredictor whose `nodes` is a
    structured array. Everything needed to evaluate is in there: the split
    feature, the threshold, the child indices, the leaf value, and the
    missing-value direction. Binning is a *training* concern -- prediction
    compares the raw feature against `num_threshold` -- so a port does not need
    the binner.
    """
    trees: list[dict[str, Any]] = []
    for iteration in model._predictors:
        for predictor in iteration:
            nodes = predictor.nodes
            trees.append({
                "is_leaf": nodes["is_leaf"].astype(np.uint8).tolist(),
                "feature_idx": nodes["feature_idx"].astype(np.int32).tolist(),
                "num_threshold": nodes["num_threshold"].astype(np.float64).tolist(),
                "left": nodes["left"].astype(np.int32).tolist(),
                "right": nodes["right"].astype(np.int32).tolist(),
                "value": nodes["value"].astype(np.float64).tolist(),
                "missing_go_to_left": nodes["missing_go_to_left"].astype(np.uint8).tolist(),
            })
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).ravel()
    return {
        "kind": "hist_gradient_boosting",
        "n_trees": len(trees),
        "n_features": int(model.n_features_in_),
        "baseline_prediction": baseline.tolist(),
        "classes": np.asarray(model.classes_).tolist(),
        # binary HistGB emits a single raw score; probability = sigmoid(baseline + sum(leaves))
        "link": "logistic",
        "trees": trees,
    }


# --------------------------------------------------------------------------
# linear models
# --------------------------------------------------------------------------
def export_linear(model: Any) -> dict[str, Any]:
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return {
        "kind": "linear",
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "classes": np.asarray(model.classes_).tolist(),
        "n_features": int(coef.shape[1]),
        # sklearn LogisticRegression: binary -> sigmoid, multiclass -> softmax
        "link": "logistic" if coef.shape[0] == 1 else "softmax",
    }


def export_any_model(model: Any) -> dict[str, Any]:
    name = type(model).__name__
    if name == "HistGradientBoostingClassifier":
        return export_hist_gradient_boosting(model)
    if hasattr(model, "coef_"):
        return export_linear(model)
    raise TypeError(f"unsupported model type for export: {name}")


# --------------------------------------------------------------------------
# tf-idf
# --------------------------------------------------------------------------
def export_tfidf(vec: Any) -> dict[str, Any]:
    """Export a fitted TfidfVectorizer.

    The Rust side must reproduce, in order: analyzer -> raw term counts;
    sublinear_tf (1 + ln(tf)); multiply by idf; L2-normalise. `idf_` already
    contains the smooth-idf constant, so the port only needs the vocabulary and
    that vector.
    """
    vocab = {str(term): int(index) for term, index in vec.vocabulary_.items()}
    idf = np.asarray(vec.idf_, dtype=np.float64)
    return {
        "kind": "tfidf",
        "analyzer": vec.analyzer,
        "ngram_range": list(vec.ngram_range),
        "lowercase": bool(vec.lowercase),
        "sublinear_tf": bool(vec.sublinear_tf),
        "norm": vec.norm,
        "smooth_idf": bool(vec.smooth_idf),
        "sublinear_note": "tf -> 1 + ln(tf) when tf > 0",
        "token_pattern": getattr(vec, "token_pattern", None),
        "strip_accents": vec.strip_accents,
        "n_features": len(vocab),
        "vocabulary": vocab,
        "idf": idf.tolist(),
    }


def export_scaler(scaler: Any) -> dict[str, Any]:
    if scaler is None:
        return {"kind": "identity"}
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    return {
        "kind": "standard_scaler",
        "with_mean": mean is not None,
        "mean": np.asarray(mean, dtype=np.float64).tolist() if mean is not None else None,
        "scale": np.asarray(scale, dtype=np.float64).tolist(),
    }


# --------------------------------------------------------------------------
# torch TCN
# --------------------------------------------------------------------------
def export_tcn(path: Path) -> dict[str, Any]:
    """Flatten one SignalTCN fold.

    The checkpoint carries no normalisation statistics -- the ten inputs are a
    probability and nine indicator flags, all already on [0, 1], so the model
    consumes them raw. An earlier version of this exporter assumed a mean/scale
    pair and failed loudly on the real file, which is the behaviour wanted.

    Architecture (`bibliography_signal_tcn.SignalTCN`), recorded here because the
    port reimplements the forward pass rather than loading torch:

        input_projection : Linear(input_dim -> hidden_dim), then * mask
        blocks[i]        : LayerNorm -> Conv1d(hidden, hidden, kernel_size=3,
                           dilation=d_i, padding=d_i) -> GELU -> + residual, * mask
        output_norm      : LayerNorm(hidden_dim)
        output           : Linear(hidden_dim -> 1), squeezed

    Dropout is identity at inference. Padding equals dilation with kernel_size 3,
    so each block is symmetric and length-preserving.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {
        key: np.asarray(value.numpy(), dtype=np.float64).tolist()
        for key, value in checkpoint["state_dict"].items()
    }
    architecture = dict(checkpoint["architecture"])
    return {
        "kind": "signal_tcn",
        "schema_version": checkpoint.get("schema_version"),
        "fold": int(checkpoint.get("fold", -1)),
        "feature_names": list(checkpoint["feature_names"]),
        "architecture": {
            "hidden_dim": int(architecture["hidden_dim"]),
            "dilations": [int(d) for d in architecture["dilations"]],
            # Retained for provenance only; inference does not apply dropout.
            "dropout": float(architecture["dropout"]),
            "kernel_size": 3,
        },
        "state_dict": tensors,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _load_pickle(path: Path) -> Any:
    """Rebind the three role-expert dataclasses that were pickled from __main__."""
    import __main__

    from .bibliography_role_experts import ConnectorBundle, HeadingBundle, HeadingTransform

    __main__.HeadingTransform = HeadingTransform
    __main__.HeadingBundle = HeadingBundle
    __main__.ConnectorBundle = ConnectorBundle
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    inputs: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "line_threshold": float(args.threshold),
        "feature_schema": args.feature_schema,
        "stages": {},
    }

    def dump(name: str, payload: Mapping[str, Any]) -> None:
        target = out / f"{name}.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        manifest["stages"][name] = {"file": target.name, "bytes": target.stat().st_size}
        print(f"  wrote {target.name} ({target.stat().st_size/1e6:.1f} MB)")

    # ---- P0D entry (HistGB folds) ----
    p0d = sorted(Path(args.entry_model_dir).glob("models/P0D.fold*.pkl"))
    if not p0d:
        raise ValueError("no P0D folds")
    dump("entry_p0d", {"folds": [export_any_model(_load_pickle(p)) for p in p0d]})
    inputs["entry_p0d"] = [_sha256(p) for p in p0d]

    # ---- line model (HistGB folds, scaler is None for v3) ----
    line = sorted(Path(args.line_model_dir).glob("models/fold*.pkl"))
    if not line:
        raise ValueError("no line-model folds")
    line_folds = []
    for path in line:
        bundle = _load_pickle(path)
        line_folds.append({
            "scaler": export_scaler(bundle.get("scaler")),
            "model": export_any_model(bundle["model"]),
        })
    dump("line_model", {"folds": line_folds})
    inputs["line_model"] = [_sha256(p) for p in line]

    # ---- heading bundle ----
    heading = sorted(Path(args.heading_model_dir).glob("models/*.pkl"))
    heading_folds = []
    for path in heading:
        bundle = _load_pickle(path)
        heading_folds.append({
            "char_tfidf": export_tfidf(bundle.transform.char_vectorizer),
            "word_tfidf": export_tfidf(bundle.transform.word_vectorizer),
            "numeric_scaler": export_scaler(bundle.transform.numeric_scaler),
            "any_model": export_any_model(bundle.any_model),
            "type_model": export_any_model(bundle.type_model),
        })
    dump("heading_bundle", {"folds": heading_folds})
    inputs["heading_bundle"] = [_sha256(p) for p in heading]

    # ---- connector bundle ----
    connector = sorted(Path(args.connector_model_dir).glob("models/*.pkl"))
    connector_folds = []
    for path in connector:
        bundle = _load_pickle(path)
        connector_folds.append({
            "arm": bundle.arm,
            "mean": np.asarray(bundle.mean, dtype=np.float64).tolist() if bundle.mean is not None else None,
            "scale": np.asarray(bundle.scale, dtype=np.float64).tolist() if bundle.scale is not None else None,
            "connector_model": export_any_model(bundle.connector_model),
            "subtype_model": export_any_model(bundle.subtype_model),
            "other_model": export_any_model(bundle.other_model),
        })
    dump("connector_bundle", {"folds": connector_folds})
    inputs["connector_bundle"] = [_sha256(p) for p in connector]

    # ---- signal TCN ----
    tcn = sorted(Path(args.signal_model_dir).glob("models/*.pt"))
    if tcn:
        dump("signal_tcn", {"folds": [export_tcn(p) for p in tcn]})
        inputs["signal_tcn"] = [_sha256(p) for p in tcn]

    manifest["inputs_sha256"] = inputs
    manifest["code_commit"] = args.code_commit
    with (out / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nexported to {out}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-model-dir", required=True)
    parser.add_argument("--signal-model-dir", required=True)
    parser.add_argument("--heading-model-dir", required=True)
    parser.add_argument("--connector-model-dir", required=True)
    parser.add_argument("--line-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--feature-schema", default="bibliography-nextgen-full-table-v3")
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

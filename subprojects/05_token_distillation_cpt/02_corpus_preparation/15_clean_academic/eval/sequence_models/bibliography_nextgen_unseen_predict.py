#!/usr/bin/env python3
"""Generate label-blind predictions for frozen nextgen candidates."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_nextgen_decode import DecoderConfig, decode_table
from .bibliography_nextgen_freeze import SCHEMA_VERSION as FREEZE_SCHEMA
from .bibliography_nextgen_models import build_context_features
from .bibliography_nextgen_scope import (
    apply_component_threshold,
    build_component_table,
    predict_component_probability,
)
from .bibliography_nextgen_table import SCHEMA_VERSION as TABLE_SCHEMA
from .bibliography_nextgen_tcn import FeatureTCN
from .bibliography_nextgen_unseen_features import UnseenTable
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-label-blind-test-prediction-v1"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _physical_segments(abs_indices: np.ndarray) -> list[tuple[int, int]]:
    starts = [0]
    starts.extend(
        (
            np.flatnonzero(np.diff(abs_indices.astype(np.int64)) > MAX_PHYSICAL_GAP)
            + 1
        ).tolist()
    )
    starts.append(len(abs_indices))
    return list(zip(starts[:-1], starts[1:], strict=True))


def _sklearn_probability(model_root: Path, context: np.ndarray) -> np.ndarray:
    paths = sorted((model_root / "models").glob("fold*.pkl"))
    if not paths:
        raise ValueError(f"no sklearn folds: {model_root}")
    result = np.zeros(len(context), dtype=np.float64)
    for path in paths:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, dict) or set(bundle) != {"kind", "scaler", "model"}:
            raise ValueError(f"unexpected nextgen sklearn bundle: {path}")
        for start in range(0, len(context), 50_000):
            values = context[start : start + 50_000]
            if bundle["scaler"] is not None:
                values = bundle["scaler"].transform(values).astype(np.float32)
            result[start : start + 50_000] += bundle["model"].predict_proba(values)[:, 1]
    return (result / len(paths)).astype(np.float32)


def _tcn_probability(
    model_root: Path,
    features: np.ndarray,
    table: UnseenTable,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for nextgen TCN inference")
    paths = sorted((model_root / "models").glob("fold*.pt"))
    if not paths:
        raise ValueError(f"no TCN folds: {model_root}")
    result = np.zeros(len(features), dtype=np.float64)
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        architecture = checkpoint["architecture"]
        model = FeatureTCN(
            features.shape[1],
            width=int(architecture["width"]),
            dilations=tuple(architecture["dilations"]),
            dropout=float(architecture["dropout"]),
        ).cpu()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean = checkpoint["mean"].numpy()
        scale = checkpoint["scale"].numpy()
        local = np.zeros(len(features), dtype=np.float32)
        with torch.inference_mode():
            for document in table.documents:
                doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
                local_abs = table.abs_indices[doc_start:doc_end]
                for segment_start, segment_end in _physical_segments(local_abs):
                    for central_start in range(segment_start, segment_end, 256):
                        central_end = min(segment_end, central_start + 256)
                        input_start = max(segment_start, central_start - 32)
                        input_end = min(segment_end, central_end + 32)
                        values = (
                            features[doc_start + input_start : doc_start + input_end]
                            - mean
                        ) / scale
                        x = torch.from_numpy(values).unsqueeze(0)
                        mask = torch.ones((1, len(values)), dtype=torch.bool)
                        probability = torch.sigmoid(model(x, mask))[0].numpy()
                        left, right = central_start - input_start, central_end - input_start
                        local[
                            doc_start + central_start : doc_start + central_end
                        ] = probability[left:right]
        result += local
    return (result / len(paths)).astype(np.float32)


def _model_probability(
    model_root: Path,
    features: np.ndarray,
    context: np.ndarray,
    table: UnseenTable,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    model_root = model_root.resolve()
    if model_root in cache:
        return cache[model_root]
    report = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
    kind = str(report.get("kind"))
    if kind == "linear" or kind == "hist":
        value = _sklearn_probability(model_root, context)
    elif kind == "tcn":
        value = _tcn_probability(model_root, features, table)
    elif kind.startswith("feature_baseline:"):
        feature = str(report["feature"])
        # The feature contract is frozen by the candidate's development report;
        # the unseen table has the identical ordered schema validated by run().
        names = table.feature_names
        value = np.asarray(features[:, names.index(feature)], dtype=np.float32)
    elif report.get("status") == "passed_aligned_development_oof_ensemble":
        value = np.zeros(len(features), dtype=np.float64)
        for member in report["members"]:
            value += float(member["weight"]) * _model_probability(
                Path(member["path"]), features, context, table, cache
            )
        value = value.astype(np.float32)
    else:
        raise ValueError(f"unsupported frozen model kind: {kind}")
    if value.shape != (len(features),) or not np.isfinite(value).all():
        raise RuntimeError(f"unseen probability contract failure: {model_root}")
    cache[model_root] = value
    return value


def _apply_scope_model(
    scope_root: Path,
    frozen_folds: Sequence[Mapping[str, Any]],
    table: UnseenTable,
    prediction: np.ndarray,
    probability: np.ndarray,
    features: np.ndarray,
    names: Sequence[str],
    threshold: float,
    guards: Mapping[str, Any],
) -> tuple[np.ndarray, int, int]:
    paths = [Path(row["path"]).resolve() for row in frozen_folds]
    if not paths:
        raise ValueError(f"no component-scope folds: {scope_root}")
    models = []
    for path, frozen in zip(paths, frozen_folds, strict=True):
        if path.parent != (scope_root / "models").resolve() or sha256_file(path) != frozen[
            "sha256"
        ]:
            raise ValueError(f"frozen component-scope fold drift: {path}")
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, dict) or set(bundle) != {"kind", "scaler", "model"}:
            raise ValueError(f"unexpected component-scope bundle: {path}")
        models.append(bundle)
    x, _documents, bounds, _target = build_component_table(
        table, prediction, probability, features, names
    )
    component_probability = predict_component_probability(models, x)
    scoped = apply_component_threshold(
        prediction,
        bounds,
        component_probability,
        threshold,
        component_features=x,
        heading_rescue_floor=guards.get("heading_rescue_floor"),
        image_fraction_veto=guards.get("image_fraction_veto"),
        rule_fraction_veto=guards.get("rule_fraction_veto"),
    )
    return scoped, len(bounds), int(np.count_nonzero(component_probability >= threshold))


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root = Path(args.feature_table_dir).resolve()
    freeze_path = Path(args.candidate_freeze).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads((table_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != "frozen_before_test_open"
        or freeze.get("test_labels_opened") is not False
        or manifest.get("schema_version") != TABLE_SCHEMA
        or manifest.get("status") != "passed_label_blind_unseen_feature_materialization"
        or manifest.get("test_labels_opened") is not False
        or manifest.get("inputs", {}).get("candidate_freeze_sha256")
        != sha256_file(freeze_path)
    ):
        raise ValueError("unseen feature table is not bound to the frozen candidates")
    features = np.load(table_root / "features.npy", mmap_mode="r", allow_pickle=False)
    abs_indices = np.load(
        table_root / "abs_indices.npy", mmap_mode="r", allow_pickle=False
    )
    char_lengths = np.load(
        table_root / "char_lengths.npy", mmap_mode="r", allow_pickle=False
    )
    auxiliary_scope = np.load(
        table_root / "auxiliary_scope.npy", mmap_mode="r", allow_pickle=False
    ).astype(bool)
    documents = list(_iter_jsonl(table_root / "documents.jsonl"))
    table = UnseenTable(
        documents, abs_indices, char_lengths, tuple(manifest["feature_names"])
    )
    names = tuple(manifest["feature_names"])
    context = build_context_features(np.asarray(features), names, table)
    output.mkdir(parents=True)
    cache: dict[Path, np.ndarray] = {}
    candidates = []
    for candidate in freeze["candidates"]:
        model_root = Path(candidate["model_dir"]).resolve()
        if sha256_file(model_root / "receipt.json") != candidate["model_receipt_sha256"]:
            raise ValueError(f"frozen model receipt drift: {candidate['name']}")
        probability = _model_probability(model_root, features, context, table, cache)
        prediction = decode_table(
            table,
            probability,
            features,
            names,
            DecoderConfig(**candidate["decoder_config"]),
            auxiliary_scope,
        )
        component_count = kept_component_count = None
        if candidate.get("scope_model_dir"):
            scope_root = Path(candidate["scope_model_dir"]).resolve()
            if sha256_file(scope_root / "receipt.json") != candidate[
                "scope_model_receipt_sha256"
            ]:
                raise ValueError(f"frozen scope-model receipt drift: {candidate['name']}")
            prediction, component_count, kept_component_count = _apply_scope_model(
                scope_root,
                candidate["scope_model_folds"],
                table,
                prediction,
                probability,
                features,
                names,
                float(candidate["scope_threshold"]),
                json.loads(
                    (scope_root / "report.json").read_text(encoding="utf-8")
                ),
            )
        probability_path = output / f"{candidate['name']}.probability.npy"
        prediction_path = output / f"{candidate['name']}.prediction.npy"
        with probability_path.open("xb") as handle:
            np.save(handle, probability, allow_pickle=False)
        with prediction_path.open("xb") as handle:
            np.save(handle, prediction, allow_pickle=False)
        candidates.append(
            {
                "name": candidate["name"],
                "model_kind": candidate["model_kind"],
                "predicted_line_count": int(np.count_nonzero(prediction)),
                "component_count_before_scope": component_count,
                "component_count_after_scope": kept_component_count,
                "probability_path": str(probability_path),
                "probability_sha256": sha256_file(probability_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_frozen_label_blind_test_prediction",
        "test_opened": True,
        "test_labels_opened": False,
        "human_gold": False,
        "document_count": len(documents),
        "line_count": len(features),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "candidate_freeze_sha256": sha256_file(freeze_path),
            "feature_receipt_sha256": sha256_file(table_root / "receipt.json"),
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
    parser.add_argument("--feature-table-dir", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Conditional filtered semi-Markov B2 span scorer.

The command exits successfully with a receipt when the frozen B1 error gate
does not require B2.  When required, it generates anchor-filtered candidate
spans, scores them out of fold, selects non-overlapping spans, and applies H0
only after span selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    decode_b0_document,
    evaluate_prediction,
)
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table


SCHEMA_VERSION = "bibliography-entry-b2-oof-v1"
SPAN_FEATURES = (
    "span_line_count",
    "anchor_count",
    "anchor_density",
    "probability_mean",
    "probability_minimum",
    "probability_maximum",
    "probability_q25",
    "probability_median",
    "probability_q75",
    "below_inside_count",
    "long_line_count",
    "header_candidate_count",
    "barrier_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_configs(selected: BlockConfig) -> tuple[BlockConfig, ...]:
    configs = {
        replace(selected, anchor_probability=max(0.40, selected.anchor_probability - 0.20), maximum_bridge_gap=bridge)
        for bridge in (1, 2, 3, 4)
    }
    configs.update(
        {
            replace(selected, anchors_required=2, anchor_window=window, maximum_bridge_gap=bridge)
            for window in (3, 5)
            for bridge in (2, 4)
        }
    )
    return tuple(sorted(configs, key=lambda item: tuple(item.__dict__.values())))


def _span_features(
    probability: np.ndarray,
    char_lengths: np.ndarray,
    header_kinds: np.ndarray,
    start: int,
    end: int,
    config: BlockConfig,
) -> np.ndarray:
    values = probability[start : end + 1]
    lengths = char_lengths[start : end + 1]
    headers = header_kinds[start : end + 1]
    anchors = (values >= config.anchor_probability) & (lengths <= config.seed_length_limit)
    quantiles = np.quantile(values, (0.25, 0.5, 0.75))
    return np.asarray(
        (
            len(values),
            np.count_nonzero(anchors),
            np.mean(anchors),
            np.mean(values),
            np.min(values),
            np.max(values),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            np.count_nonzero(values < config.inside_probability),
            np.count_nonzero(lengths > config.seed_length_limit),
            np.count_nonzero(headers > 0),
            np.count_nonzero(values < 0.05),
        ),
        dtype=np.float32,
    )


def generate_candidates(
    table: Any, probability: np.ndarray, config: BlockConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    document_indices = []
    starts = []
    ends = []
    labels = []
    gold_all = table.original_labels == LABEL_TO_ID["BIB"]
    for document_index, document in enumerate(table.documents):
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        spans: set[tuple[int, int]] = set()
        for candidate_config in _candidate_configs(config):
            mask = decode_b0_document(
                probability[doc_start:doc_end],
                table.char_lengths[doc_start:doc_end],
                table.abs_indices[doc_start:doc_end],
                candidate_config,
            )
            spans.update(blocks_from_mask(mask, table.abs_indices[doc_start:doc_end]))
        gold_spans = blocks_from_mask(gold_all[doc_start:doc_end], table.abs_indices[doc_start:doc_end])
        for start, end in sorted(spans):
            features.append(
                _span_features(
                    probability[doc_start:doc_end],
                    table.char_lengths[doc_start:doc_end],
                    table.header_kinds[doc_start:doc_end],
                    start,
                    end,
                    config,
                )
            )
            document_indices.append(document_index)
            starts.append(start)
            ends.append(end)
            labels.append(int(any(_span_iou((start, end), gold) >= 0.5 for gold in gold_spans)))
    if not features:
        raise ValueError("B2 candidate generator produced no spans")
    return (
        np.stack(features),
        np.asarray(document_indices, dtype=np.uint32),
        np.asarray(starts, dtype=np.uint32),
        np.asarray(ends, dtype=np.uint32),
        np.asarray(labels, dtype=np.uint8),
    )


def _span_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = left[1] - left[0] + right[1] - right[0] + 2 - intersection
    return intersection / union if union else 0.0


def decode_candidates(
    table: Any,
    scores: np.ndarray,
    document_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    probability: np.ndarray,
    config: BlockConfig,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    prediction = np.zeros(len(table.targets), dtype=bool)
    for document_index, document in enumerate(table.documents):
        candidate_rows = np.flatnonzero((document_indices == document_index) & (scores >= threshold))
        chosen: list[tuple[int, int]] = []
        for row in sorted(candidate_rows, key=lambda index: (-float(scores[index]), int(starts[index]), int(ends[index]))):
            span = (int(starts[row]), int(ends[row]))
            if any(_span_iou(span, existing) > 0 for existing in chosen):
                continue
            chosen.append(span)
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local = np.zeros(doc_end - doc_start, dtype=bool)
        for start, end in chosen:
            local[start : end + 1] = True
        local = attach_h0_document(
            local,
            probability[doc_start:doc_end],
            table.header_kinds[doc_start:doc_end],
            table.abs_indices[doc_start:doc_end],
            config,
        )
        prediction[doc_start:doc_end] = local
    return prediction


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError("scikit-learn version differs from the pinned ladder")
    table = load_table(args.table_dir)
    b1_root = Path(args.b1_oof_dir).resolve()
    b1_report = json.loads((b1_root / "b1_oof_report.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    if not bool(b1_report["b2_gate"]["required"]):
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed_skipped_by_predeclared_b1_gate",
            "code_commit": str(args.code_commit),
            "slurm_job_id": str(args.slurm_job_id),
            "b1_oof_report_sha256": _sha256(b1_root / "b1_oof_report.json"),
            "b2_required": False,
            "validation_opened": False,
            "production_eligible": False,
        }
        _write_json(output_dir / "b2_oof_report.json", report)
        _write_json(output_dir / "receipt.json", report)
        return report

    arm = str(b1_report["selected_arm"])
    line_root = Path(args.line_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    block_report = json.loads((block_root / "block_oof_report.json").read_text(encoding="utf-8"))
    config = BlockConfig(**block_report["arms"][arm]["selected_config"])
    probability = np.load(line_root / f"{arm}.oof_probability.npy", mmap_mode="r", allow_pickle=False)
    features, documents, starts, ends, labels = generate_candidates(table, probability, config)
    scores = np.zeros(len(labels), dtype=np.float32)
    fold_rows = []
    models = []
    for fold in range(int(table.manifest["n_folds"])):
        document_folds = np.asarray([int(document["fold"]) for document in table.documents])
        holdout = document_folds[documents] == fold
        fit = ~holdout
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=75,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(args.seed) + fold,
        )
        model.fit(features[fit], labels[fit])
        scores[holdout] = model.predict_proba(features[holdout])[:, 1]
        models.append(model)
        fold_rows.append(
            {
                "fold": fold,
                "fit_candidate_count": int(np.count_nonzero(fit)),
                "holdout_candidate_count": int(np.count_nonzero(holdout)),
            }
        )
    prediction = decode_candidates(
        table,
        scores,
        documents,
        starts,
        ends,
        probability,
        config,
    )
    metrics = evaluate_prediction(table, prediction)
    b1_metrics = b1_report["variants"][b1_report["selected_key"]]["metrics"]
    accepted = (
        metrics["line_precision"] >= b1_metrics["line_precision"]
        and metrics["token_recall"] > b1_metrics["token_recall"]
    )
    _save_array(output_dir / "candidate_features.npy", features)
    _save_array(output_dir / "candidate_document_indices.npy", documents)
    _save_array(output_dir / "candidate_starts.npy", starts)
    _save_array(output_dir / "candidate_ends.npy", ends)
    _save_array(output_dir / "candidate_labels.npy", labels)
    _save_array(output_dir / "candidate_oof_scores.npy", scores)
    _save_array(output_dir / "b2_oof_prediction.npy", prediction)
    for fold, model in enumerate(models):
        with (output_dir / f"b2.fold{fold}.pkl").open("xb") as handle:
            import pickle

            pickle.dump(model, handle, protocol=5)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_b2_validation_unopened",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "b1_oof_report_sha256": _sha256(b1_root / "b1_oof_report.json"),
        "b2_required": True,
        "selected_line_arm": arm,
        "span_feature_names": list(SPAN_FEATURES),
        "candidate_count": len(labels),
        "positive_candidate_count": int(np.count_nonzero(labels)),
        "folds": fold_rows,
        "metrics": metrics,
        "b1_metrics": b1_metrics,
        "accepted_over_b1": accepted,
        "selection_rule": "accept only when line precision does not fall and token recall improves",
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "b2_oof_report.json", report)
    outputs = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--b1-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3141)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

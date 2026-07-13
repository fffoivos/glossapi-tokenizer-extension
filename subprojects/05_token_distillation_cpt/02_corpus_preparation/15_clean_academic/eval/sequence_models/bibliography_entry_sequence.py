#!/usr/bin/env python3
"""Constrained B1 CRF over train-only out-of-fold entry probabilities."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import BlockConfig, attach_h0_document, evaluate_prediction
from .bibliography_entry_dataset import LABEL_TO_ID, MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .feature_crf import LinearChainCRF, train_model
from .features import TAGS, TAG_TO_ID, classes_to_bioes


SCHEMA_VERSION = "bibliography-entry-b1-oof-v1"
FEATURE_NAMES_BASE = (
    "bias",
    "entry_logit",
    "entry_probability",
    "local_mean_probability",
    "local_max_probability",
    "log1p_char_length",
    "over_seed_length_limit",
    "very_low_entry_probability",
    "physical_position",
)


@dataclass
class B1Example:
    document: None
    features: list[dict[int, float]]
    tags: np.ndarray
    line_indices: tuple[int, ...]
    char_lengths: np.ndarray
    token_counts: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_aggregates(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(probability, (1, 1), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3)
    return windows.mean(axis=1), windows.max(axis=1)


def _feature_rows(
    probability: np.ndarray,
    char_lengths: np.ndarray,
    abs_indices: np.ndarray,
    n_physical_lines: int,
    header_kinds: np.ndarray,
    *,
    seed_length_limit: int,
    include_header: bool,
) -> list[dict[int, float]]:
    mean_probability, max_probability = _local_aggregates(probability)
    logits = np.log(np.clip(probability, 1.0e-6, 1 - 1.0e-6) / np.clip(1 - probability, 1.0e-6, 1.0))
    rows = []
    for index in range(len(probability)):
        values = (
            1.0,
            float(logits[index]),
            float(probability[index]),
            float(mean_probability[index]),
            float(max_probability[index]),
            math.log1p(int(char_lengths[index])),
            float(char_lengths[index] > seed_length_limit),
            float(probability[index] < 0.05),
            float(abs_indices[index]) / max(int(n_physical_lines), 1),
        )
        row = {offset: value for offset, value in enumerate(values) if value != 0.0}
        if include_header and header_kinds[index] > 0:
            row[len(FEATURE_NAMES_BASE)] = 1.0
        rows.append(row)
    return rows


def make_examples(
    table: Any,
    probability: np.ndarray,
    document_indices: Sequence[int],
    *,
    seed_length_limit: int,
    include_header: bool,
) -> list[B1Example]:
    examples: list[B1Example] = []
    unknown_id = LABEL_TO_ID["UNKNOWN"]
    bib_id = LABEL_TO_ID["BIB"]
    for document_index in document_indices:
        document = table.documents[document_index]
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        labels = table.original_labels[doc_start:doc_end]
        absolute = table.abs_indices[doc_start:doc_end]
        features = _feature_rows(
            probability[doc_start:doc_end],
            table.char_lengths[doc_start:doc_end],
            absolute,
            int(document["n_physical_lines"]),
            table.header_kinds[doc_start:doc_end],
            seed_length_limit=seed_length_limit,
            include_header=include_header,
        )
        segment_start = 0
        while segment_start < len(labels):
            while segment_start < len(labels) and labels[segment_start] == unknown_id:
                segment_start += 1
            segment_end = segment_start
            while segment_end < len(labels) and labels[segment_end] != unknown_id:
                if (
                    segment_end > segment_start
                    and int(absolute[segment_end]) - int(absolute[segment_end - 1]) > MAX_PHYSICAL_GAP
                ):
                    break
                segment_end += 1
            if segment_end > segment_start:
                classes = ["BIB" if value == bib_id else "O" for value in labels[segment_start:segment_end]]
                tags = classes_to_bioes(classes)
                examples.append(
                    B1Example(
                        document=None,
                        features=features[segment_start:segment_end],
                        tags=np.asarray([TAG_TO_ID[tag] for tag in tags], dtype=np.int64),
                        line_indices=tuple(range(doc_start + segment_start, doc_start + segment_end)),
                        char_lengths=np.asarray(table.char_lengths[doc_start + segment_start : doc_start + segment_end]),
                        token_counts=np.asarray(table.token_counts[doc_start + segment_start : doc_start + segment_end]),
                    )
                )
            segment_start = max(segment_end, segment_start + 1)
    return examples


def constrained_viterbi(
    model: LinearChainCRF,
    rows: Sequence[Mapping[int, float]],
    char_lengths: np.ndarray,
    *,
    seed_length_limit: int,
    deletion_bias: float,
) -> np.ndarray:
    emissions = model.emission_scores(rows).copy()
    for tag_index, tag in enumerate(TAGS):
        if tag != "O":
            emissions[:, tag_index] -= float(deletion_bias)
        if tag in {"B-BIB", "S-BIB"}:
            emissions[char_lengths > seed_length_limit, tag_index] = -np.inf
    active = np.flatnonzero(model.active_tag_mask)
    emissions = emissions[:, active]
    transition_mask = model.transition_mask[np.ix_(active, active)]
    start_mask = model.start_mask[active]
    end_mask = model.end_mask[active]
    transition = np.where(transition_mask, model.transition[np.ix_(active, active)], -np.inf)
    start = np.where(start_mask, model.start[active], -np.inf)
    end = np.where(end_mask, model.end[active], -np.inf)
    score = np.empty((len(rows), len(active)), dtype=np.float64)
    back = np.zeros((len(rows), len(active)), dtype=np.int16)
    score[0] = start + emissions[0]
    for position in range(1, len(rows)):
        candidates = score[position - 1][:, None] + transition
        back[position] = np.argmax(candidates, axis=0)
        score[position] = emissions[position] + np.max(candidates, axis=0)
    local_tags = np.empty(len(rows), dtype=np.int64)
    allowed_end = np.flatnonzero(end_mask)
    local_tags[-1] = int(allowed_end[np.argmax(score[-1, allowed_end] + end[allowed_end])])
    for position in range(len(rows) - 1, 0, -1):
        local_tags[position - 1] = back[position, local_tags[position]]
    return active[local_tags]


def _example_metrics(
    examples: Sequence[B1Example],
    model: LinearChainCRF,
    *,
    seed_length_limit: int,
    deletion_bias: float,
) -> dict[str, float]:
    tp = fp = fn = 0
    for example in examples:
        predicted = constrained_viterbi(
            model,
            example.features,
            example.char_lengths,
            seed_length_limit=seed_length_limit,
            deletion_bias=deletion_bias,
        )
        guess = np.asarray([TAGS[int(tag)] != "O" for tag in predicted])
        truth = np.asarray([TAGS[int(tag)] != "O" for tag in example.tags])
        tp += int(np.count_nonzero(guess & truth))
        fp += int(np.count_nonzero(guess & ~truth))
        fn += int(np.count_nonzero(~guess & truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    beta2 = 0.25
    f05 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if beta2 * precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f0_5": f05}


def _select_bias(
    examples: Sequence[B1Example], model: LinearChainCRF, *, seed_length_limit: int
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for bias in (0.0, 0.5, 1.0, 1.5, 2.0):
        metrics = _example_metrics(
            examples,
            model,
            seed_length_limit=seed_length_limit,
            deletion_bias=bias,
        )
        rows.append({"deletion_bias": bias, **metrics})
    eligible = [row for row in rows if row["precision"] >= 0.99]
    selected = max(
        eligible or rows,
        key=lambda row: (
            row["recall"] if eligible else row["f0_5"],
            row["precision"],
            -row["deletion_bias"],
        ),
    )
    return float(selected["deletion_bias"]), rows


def _train_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        line_oof_dir,
        arm,
        variant,
        fold,
        seed_length_limit,
        epochs,
        learning_rate,
        l2,
        gradient_clip,
        seed,
    ) = task
    table = load_table(table_dir)
    probability = np.load(
        Path(line_oof_dir) / f"{arm}.oof_probability.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    include_header = variant == "with_header"
    fit_docs = [index for index, document in enumerate(table.documents) if int(document["fold"]) != fold]
    holdout_docs = [index for index, document in enumerate(table.documents) if int(document["fold"]) == fold]
    fit_examples = make_examples(
        table,
        probability,
        fit_docs,
        seed_length_limit=seed_length_limit,
        include_header=include_header,
    )
    holdout_examples = make_examples(
        table,
        probability,
        holdout_docs,
        seed_length_limit=seed_length_limit,
        include_header=include_header,
    )
    n_features = len(FEATURE_NAMES_BASE) + int(include_header)
    model = LinearChainCRF(n_features, seed=seed + fold, active_classes=("BIB",))
    history = train_model(
        model,
        fit_examples,  # type: ignore[arg-type]
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        gradient_clip=gradient_clip,
        seed=seed + fold,
    )
    deletion_bias, calibration = _select_bias(
        fit_examples,
        model,
        seed_length_limit=seed_length_limit,
    )
    indices: list[int] = []
    predictions: list[bool] = []
    for example in holdout_examples:
        tags = constrained_viterbi(
            model,
            example.features,
            example.char_lengths,
            seed_length_limit=seed_length_limit,
            deletion_bias=deletion_bias,
        )
        indices.extend(example.line_indices)
        predictions.extend(TAGS[int(tag)] != "O" for tag in tags)
    return {
        "arm": arm,
        "variant": variant,
        "fold": fold,
        "indices": np.asarray(indices, dtype=np.uint32),
        "predictions": np.asarray(predictions, dtype=bool),
        "model": model,
        "history": history,
        "deletion_bias": deletion_bias,
        "calibration": calibration,
        "fit_document_count": len(fit_docs),
        "holdout_document_count": len(holdout_docs),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def attach_h0_table(
    table: Any,
    prediction: np.ndarray,
    probability: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    result = prediction.copy()
    for document in table.documents:
        start, end = int(document["line_start"]), int(document["line_end"])
        result[start:end] = attach_h0_document(
            result[start:end],
            probability[start:end],
            table.header_kinds[start:end],
            table.abs_indices[start:end],
            config,
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir)
    block_root = Path(args.block_oof_dir).resolve()
    block_report = json.loads((block_root / "block_oof_report.json").read_text(encoding="utf-8"))
    retained = tuple(block_report["retained_for_b1"])
    if len(retained) != 2 or block_report.get("validation_opened") is not False:
        raise ValueError("B0 report does not contain two safe retained arms")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()

    tasks = []
    for arm in retained:
        seed_limit = int(block_report["arms"][arm]["selected_config"]["seed_length_limit"])
        for variant in ("no_header", "with_header"):
            for fold in range(int(table.manifest["n_folds"])):
                tasks.append(
                    (
                        str(table.root),
                        str(Path(args.line_oof_dir).resolve()),
                        arm,
                        variant,
                        fold,
                        seed_limit,
                        int(args.epochs),
                        float(args.learning_rate),
                        float(args.l2),
                        float(args.gradient_clip),
                        int(args.seed),
                    )
                )
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        results = list(executor.map(_train_fold, tasks, chunksize=1))

    variants: dict[str, Any] = {}
    for arm in retained:
        probability = np.load(
            Path(args.line_oof_dir) / f"{arm}.oof_probability.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        config = BlockConfig(**block_report["arms"][arm]["selected_config"])
        for variant in ("no_header", "with_header"):
            key = f"{arm}:{variant}"
            prediction = np.zeros(len(table.targets), dtype=bool)
            fold_rows = []
            for result in results:
                if result["arm"] != arm or result["variant"] != variant:
                    continue
                prediction[result["indices"]] = result["predictions"]
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "arm": arm,
                    "variant": variant,
                    "fold": result["fold"],
                    "feature_names": list(FEATURE_NAMES_BASE) + (["exact_header_candidate"] if variant == "with_header" else []),
                    "deletion_bias": result["deletion_bias"],
                    "calibration": result["calibration"],
                    "history": result["history"],
                    "active_classes": ["BIB"],
                }
                result["model"].save(model_dir / f"{arm}.{variant}.fold{result['fold']}.npz", metadata)
                fold_rows.append({name: result[name] for name in ("fold", "history", "deletion_bias", "calibration", "fit_document_count", "holdout_document_count")})
            prediction = attach_h0_table(table, prediction, probability, config)
            _save_array(output_dir / f"{arm}.{variant}.oof_prediction.npy", prediction)
            metrics = evaluate_prediction(table, prediction)
            variants[key] = {"metrics": metrics, "folds": sorted(fold_rows, key=lambda row: row["fold"])}

    selected_by_arm = {}
    for arm in retained:
        baseline_key, header_key = f"{arm}:no_header", f"{arm}:with_header"
        baseline, header = variants[baseline_key]["metrics"], variants[header_key]["metrics"]
        # Header evidence is accepted only when it does not lower precision and
        # improves recall; otherwise retain the required no-header baseline.
        selected = header_key if (
            header["line_precision"] >= baseline["line_precision"]
            and header["token_recall"] > baseline["token_recall"]
        ) else baseline_key
        selected_by_arm[arm] = selected

    selected_arm = max(
        retained,
        key=lambda arm: (
            variants[selected_by_arm[arm]]["metrics"]["line_precision"] >= 0.99,
            variants[selected_by_arm[arm]]["metrics"]["token_recall"],
            variants[selected_by_arm[arm]]["metrics"]["token_f0_5"],
        ),
    )
    selected_key = selected_by_arm[selected_arm]
    selected_metrics = variants[selected_key]["metrics"]
    b0_metrics = block_report["arms"][selected_arm]["selected_b0_plus_h0_metrics"]
    b1_errors = selected_metrics["split_error_count"] + selected_metrics["merge_error_count"]
    b0_errors = b0_metrics["split_error_count"] + b0_metrics["merge_error_count"]
    b1_error_rate = b1_errors / max(int(selected_metrics["gold_block_count"]), 1)
    b2_required = b1_error_rate > 0.05 and b1_errors >= 0.9 * b0_errors
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_train_oof_b1_validation_unopened",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "block_oof_report_sha256": _sha256(block_root / "block_oof_report.json"),
        "retained_arms": list(retained),
        "variants": variants,
        "selected_variant_by_arm": selected_by_arm,
        "selected_arm": selected_arm,
        "selected_key": selected_key,
        "b2_gate": {
            "required": b2_required,
            "rule": "run when B1 split+merge rate exceeds 5% and is at least 90% of B0/H0 errors",
            "b1_split_merge_rate": b1_error_rate,
            "b1_split_merge_count": b1_errors,
            "b0_split_merge_count": b0_errors,
        },
        "training": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "gradient_clip": float(args.gradient_clip),
            "workers": int(args.workers),
        },
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "b1_oof_report.json", report)
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
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

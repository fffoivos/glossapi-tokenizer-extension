#!/usr/bin/env python3
"""Train-only learned gate over high-recall bibliography components.

The line and sequence models propose spans at several permissive deletion
biases.  Two small component classifiers then decide whether each proposed
span is a coherent bibliography region.  Every candidate feature has one
structural job; no new citation regex or validation input is used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import PINNED_SKLEARN_VERSION, load_table
from .bibliography_entry_sequence import constrained_viterbi, make_examples
from .bibliography_entry_sequence_ablation import VARIANTS
from .feature_crf import LinearChainCRF
from .features import TAGS


SCHEMA_VERSION = "bibliography-entry-component-gate-oof-v1"
FEATURE_NAMES = (
    "log1p_component_line_count",
    "log1p_strong_anchor_count",
    "median_entry_probability",
    "longest_weak_run_fraction",
    "exact_header_at_or_before_start",
)
EXPECTED_DIRECTIONS = (1, 1, 1, -1, 1)
MODEL_ARMS = ("logistic_l2", "monotonic_hgb")
POSITIVE_PURITY = 0.80
NEGATIVE_PURITY = 0.20


@dataclass(frozen=True)
class CandidateSet:
    features: np.ndarray
    document_indices: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    labels: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def _span_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(
        0, min(left[1], right[1]) - max(left[0], right[0]) + 1
    )
    union = left[1] - left[0] + right[1] - right[0] + 2 - intersection
    return intersection / union if union else 0.0


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for enabled in values:
        current = current + 1 if bool(enabled) else 0
        longest = max(longest, current)
    return longest


def candidate_supervision(gold_lines: np.ndarray) -> int:
    """Label pure candidates and mask mixed boundary candidates.

    A candidate wholly inside one long bibliography block is useful even when
    its span IoU with that entire block is small. Candidate purity expresses
    the actual removal decision: how much of this proposed component is BIB.
    """

    if not len(gold_lines):
        raise ValueError("candidate supervision needs at least one line")
    purity = float(np.mean(np.asarray(gold_lines, dtype=bool)))
    if purity >= POSITIVE_PURITY:
        return 1
    if purity <= NEGATIVE_PURITY:
        return 0
    return -1


def component_feature_vector(
    probability: np.ndarray,
    char_lengths: np.ndarray,
    header_kinds: np.ndarray,
    abs_indices: np.ndarray,
    start: int,
    end: int,
    config: BlockConfig,
) -> np.ndarray:
    """Return five non-redundant structural measurements for one component."""

    values = np.asarray(probability[start : end + 1], dtype=np.float64)
    lengths = np.asarray(char_lengths[start : end + 1])
    if not len(values):
        raise ValueError("component must contain at least one line")
    strong = (values >= config.anchor_probability) & (
        lengths <= config.seed_length_limit
    )
    weak = values < config.inside_probability
    header_at_or_before = any(
        int(header_kinds[index]) > 0
        and 0 <= int(abs_indices[start]) - int(abs_indices[index])
        <= config.header_window
        for index in range(
            max(0, start - config.header_window), start + 1
        )
    )
    return np.asarray(
        (
            np.log1p(len(values)),
            np.log1p(np.count_nonzero(strong)),
            np.median(values),
            _longest_true_run(weak) / len(values),
            float(header_at_or_before),
        ),
        dtype=np.float32,
    )


def _load_quality_exclusions(path: Path) -> tuple[set[str], dict[str, Any]]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet.get("schema_version") != "bibliography-training-quality-decisions-v1":
        raise ValueError("unsupported training-quality decision schema")
    decisions = packet.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != int(
        packet["candidate_count"]
    ):
        raise ValueError("training-quality decisions are incomplete")
    document_ids = [str(row["document_id"]) for row in decisions]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("training-quality decisions contain duplicate documents")
    excluded = {
        str(row["document_id"])
        for row in decisions
        if row.get("decision") == "exclude"
    }
    if len(excluded) != int(packet["exclude_count"]):
        raise ValueError("training-quality exclusion count is inconsistent")
    return excluded, packet


def _decode_fold(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        probability_path,
        model_root,
        arm,
        variant,
        fold,
        seed_length_limit,
        deletion_biases,
    ) = task
    table = load_table(table_dir, expected_split="train")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    model, metadata = LinearChainCRF.load(
        Path(model_root) / f"{arm}.{variant}.fold{fold}.npz"
    )
    if (
        metadata.get("variant") != variant
        or tuple(metadata.get("dropped_feature_names", ())) != VARIANTS[variant]
    ):
        raise ValueError("sequence checkpoint does not match component proposal")
    document_indices = [
        index
        for index, document in enumerate(table.documents)
        if int(document["fold"]) == fold
    ]
    examples = make_examples(
        table,
        probability,
        document_indices,
        seed_length_limit=seed_length_limit,
        include_header=False,
        dropped_feature_names=VARIANTS[variant],
    )
    indices: list[int] = []
    predictions = {float(bias): [] for bias in deletion_biases}
    for example in examples:
        indices.extend(example.line_indices)
        for bias in deletion_biases:
            tags = constrained_viterbi(
                model,
                example.features,
                example.char_lengths,
                seed_length_limit=seed_length_limit,
                deletion_bias=float(bias),
            )
            predictions[float(bias)].extend(
                TAGS[int(tag)] != "O" for tag in tags
            )
    return {
        "variant": variant,
        "fold": fold,
        "indices": np.asarray(indices, dtype=np.uint32),
        "predictions": {
            bias: np.asarray(values, dtype=bool)
            for bias, values in predictions.items()
        },
    }


def _proposal_masks(
    table: Any,
    probability_path: Path,
    sequence_root: Path,
    *,
    arm: str,
    variants: Sequence[str],
    deletion_biases: Sequence[float],
    config: BlockConfig,
    workers: int,
) -> dict[tuple[str, float], np.ndarray]:
    tasks = [
        (
            str(table.root),
            str(probability_path),
            str(sequence_root / "models"),
            arm,
            variant,
            fold,
            config.seed_length_limit,
            tuple(deletion_biases),
        )
        for variant in variants
        for fold in range(int(table.manifest["n_folds"]))
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        results = list(executor.map(_decode_fold, tasks, chunksize=1))
    masks = {
        (variant, float(bias)): np.zeros(len(table.targets), dtype=bool)
        for variant in variants
        for bias in deletion_biases
    }
    for result in results:
        for bias, values in result["predictions"].items():
            masks[(result["variant"], float(bias))][result["indices"]] = values
    return masks


def generate_candidates(
    table: Any,
    probability: np.ndarray,
    masks: Mapping[tuple[str, float], np.ndarray],
    *,
    variant: str,
    deletion_biases: Sequence[float],
    config: BlockConfig,
    qualified_documents: set[int],
) -> CandidateSet:
    features: list[np.ndarray] = []
    document_indices: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    labels: list[int] = []
    gold_all = table.original_labels == LABEL_TO_ID["BIB"]
    for document_index, document in enumerate(table.documents):
        if document_index not in qualified_documents:
            continue
        doc_start, doc_end = int(document["line_start"]), int(
            document["line_end"]
        )
        spans: set[tuple[int, int]] = set()
        for bias in deletion_biases:
            spans.update(
                blocks_from_mask(
                    masks[(variant, float(bias))][doc_start:doc_end],
                    table.abs_indices[doc_start:doc_end],
                )
            )
        local_gold = gold_all[doc_start:doc_end]
        for start, end in sorted(spans):
            features.append(
                component_feature_vector(
                    probability[doc_start:doc_end],
                    table.char_lengths[doc_start:doc_end],
                    table.header_kinds[doc_start:doc_end],
                    table.abs_indices[doc_start:doc_end],
                    start,
                    end,
                    config,
                )
            )
            document_indices.append(document_index)
            starts.append(start)
            ends.append(end)
            labels.append(candidate_supervision(local_gold[start : end + 1]))
    if not features:
        raise ValueError(f"no component proposals for {variant}")
    return CandidateSet(
        features=np.stack(features),
        document_indices=np.asarray(document_indices, dtype=np.uint32),
        starts=np.asarray(starts, dtype=np.uint32),
        ends=np.asarray(ends, dtype=np.uint32),
        labels=np.asarray(labels, dtype=np.int8),
    )


def _make_model(arm: str, seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if arm == "logistic_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
    if arm == "monotonic_hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=100,
            max_depth=2,
            min_samples_leaf=20,
            l2_regularization=2.0,
            monotonic_cst=list(EXPECTED_DIRECTIONS),
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"unknown component model arm: {arm}")


def _crossfit_scores(
    table: Any,
    candidates: CandidateSet,
    *,
    arm: str,
    seed: int,
) -> tuple[np.ndarray, list[Any], list[dict[str, Any]], bool]:
    document_folds = np.asarray(
        [int(document["fold"]) for document in table.documents]
    )
    candidate_folds = document_folds[candidates.document_indices]
    labelled = candidates.labels >= 0
    scores = np.full(len(candidates.labels), np.nan, dtype=np.float32)
    models: list[Any] = []
    fold_rows: list[dict[str, Any]] = []
    direction_ok = True
    for fold in range(int(table.manifest["n_folds"])):
        holdout = candidate_folds == fold
        fit = (candidate_folds != fold) & labelled
        model = _make_model(arm, seed + fold)
        model.fit(candidates.features[fit], candidates.labels[fit])
        scores[holdout] = model.predict_proba(
            candidates.features[holdout]
        )[:, 1]
        coefficient = None
        fold_direction_ok = True
        if arm == "logistic_l2":
            coefficient = model.named_steps["logisticregression"].coef_[
                0
            ].tolist()
            fold_direction_ok = all(
                float(value) * expected >= 0
                for value, expected in zip(
                    coefficient, EXPECTED_DIRECTIONS, strict=True
                )
            )
            direction_ok &= fold_direction_ok
        models.append(model)
        fold_rows.append(
            {
                "fold": fold,
                "fit_candidate_count": int(np.count_nonzero(fit)),
                "holdout_candidate_count": int(np.count_nonzero(holdout)),
                "masked_fit_candidate_count": int(
                    np.count_nonzero((candidate_folds != fold) & ~labelled)
                ),
                "coefficient_standardized": coefficient,
                "direction_contract_satisfied": fold_direction_ok,
            }
        )
    if np.isnan(scores).any():
        raise ValueError("component OOF scores are incomplete")
    return scores, models, fold_rows, direction_ok


def decode_candidates(
    table: Any,
    candidates: CandidateSet,
    scores: np.ndarray,
    probability: np.ndarray,
    config: BlockConfig,
    *,
    threshold: float,
    qualified_documents: set[int],
) -> np.ndarray:
    prediction = np.zeros(len(table.targets), dtype=bool)
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        rows = np.flatnonzero(
            (candidates.document_indices == document_index)
            & (scores >= threshold)
        )
        chosen: list[tuple[int, int]] = []
        for row in sorted(
            rows,
            key=lambda index: (
                -float(scores[index]),
                -(int(candidates.ends[index]) - int(candidates.starts[index])),
                int(candidates.starts[index]),
            ),
        ):
            span = (int(candidates.starts[row]), int(candidates.ends[row]))
            if any(_span_iou(span, existing) > 0 for existing in chosen):
                continue
            chosen.append(span)
        doc_start, doc_end = int(document["line_start"]), int(
            document["line_end"]
        )
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


def _proposal_ceiling(
    table: Any,
    candidates: CandidateSet,
    qualified_documents: set[int],
) -> dict[str, float | int]:
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    reachable = np.zeros(len(gold), dtype=bool)
    for document_index, start, end in zip(
        candidates.document_indices,
        candidates.starts,
        candidates.ends,
        strict=True,
    ):
        document = table.documents[int(document_index)]
        offset = int(document["line_start"])
        reachable[offset + int(start) : offset + int(end) + 1] = True
    selected = np.zeros(len(gold), dtype=bool)
    for document_index in qualified_documents:
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        selected[start:end] = True
    gold_selected = gold & selected
    token_total = int(table.token_counts[gold_selected].astype(np.int64).sum())
    token_hit = int(
        table.token_counts[gold_selected & reachable].astype(np.int64).sum()
    )
    line_total = int(np.count_nonzero(gold_selected))
    line_hit = int(np.count_nonzero(gold_selected & reachable))
    return {
        "gold_line_count": line_total,
        "reachable_gold_line_count": line_hit,
        "line_recall_ceiling": line_hit / line_total if line_total else 0.0,
        "gold_token_count": token_total,
        "reachable_gold_token_count": token_hit,
        "token_recall_ceiling": token_hit / token_total if token_total else 0.0,
    }


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        row["model_arm"] == "logistic_l2",
        -float(row["threshold"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sklearn

    if sklearn.__version__ != PINNED_SKLEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {PINNED_SKLEARN_VERSION}, got {sklearn.__version__}"
        )
    table = load_table(args.table_dir, expected_split="train")
    line_root = Path(args.line_oof_dir).resolve()
    sequence_root = Path(args.sequence_oof_dir).resolve()
    block_root = Path(args.block_oof_dir).resolve()
    quality_path = Path(args.quality_decisions).resolve()
    sequence_report_path = sequence_root / "sequence_ablation_oof_report.json"
    block_report_path = block_root / "block_oof_report.json"
    sequence_report = json.loads(
        sequence_report_path.read_text(encoding="utf-8")
    )
    if sequence_report.get("validation_opened") is not False:
        raise ValueError("component gate requires validation-isolated proposals")
    arm = str(args.arm)
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    config = BlockConfig(**block_report["arms"][arm]["selected_config"])
    probability_path = line_root / f"{arm}.oof_probability.npy"
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    excluded_ids, quality_packet = _load_quality_exclusions(quality_path)
    known_ids = {str(document["document_id"]) for document in table.documents}
    if not excluded_ids <= known_ids:
        raise ValueError("quality decisions exclude an unknown train document")
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)

    masks = _proposal_masks(
        table,
        probability_path,
        sequence_root,
        arm=arm,
        variants=args.variants,
        deletion_biases=args.deletion_biases,
        config=config,
        workers=int(args.workers),
    )
    candidate_sets = {
        variant: generate_candidates(
            table,
            probability,
            masks,
            variant=variant,
            deletion_biases=args.deletion_biases,
            config=config,
            qualified_documents=qualified_documents,
        )
        for variant in args.variants
    }

    output_dir.mkdir(parents=True)
    model_dir = output_dir / "models"
    model_dir.mkdir()
    rows: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    score_arrays: dict[tuple[str, str], np.ndarray] = {}
    for variant, candidates in candidate_sets.items():
        prefix = output_dir / f"{variant}.candidates"
        _save_array(prefix.with_suffix(".features.npy"), candidates.features)
        _save_array(
            prefix.with_suffix(".document_indices.npy"),
            candidates.document_indices,
        )
        _save_array(prefix.with_suffix(".starts.npy"), candidates.starts)
        _save_array(prefix.with_suffix(".ends.npy"), candidates.ends)
        _save_array(prefix.with_suffix(".labels.npy"), candidates.labels)
        for model_arm in args.model_arms:
            scores, models, folds, direction_ok = _crossfit_scores(
                table,
                candidates,
                arm=model_arm,
                seed=int(args.seed),
            )
            score_arrays[(variant, model_arm)] = scores
            _save_array(
                output_dir / f"{variant}.{model_arm}.oof_scores.npy", scores
            )
            for fold, model in enumerate(models):
                with (
                    model_dir / f"{variant}.{model_arm}.fold{fold}.pkl"
                ).open("xb") as handle:
                    pickle.dump(model, handle, protocol=5)
            model_reports[f"{variant}:{model_arm}"] = {
                "folds": folds,
                "direction_contract_satisfied": direction_ok,
            }
            for threshold in args.thresholds:
                prediction = decode_candidates(
                    table,
                    candidates,
                    scores,
                    probability,
                    config,
                    threshold=float(threshold),
                    qualified_documents=qualified_documents,
                )
                rows.append(
                    {
                        "variant": variant,
                        "model_arm": model_arm,
                        "threshold": float(threshold),
                        "direction_contract_satisfied": direction_ok,
                        "metrics": evaluate_prediction(
                            table,
                            prediction,
                            document_subset=qualified_documents,
                        ),
                    }
                )

    safe_rows = [
        row
        for row in rows
        if row["direction_contract_satisfied"] and is_safe_candidate(row)
    ]
    selected = max(safe_rows, key=_candidate_key) if safe_rows else None
    precision95_rows = [
        row for row in rows if float(row["metrics"]["line_precision"]) >= 0.95
    ]
    diagnostic_highest_recall = max(rows, key=_candidate_key)
    diagnostic_highest_recall_at_precision95 = (
        max(precision95_rows, key=_candidate_key) if precision95_rows else None
    )
    if selected is not None:
        candidates = candidate_sets[selected["variant"]]
        prediction = decode_candidates(
            table,
            candidates,
            score_arrays[(selected["variant"], selected["model_arm"])],
            probability,
            config,
            threshold=float(selected["threshold"]),
            qualified_documents=qualified_documents,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", prediction)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "feature_names": list(FEATURE_NAMES),
        "expected_feature_directions": list(EXPECTED_DIRECTIONS),
        "feature_reference": {
            "log1p_component_line_count": "Structural extent: larger coherent regions are more plausible bibliography blocks than isolated citations.",
            "log1p_strong_anchor_count": "Repeated support: count normal-length lines with frozen entry probability at least 0.70.",
            "median_entry_probability": "Typical evidence: measure how bibliography-like the middle line score is, not the strongest outlier.",
            "longest_weak_run_fraction": "Internal contradiction: measure the longest uninterrupted prose-like hole as a fraction of the component.",
            "exact_header_at_or_before_start": "Independent structure: record an exact multilingual bibliography heading on the first component line or immediately before it.",
        },
        "proposal": {
            "variants": list(args.variants),
            "deletion_biases": [float(value) for value in args.deletion_biases],
            "candidate_counts": {
                variant: len(candidates.labels)
                for variant, candidates in candidate_sets.items()
            },
            "positive_candidate_counts": {
                variant: int(np.count_nonzero(candidates.labels == 1))
                for variant, candidates in candidate_sets.items()
            },
            "negative_candidate_counts": {
                variant: int(np.count_nonzero(candidates.labels == 0))
                for variant, candidates in candidate_sets.items()
            },
            "masked_boundary_candidate_counts": {
                variant: int(np.count_nonzero(candidates.labels < 0))
                for variant, candidates in candidate_sets.items()
            },
            "candidate_supervision": {
                "positive": "at least 80% of candidate lines are silver BIB",
                "negative": "at most 20% of candidate lines are silver BIB",
                "masked": "mixed 20%-80% boundary candidates are scored but not used for fitting"
            },
            "recall_ceilings": {
                variant: _proposal_ceiling(
                    table, candidates, qualified_documents
                )
                for variant, candidates in candidate_sets.items()
            },
        },
        "model_reports": model_reports,
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": diagnostic_highest_recall,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_highest_recall_at_precision95,
        "selection_rule": "require expected feature directions, line precision>=0.99, and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "block_config": config.__dict__,
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
            "excluded_document_ids": sorted(excluded_ids),
        },
        "input_hashes": {
            "sequence_ablation_report": _sha256(sequence_report_path),
            "block_oof_report": _sha256(block_report_path),
            "line_oof_probability": _sha256(probability_path),
            "quality_decisions": _sha256(quality_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "component_gate_oof_report.json", report)
    outputs = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    receipt = {**report, "outputs": outputs}
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--block-oof-dir", required=True)
    parser.add_argument("--sequence-oof-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", default="D1")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=("no_length", "no_length_or_position"),
    )
    parser.add_argument(
        "--model-arms", nargs="+", choices=MODEL_ARMS, default=MODEL_ARMS
    )
    parser.add_argument(
        "--deletion-biases",
        type=float,
        nargs="+",
        default=(-2.0, -1.5, -1.0, -0.5, 0.0),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1618)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

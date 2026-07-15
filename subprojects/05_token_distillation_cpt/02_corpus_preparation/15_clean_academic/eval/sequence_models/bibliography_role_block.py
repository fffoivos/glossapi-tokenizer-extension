#!/usr/bin/env python3
"""Cost-sensitive semi-Markov bibliography block decoder over role probabilities."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_role_v2 import OPERATIONAL_ROLES, ROLE_TO_ID
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-role-semimarkov-oof-v1"
ROLE_INDEX = {role: index for index, role in enumerate(OPERATIONAL_ROLES)}
INSIDE_ROLES = frozenset({"ENTRY", "CONTINUATION", "FILLER", "BIB_HEADER", "BIB_SUBHEADER"})
INSIDE_ROLE_IDS = frozenset(ROLE_TO_ID[role] for role in INSIDE_ROLES)
FEATURE_NAMES = (
    *(f"role_mean:{role}" for role in OPERATIONAL_ROLES),
    *(f"first_role:{role}" for role in OPERATIONAL_ROLES),
    *(f"last_role:{role}" for role in OPERATIONAL_ROLES),
    "connector_mean",
    "entry_max",
    "entry_min",
    "entry_anchor_count_log1p",
    "entry_anchor_density",
    "span_line_count_log1p",
    "span_char_count_log1p",
    "bib_header_at_start",
    "bib_subheader_internal_max",
    "non_bib_header_internal_max",
    "other_internal_mean",
    "block_bias",
)


@dataclass(frozen=True)
class BlockExample:
    document_id: str
    work_id: str
    source: str
    fold: int
    role_probability: np.ndarray
    connector_probability: np.ndarray
    abs_indices: np.ndarray
    char_lengths: np.ndarray
    gold_roles: np.ndarray
    trusted: np.ndarray

    @property
    def gold_inside(self) -> np.ndarray:
        return self.trusted.astype(bool) & np.isin(self.gold_roles, tuple(INSIDE_ROLE_IDS))


@dataclass(frozen=True)
class DecoderConfig:
    seed_threshold: float = 0.25
    seed_length_limit: int = 330
    anchors_required: int = 2
    candidate_radius: int = 30
    heading_assignment_threshold: float = 0.5


@dataclass(frozen=True)
class StructuredConfig:
    false_positive_cost: float
    fragmentation_cost: float
    false_negative_cost: float = 1.0
    epochs: int = 20
    learning_rate: float = 0.05
    l2: float = 1.0e-4


@dataclass
class StructuredModel:
    weights: np.ndarray
    decoder: DecoderConfig
    training: StructuredConfig


def _contiguous_spans(mask: np.ndarray, abs_indices: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(mask.astype(bool)):
        physical_break = index > 0 and int(abs_indices[index]) - int(abs_indices[index - 1]) > MAX_PHYSICAL_GAP
        if enabled and (start is None or physical_break):
            if start is not None:
                spans.append((start, index - 1))
            start = index
        elif not enabled and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _barriers(example: BlockExample, config: DecoderConfig) -> tuple[np.ndarray, np.ndarray]:
    p = example.role_probability
    heading_indices = np.asarray([
        ROLE_INDEX["BIB_HEADER"], ROLE_INDEX["BIB_SUBHEADER"], ROLE_INDEX["NON_BIB_HEADER"],
    ])
    heading_winner = heading_indices[np.argmax(p[:, heading_indices], axis=1)]
    heading_probability = p[np.arange(len(p)), heading_winner]
    assigned = heading_probability >= config.heading_assignment_threshold
    non_bib = assigned & (heading_winner == ROLE_INDEX["NON_BIB_HEADER"])
    main_bib = assigned & (heading_winner == ROLE_INDEX["BIB_HEADER"])
    return non_bib, main_bib


def candidate_spans(example: BlockExample, config: DecoderConfig) -> list[tuple[int, int]]:
    p_entry = example.role_probability[:, ROLE_INDEX["ENTRY"]]
    non_bib_barrier, bib_header = _barriers(example, config)
    anchors = np.flatnonzero(
        (p_entry >= config.seed_threshold)
        & (example.char_lengths <= config.seed_length_limit)
        & ~non_bib_barrier
    )
    if len(anchors) < config.anchors_required:
        return []
    clusters: list[list[int]] = []
    for anchor in anchors:
        if (
            clusters
            and anchor - clusters[-1][-1] <= 2 * config.candidate_radius
            and not np.any(non_bib_barrier[clusters[-1][-1] + 1 : anchor + 1])
            and not np.any(
                np.diff(example.abs_indices[clusters[-1][-1] : anchor + 1].astype(np.int64))
                > MAX_PHYSICAL_GAP
            )
        ):
            clusters[-1].append(int(anchor))
        else:
            clusters.append([int(anchor)])
    result: set[tuple[int, int]] = set()
    for cluster in clusters:
        if len(cluster) < config.anchors_required:
            continue
        first, last = cluster[0], cluster[-1]
        left_limit = max(0, first - config.candidate_radius)
        right_limit = min(len(p_entry) - 1, last + config.candidate_radius)
        for index in range(first, left_limit, -1):
            if non_bib_barrier[index - 1] or int(example.abs_indices[index]) - int(example.abs_indices[index - 1]) > MAX_PHYSICAL_GAP:
                left_limit = index
                break
            if bib_header[index - 1]:
                left_limit = index - 1
                break
        for index in range(last, right_limit):
            if non_bib_barrier[index + 1] or int(example.abs_indices[index + 1]) - int(example.abs_indices[index]) > MAX_PHYSICAL_GAP:
                right_limit = index
                break
        starts = range(left_limit, first + 1)
        ends = range(last, right_limit + 1)
        for start in starts:
            if np.any(bib_header[start + 1 : last + 1]):
                continue
            for end in ends:
                if np.count_nonzero(
                    (p_entry[start : end + 1] >= config.seed_threshold)
                    & (example.char_lengths[start : end + 1] <= config.seed_length_limit)
                ) < config.anchors_required:
                    continue
                result.add((start, end))
    return sorted(result)


def span_features(example: BlockExample, start: int, end: int, config: DecoderConfig) -> np.ndarray:
    p = example.role_probability[start : end + 1]
    entry = p[:, ROLE_INDEX["ENTRY"]]
    anchors = (entry >= config.seed_threshold) & (
        example.char_lengths[start : end + 1] <= config.seed_length_limit
    )
    internal = p[1:-1] if len(p) > 2 else p[:0]
    values = np.concatenate(
        (
            p.mean(axis=0), p[0], p[-1],
            np.asarray(
                (
                    float(example.connector_probability[start : end + 1].mean()),
                    float(entry.max(initial=0.0)), float(entry.min(initial=0.0)),
                    math.log1p(int(np.count_nonzero(anchors))), float(np.mean(anchors)),
                    math.log1p(len(p)),
                    math.log1p(int(example.char_lengths[start : end + 1].sum())),
                    float(p[0, ROLE_INDEX["BIB_HEADER"]]),
                    float(internal[:, ROLE_INDEX["BIB_SUBHEADER"]].max(initial=0.0)) if len(internal) else 0.0,
                    float(internal[:, ROLE_INDEX["NON_BIB_HEADER"]].max(initial=0.0)) if len(internal) else 0.0,
                    float(internal[:, ROLE_INDEX["OTHER"]].mean()) if len(internal) else 0.0,
                    1.0,
                ),
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise RuntimeError("block span feature contract failure")
    return values


def _weighted_interval_schedule(
    spans: Sequence[tuple[int, int]], scores: np.ndarray,
) -> list[tuple[int, int]]:
    if not spans:
        return []
    order = sorted(range(len(spans)), key=lambda index: (spans[index][1], spans[index][0]))
    ordered = [spans[index] for index in order]
    ordered_scores = scores[order]
    previous = []
    for index, (start, _) in enumerate(ordered):
        candidate = index - 1
        while candidate >= 0 and ordered[candidate][1] >= start:
            candidate -= 1
        previous.append(candidate)
    best = np.zeros(len(ordered) + 1, dtype=np.float64)
    take = np.zeros(len(ordered), dtype=bool)
    for index in range(1, len(ordered) + 1):
        include = float(ordered_scores[index - 1]) + best[previous[index - 1] + 1]
        exclude = best[index - 1]
        if include > exclude and include > 0:
            best[index] = include
            take[index - 1] = True
        else:
            best[index] = exclude
    chosen = []
    cursor = len(ordered) - 1
    while cursor >= 0:
        include = float(ordered_scores[cursor]) + best[previous[cursor] + 1]
        if take[cursor] and math.isclose(best[cursor + 1], include, rel_tol=1e-9, abs_tol=1e-9):
            chosen.append(ordered[cursor])
            cursor = previous[cursor]
        else:
            cursor -= 1
    return sorted(chosen)


def path_features(
    example: BlockExample, spans: Sequence[tuple[int, int]], config: DecoderConfig,
) -> np.ndarray:
    result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for start, end in spans:
        result += span_features(example, start, end, config)
    return result


def decode(
    example: BlockExample, model: StructuredModel, *, loss_augmented: bool = False,
) -> list[tuple[int, int]]:
    spans = candidate_spans(example, model.decoder)
    if not spans:
        return []
    gold = example.gold_inside
    total_chars = max(1.0, float(example.char_lengths[example.trusted.astype(bool)].sum()))
    scores = []
    for start, end in spans:
        feature = span_features(example, start, end, model.decoder)
        score = float(np.dot(model.weights, feature))
        if loss_augmented:
            trusted = example.trusted[start : end + 1].astype(bool)
            inside = gold[start : end + 1]
            chars = example.char_lengths[start : end + 1].astype(np.float64)
            false_positive = float(chars[trusted & ~inside].sum()) / total_chars
            recovered_gold = float(chars[inside].sum()) / total_chars
            score += model.training.false_positive_cost * false_positive
            score -= model.training.false_negative_cost * recovered_gold
            if not np.any(inside):
                score += model.training.fragmentation_cost
        scores.append(score)
    return _weighted_interval_schedule(spans, np.asarray(scores, dtype=np.float64))


def train_structured(
    examples: Sequence[BlockExample], decoder: DecoderConfig, training: StructuredConfig,
    *, seed: int,
) -> StructuredModel:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    averaged = np.zeros_like(weights)
    steps = 0
    rng = np.random.default_rng(seed)
    for epoch in range(training.epochs):
        order = rng.permutation(len(examples))
        for position in order:
            example = examples[int(position)]
            gold_spans = _contiguous_spans(example.gold_inside, example.abs_indices)
            current = StructuredModel(weights, decoder, training)
            predicted = decode(example, current, loss_augmented=True)
            weights *= 1.0 - training.learning_rate * training.l2
            if predicted != gold_spans:
                delta = path_features(example, gold_spans, decoder) - path_features(example, predicted, decoder)
                weights += training.learning_rate * delta
            averaged += weights
            steps += 1
    final = averaged / max(1, steps)
    return StructuredModel(final.astype(np.float32), decoder, training)


def prediction_mask(length: int, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    for start, end in spans:
        result[start : end + 1] = True
    return result


def gold_is_seed_reachable(example: BlockExample, config: DecoderConfig) -> bool:
    p_entry = example.role_probability[:, ROLE_INDEX["ENTRY"]]
    seed = (p_entry >= config.seed_threshold) & (
        example.char_lengths <= config.seed_length_limit
    )
    for start, end in _contiguous_spans(example.gold_inside, example.abs_indices):
        if np.count_nonzero(seed[start : end + 1]) < config.anchors_required:
            return False
    return True


def evaluate(examples: Sequence[BlockExample], models: Mapping[int, StructuredModel]) -> dict[str, Any]:
    tp = fp = fn = spurious_zero = zero_documents = hard_stop_crossings = 0
    char_tp = char_fp = char_fn = 0
    predicted_blocks = gold_blocks = 0
    exact = 0
    ious = []
    by_source: dict[str, dict[str, int]] = {}
    for example in examples:
        spans = decode(example, models[example.fold])
        pred = prediction_mask(len(example.gold_roles), spans)
        gold = example.gold_inside
        trusted = example.trusted.astype(bool)
        local_tp = int(np.count_nonzero(pred & gold))
        local_fp = int(np.count_nonzero(pred & trusted & ~gold))
        local_fn = int(np.count_nonzero(~pred & gold))
        tp += local_tp
        fp += local_fp
        fn += local_fn
        lengths = example.char_lengths.astype(np.int64)
        local_char_tp = int(lengths[pred & gold].sum())
        local_char_fp = int(lengths[pred & trusted & ~gold].sum())
        local_char_fn = int(lengths[~pred & gold].sum())
        char_tp += local_char_tp
        char_fp += local_char_fp
        char_fn += local_char_fn
        source = by_source.setdefault(
            example.source,
            {"tp": 0, "fp": 0, "fn": 0, "char_tp": 0, "char_fp": 0, "char_fn": 0},
        )
        source["tp"] += local_tp
        source["fp"] += local_fp
        source["fn"] += local_fn
        source["char_tp"] += local_char_tp
        source["char_fp"] += local_char_fp
        source["char_fn"] += local_char_fn
        predicted_blocks += len(spans)
        gold_blocks += len(_contiguous_spans(gold, example.abs_indices))
        if not gold.any():
            zero_documents += 1
            spurious_zero += len(spans)
        hard_stop = example.trusted.astype(bool) & (example.gold_roles == ROLE_TO_ID["NON_BIB_HEADER"])
        hard_stop_crossings += int(np.count_nonzero(pred & hard_stop))
        exact += int(np.array_equal(pred & trusted, gold))
        union = np.count_nonzero((pred | gold) & trusted)
        ious.append(np.count_nonzero(pred & gold) / union if union else 1.0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    char_precision = char_tp / (char_tp + char_fp) if char_tp + char_fp else 1.0
    char_recall = char_tp / (char_tp + char_fn) if char_tp + char_fn else 0.0
    return {
        "line_precision": precision, "line_recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "char_precision": char_precision, "char_recall": char_recall,
        "char_tp": char_tp, "char_fp": char_fp, "char_fn": char_fn,
        "predicted_block_count": predicted_blocks, "gold_block_count": gold_blocks,
        "exact_document_rate": exact / len(examples) if examples else 0.0,
        "mean_line_iou": float(np.mean(ious)) if ious else 0.0,
        "spurious_blocks_per_zero_document": spurious_zero / zero_documents if zero_documents else 0.0,
        "trusted_hard_stop_crossings": hard_stop_crossings,
        "by_source": {
            source: {
                **counts,
                "precision": counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 1.0,
                "recall": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0,
                "char_precision": counts["char_tp"] / (counts["char_tp"] + counts["char_fp"])
                if counts["char_tp"] + counts["char_fp"] else 1.0,
                "char_recall": counts["char_tp"] / (counts["char_tp"] + counts["char_fn"])
                if counts["char_tp"] + counts["char_fn"] else 0.0,
            }
            for source, counts in sorted(by_source.items())
        },
    }


def load_examples(root: Path) -> tuple[list[BlockExample], Mapping[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "role_probability", "connector_probability", "abs_indices", "char_lengths",
            "gold_roles", "trusted",
        )
    }
    documents = list(_iter_jsonl(root / "documents.jsonl"))
    examples = []
    for row in documents:
        start, end = int(row["line_start"]), int(row["line_end"])
        examples.append(BlockExample(
            document_id=str(row["document_id"]), work_id=str(row["work_id"]),
            source=str(row["source"]), fold=int(row["fold"]),
            role_probability=np.asarray(arrays["role_probability"][start:end]),
            connector_probability=np.asarray(arrays["connector_probability"][start:end]),
            abs_indices=np.asarray(arrays["abs_indices"][start:end]),
            char_lengths=np.asarray(arrays["char_lengths"][start:end]),
            gold_roles=np.asarray(arrays["gold_roles"][start:end]),
            trusted=np.asarray(arrays["trusted"][start:end]),
        ))
    return examples, manifest


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root, output = Path(args.table_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    examples, manifest = load_examples(table_root)
    outer_models: dict[int, StructuredModel] = {}
    reports = []
    n_folds = int(manifest["n_folds"])
    decoder = DecoderConfig()
    grid = [StructuredConfig(fp, fragment) for fp in (2.0, 4.0, 8.0) for fragment in (0.0, 1.0, 2.0)]
    for outer in range(n_folds):
        inner = (outer + 1) % n_folds
        train_all = [row for row in examples if row.fold not in {outer, inner}]
        train = [row for row in train_all if gold_is_seed_reachable(row, decoder)]
        tune = [row for row in examples if row.fold == inner]
        if not train or not tune:
            raise ValueError(f"fold {outer} has an empty train or tune partition")
        candidates = []
        for offset, config in enumerate(grid):
            model = train_structured(train, decoder, config, seed=args.seed + outer * 100 + offset)
            metrics = evaluate(tune, {inner: model})
            candidates.append({"config": config.__dict__, "metrics": metrics})
        eligible = [
            row for row in candidates
            if row["metrics"]["line_precision"] >= 0.99
            and row["metrics"]["trusted_hard_stop_crossings"] == 0
        ]
        pool = eligible or candidates
        selected = max(pool, key=lambda row: (
            row["metrics"]["line_recall"], row["metrics"]["line_precision"],
            -row["config"]["false_positive_cost"], -row["config"]["fragmentation_cost"],
        ))
        config = StructuredConfig(**selected["config"])
        fit_all = [row for row in examples if row.fold != outer]
        fit = [row for row in fit_all if gold_is_seed_reachable(row, decoder)]
        model = train_structured(fit, decoder, config, seed=args.seed + outer)
        outer_models[outer] = model
        with (output / f"fold{outer}.pkl").open("xb") as handle:
            pickle.dump(model, handle, protocol=5)
        reports.append({
            "outer_fold": outer, "inner_fold": inner, "selected": selected,
            "inner_train_sequence_count": len(train),
            "inner_train_excluded_seed_unreachable": len(train_all) - len(train),
            "outer_fit_sequence_count": len(fit),
            "outer_fit_excluded_seed_unreachable": len(fit_all) - len(fit),
            "candidates": candidates,
        })
    metrics = evaluate(examples, outer_models)
    report = {
        "schema_version": SCHEMA_VERSION, "status": "passed_grouped_oof_structured_training",
        "validation_opened": False, "code_commit": args.code_commit, "slurm_job_id": args.slurm_job_id,
        "feature_names": FEATURE_NAMES, "decoder_config": decoder.__dict__,
        "folds": reports, "oof_metrics": metrics,
        "seed_reachable_sequence_count": sum(
            gold_is_seed_reachable(row, decoder) for row in examples
        ),
        "table_manifest_sha256": sha256_file(table_root / "manifest.json"),
        "deployment_gate_passed": (
            metrics["line_precision"] >= 0.99
            and metrics["line_recall"] >= 0.95
            and metrics["char_precision"] >= 0.99
            and metrics["char_recall"] >= 0.95
            and metrics["trusted_hard_stop_crossings"] == 0
            and metrics["spurious_blocks_per_zero_document"] <= 0.02
        ),
    }
    _write_json_new(output / "report.json", report)
    _write_json_new(output / "receipt.json", {
        **report,
        "outputs": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    })
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

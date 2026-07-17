#!/usr/bin/env python3
"""Materialize gap-only examples between two frozen bibliography anchors.

The line sequence contains only the lines strictly between the anchors.  The
anchor lines are never copied into the model input.  Their only contribution
is through the directional join features already calculated for the first and
last gap lines.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .bibliography_entry_models import load_table
from .bibliography_filler_feature_audit import _HistoricalConnectorUnpickler
from .bibliography_positional_models import load_positional_table
from .bibliography_role_experts import CONNECTOR_PROBABILITY_COLUMNS
from .bibliography_role_features import (
    HEADING_PROBABILITY_NAMES,
    candidate_window_mask,
    connector_feature_names,
    connector_feature_row,
    p0d_matrix,
)
from .bibliography_role_tables import (
    _aligned_source,
    _load_p0d_models,
    _prepare_output,
    _save,
    _write_json_new,
)
from .bibliography_scope_rules import auxiliary_scope_mask
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-connect-table-v1"
LABEL_BIB = 1

EXTRA_FEATURE_NAMES = (
    "current_entry_probability",
    *(f"predicted:{name}" for name in CONNECTOR_PROBABILITY_COLUMNS),
    "document_repetition_log1p",
    "document_repetition_fraction",
    "empty_table_cell_fraction",
    "isolated_ocr_glyph",
    "gap_relative_position",
    "gap_distance_left_log1p",
    "gap_distance_right_log1p",
    "gap_line_count_log1p",
)
FEATURE_NAMES = (*connector_feature_names(), *EXTRA_FEATURE_NAMES)


@dataclass(frozen=True)
class GapPair:
    document_index: int
    document_id: str
    work_id: str
    source: str
    fold: int
    left: int
    right: int
    target: int
    length_bucket: str

    @property
    def gap_length(self) -> int:
        return self.right - self.left - 1

    @property
    def identity(self) -> str:
        return f"{self.document_id}:{self.left}:{self.right}"


def gap_length_bucket(length: int) -> str:
    if length <= 0:
        raise ValueError("gap model requires at least one interior line")
    for limit, label in (
        (1, "1"), (2, "2"), (5, "3-5"), (10, "6-10"),
        (30, "11-30"), (60, "31-60"), (120, "61-120"),
        (200, "121-200"),
    ):
        if length <= limit:
            return label
    return ">200"


def _stable_rank(pair: GapPair, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pair.identity}".encode("utf-8")).hexdigest()


def select_pairs(
    pairs: Sequence[GapPair], *, positive_to_negative: int,
    minimum_positive_per_group: int, seed: int,
) -> list[GapPair]:
    """Keep every negative and a deterministic source/fold/length positive sample."""

    grouped: dict[tuple[int, str, str], dict[int, list[GapPair]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for pair in pairs:
        grouped[(pair.fold, pair.source, pair.length_bucket)][pair.target].append(pair)
    selected: list[GapPair] = []
    for values in grouped.values():
        negatives, positives = values[0], values[1]
        selected.extend(negatives)
        quota = max(minimum_positive_per_group, positive_to_negative * len(negatives))
        selected.extend(sorted(positives, key=lambda pair: _stable_rank(pair, seed))[:quota])
    return sorted(selected, key=lambda pair: (
        pair.document_index, pair.left, pair.right, pair.target,
    ))


def typed_heading_barrier(probability: np.ndarray, *, threshold: float) -> np.ndarray:
    """Main bibliography and non-bibliography headings split the edge.

    Bibliography subheaders are deliberately not barriers.
    """

    if probability.ndim != 2 or probability.shape[1] != len(HEADING_PROBABILITY_NAMES):
        raise ValueError("typed heading probability has the wrong shape")
    main_bib = probability[:, 0]
    non_bib = probability[:, 2]
    return (np.maximum(main_bib, non_bib) >= threshold) & (
        np.maximum(main_bib, non_bib) >= probability[:, 1]
    )


def _normalize_repetition(text: str) -> str:
    return " ".join(text.casefold().split())


def _empty_table_cell_fraction(text: str) -> float:
    if "|" not in text:
        return 0.0
    cells = text.split("|")
    if len(cells) <= 2:
        return 0.0
    interior = cells[1:-1] if not cells[0].strip() and not cells[-1].strip() else cells
    return sum(not cell.strip() for cell in interior) / max(1, len(interior))


def _isolated_ocr_glyph(text: str) -> float:
    stripped = text.strip()
    return float(bool(stripped) and len(stripped) <= 3 and not re.search(r"[\w\d]", stripped))


def _load_connector_models(root: Path, n_folds: int) -> list[Any]:
    result = []
    for fold in range(n_folds):
        with (root / "models" / f"fold{fold}.pkl").open("rb") as handle:
            result.append(_HistoricalConnectorUnpickler(handle).load())
    return result


def _scatter_heading_probability(root: Path, line_count: int) -> np.ndarray:
    rows = np.load(root / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    probability = np.load(root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    if probability.shape != (len(rows), len(HEADING_PROBABILITY_NAMES) + 1):
        raise ValueError("heading OOF probability is malformed")
    result = np.zeros((line_count, len(HEADING_PROBABILITY_NAMES)), dtype=np.float32)
    result[rows] = probability[:, 1:]
    return result


def _pair_inventory(
    *, table: Any, labels_by_doc: Sequence[Sequence[int]],
    entry_probability: np.ndarray,
    heading_probability: np.ndarray, auxiliary_scope_by_doc: Sequence[Sequence[bool]],
    entry_threshold: float, seed_length_limit: int, heading_threshold: float,
    max_gap_lines: int,
) -> tuple[list[GapPair], dict[str, Any]]:
    pairs: list[GapPair] = []
    counts = Counter()
    by_source = Counter()
    by_reason = Counter()
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        entry = entry_probability[start:end]
        lengths = table.char_lengths[start:end]
        abs_indices = table.abs_indices[start:end]
        labels = np.asarray(labels_by_doc[document_index], dtype=np.uint8)
        heading_barrier = typed_heading_barrier(
            heading_probability[start:end], threshold=heading_threshold,
        )
        auxiliary = np.asarray(auxiliary_scope_by_doc[document_index], dtype=bool)
        if auxiliary.shape != (end - start,):
            raise ValueError("auxiliary scope is not aligned to its document")
        seeds = np.flatnonzero(
            (entry >= entry_threshold) & (lengths <= seed_length_limit)
        )
        counts["anchor_count"] += len(seeds)
        for left, right in zip(seeds[:-1], seeds[1:]):
            left, right = int(left), int(right)
            gap_length = right - left - 1
            counts["adjacent_anchor_pairs"] += 1
            if gap_length <= 0:
                counts["zero_length_gap"] += 1
                continue
            if gap_length > max_gap_lines:
                counts["over_length_budget"] += 1
                continue
            if not (labels[left] == LABEL_BIB and labels[right] == LABEL_BIB):
                counts["endpoint_not_silver_bib"] += 1
                continue
            interior = slice(left + 1, right)
            physical = bool(np.any(
                np.diff(abs_indices[left : right + 1].astype(np.int64)) > MAX_PHYSICAL_GAP
            ))
            typed = bool(np.any(heading_barrier[interior]))
            scoped = bool(np.any(auxiliary[interior]))
            target = int(np.all(labels[left : right + 1] == LABEL_BIB))
            counts["eligible_endpoint_pair"] += 1
            counts["positive_before_barrier" if target else "negative_before_barrier"] += 1
            for active, reason in ((physical, "physical"), (typed, "typed_heading"), (scoped, "exact_scope")):
                if active:
                    by_reason[f"{reason}:{'positive' if target else 'negative'}"] += 1
            if physical or typed or scoped:
                counts["deterministically_interrupted_positive" if target else "deterministically_interrupted_negative"] += 1
                continue
            fold = int(table.folds[start + left])
            if np.any(table.folds[start + left : start + right + 1] != fold):
                raise ValueError("work fold changes inside a gap pair")
            pair = GapPair(
                document_index=document_index,
                document_id=str(metadata["document_id"]),
                work_id=str(metadata["work_id"]),
                source=str(metadata["source"]),
                fold=fold,
                left=left,
                right=right,
                target=target,
                length_bucket=gap_length_bucket(gap_length),
            )
            pairs.append(pair)
            by_source[f"{pair.source}:{'positive' if target else 'negative'}"] += 1
    return pairs, {
        "counts": dict(sorted(counts.items())),
        "eligible_by_source_and_target": dict(sorted(by_source.items())),
        "barrier_reasons": dict(sorted(by_reason.items())),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    positional_root = Path(args.positional_table_dir).resolve()
    entry_root = Path(args.entry_oof_dir).resolve()
    heading_root = Path(args.heading_oof_dir).resolve()
    connector_root = Path(args.connector_oof_dir).resolve()
    output = Path(args.output_dir).resolve()
    table = load_table(base_root, expected_split=args.split)
    positional = load_positional_table(
        positional_root, sha256_file(base_root / "manifest.json"), len(table.targets)
    )
    entry_path = entry_root / "P0D.oof_probability.npy"
    entry_probability = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    if entry_probability.shape != (len(table.targets),):
        raise ValueError("entry OOF probability is malformed")
    heading_probability = _scatter_heading_probability(heading_root, len(table.targets))
    texts_by_doc, _, labels_by_doc, _ = _aligned_source(source, table, args.split)
    auxiliary_scope_by_doc = [auxiliary_scope_mask(texts) for texts in texts_by_doc]
    n_folds = int(table.manifest["n_folds"])
    p0d_models = _load_p0d_models(entry_root, n_folds)
    connector_models = _load_connector_models(connector_root, n_folds)

    inventory, inventory_report = _pair_inventory(
        table=table, labels_by_doc=labels_by_doc,
        entry_probability=entry_probability, heading_probability=heading_probability,
        auxiliary_scope_by_doc=auxiliary_scope_by_doc, entry_threshold=args.entry_threshold,
        seed_length_limit=args.seed_length_limit,
        heading_threshold=args.heading_threshold, max_gap_lines=args.max_gap_lines,
    )
    selected = select_pairs(
        inventory, positive_to_negative=args.positive_to_negative,
        minimum_positive_per_group=args.minimum_positive_per_group, seed=args.seed,
    )
    if not selected or {pair.target for pair in selected} != {0, 1}:
        raise ValueError("selected gap table needs both connect and do-not-connect examples")

    by_document: dict[int, list[GapPair]] = defaultdict(list)
    for pair in selected:
        by_document[pair.document_index].append(pair)
    feature_sequences: list[np.ndarray] = []
    line_target_sequences: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    offsets = [0]
    selected_counts = Counter()
    for document_index, pairs in sorted(by_document.items()):
        metadata = table.documents[document_index]
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        texts = texts_by_doc[document_index]
        local_counts = table.counts[start:end]
        local_gaps = positional.gap_summaries[start:end]
        local_abs = table.abs_indices[start:end]
        local_entry = entry_probability[start:end]
        local_heading = heading_probability[start:end]
        local_candidate = candidate_window_mask(
            local_entry, np.any(local_heading > 0, axis=1), local_abs,
            entry_threshold=args.entry_threshold, radius=30,
        )
        fold = int(pairs[0].fold)
        if any(pair.fold != fold for pair in pairs):
            raise ValueError("document gap pairs have inconsistent folds")
        p0d_model = p0d_models[fold]
        connector_model = connector_models[fold]

        def score_counts(values: np.ndarray) -> float:
            return float(p0d_model.predict_proba(p0d_matrix(values))[:, 1][0])

        requested = sorted({offset for pair in pairs for offset in range(pair.left + 1, pair.right)})
        base_rows = []
        for offset in requested:
            base_rows.append(connector_feature_row(
                index=offset, texts=texts, counts=local_counts,
                gap_summaries=local_gaps, abs_indices=local_abs,
                entry_probability=local_entry, heading_probability=local_heading,
                candidate_mask=local_candidate, score_counts=score_counts,
                entry_threshold=args.entry_threshold,
            ).values)
        base_matrix = np.stack(base_rows).astype(np.float32)
        connector_probability = connector_model.predict(base_matrix)
        repetitions = Counter(_normalize_repetition(text) for text in texts if _normalize_repetition(text))
        row_lookup: dict[int, np.ndarray] = {}
        for local_row, offset in enumerate(requested):
            normalized = _normalize_repetition(texts[offset])
            repetition_count = repetitions.get(normalized, 0) if normalized else 0
            extras = np.asarray((
                float(local_entry[offset]),
                *connector_probability[local_row].tolist(),
                math.log1p(repetition_count),
                repetition_count / max(1, len(texts)),
                _empty_table_cell_fraction(texts[offset]),
                _isolated_ocr_glyph(texts[offset]),
            ), dtype=np.float32)
            row_lookup[offset] = np.concatenate((base_matrix[local_row], extras))

        for pair in pairs:
            length = pair.gap_length
            rows = []
            for position, offset in enumerate(range(pair.left + 1, pair.right)):
                relative = (position + 1) / (length + 1)
                position_features = np.asarray((
                    relative,
                    math.log1p(position + 1),
                    math.log1p(length - position),
                    math.log1p(length),
                ), dtype=np.float32)
                rows.append(np.concatenate((row_lookup[offset], position_features)))
            sequence = np.stack(rows).astype(np.float32)
            if sequence.shape != (length, len(FEATURE_NAMES)) or not np.isfinite(sequence).all():
                raise RuntimeError("gap feature sequence violates its contract")
            feature_sequences.append(sequence)
            line_target_sequences.append(np.asarray(
                labels_by_doc[document_index][pair.left + 1 : pair.right],
                dtype=np.uint8,
            ) == LABEL_BIB)
            offsets.append(offsets[-1] + length)
            metadata_rows.append({
                "document_id": pair.document_id,
                "work_id": pair.work_id,
                "source": pair.source,
                "fold": pair.fold,
                "left_local_index": pair.left,
                "right_local_index": pair.right,
                "left_abs_idx": int(local_abs[pair.left]),
                "right_abs_idx": int(local_abs[pair.right]),
                "gap_line_count": length,
                "length_bucket": pair.length_bucket,
                "target_connect": pair.target,
                "label_tier": "LLM_silver_region",
            })
            selected_counts[f"{pair.source}:{'positive' if pair.target else 'negative'}"] += 1

    _prepare_output(output)
    arrays = {
        "features": np.concatenate(feature_sequences).astype(np.float32),
        "line_targets": np.concatenate(line_target_sequences).astype(np.uint8),
        "gap_offsets": np.asarray(offsets, dtype=np.uint64),
        "targets": np.asarray([pair.target for pair in selected], dtype=np.uint8),
        "folds": np.asarray([pair.fold for pair in selected], dtype=np.uint8),
        "gap_lengths": np.asarray([pair.gap_length for pair in selected], dtype=np.uint16),
    }
    if len(metadata_rows) != len(selected):
        raise RuntimeError("gap metadata and selected pair order diverged")
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    with (output / "gaps.jsonl").open("x", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_gap_only_table_materialization",
        "split": args.split,
        "validation_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "label_tier": "LLM_silver_region",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "gap_count": len(selected),
        "gap_line_count": int(arrays["gap_offsets"][-1]),
        "positive_count": int(np.count_nonzero(arrays["targets"])),
        "negative_count": int(len(selected) - np.count_nonzero(arrays["targets"])),
        "selected_by_source_and_target": dict(sorted(selected_counts.items())),
        "inventory": inventory_report,
        "policy": {
            "anchors": "P0D OOF >= threshold, length-limited, both silver-BIB endpoints",
            "sequence": "strictly interior gap lines; anchor lines excluded",
            "hard_barriers": "physical discontinuity, typed main/non-BIB heading, exact negative scope",
            "bib_subheader": "soft connector, never a hard barrier",
            "positive_sampling": {
                "positive_to_negative": args.positive_to_negative,
                "minimum_positive_per_source_fold_length_group": args.minimum_positive_per_group,
                "seed": args.seed,
            },
        },
        "inputs": {
            "source_sha256": sha256_file(source),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "positional_manifest_sha256": sha256_file(positional_root / "manifest.json"),
            "entry_oof_sha256": sha256_file(entry_path),
            "heading_report_sha256": sha256_file(heading_root / "report.json"),
            "heading_oof_sha256": sha256_file(heading_root / "oof_probability.npy"),
            "connector_report_sha256": sha256_file(connector_root / "report.json"),
            "connector_receipt_sha256": sha256_file(connector_root / "receipt.json"),
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(output / "receipt.json", {**manifest, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--positional-table-dir", required=True)
    parser.add_argument("--entry-oof-dir", required=True)
    parser.add_argument("--heading-oof-dir", required=True)
    parser.add_argument("--connector-oof-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--entry-threshold", type=float, default=0.25)
    parser.add_argument("--seed-length-limit", type=int, default=330)
    parser.add_argument("--heading-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap-lines", type=int, default=384)
    parser.add_argument("--positive-to-negative", type=int, default=4)
    parser.add_argument("--minimum-positive-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize multi-regime, gap-only bibliography connection examples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_filler_feature_audit import _HistoricalConnectorUnpickler
from .bibliography_gap_candidates import (
    DEPLOYMENT_THRESHOLD,
    REGIME_ORDER,
    THRESHOLD_LADDER,
    CandidateContext,
    GapCandidate,
    cap_nonbib_by_work,
    enumerate_component_gaps,
    normalize_boundary_weights,
    sample_nonbib_spans,
)
from .bibliography_gap_connect_table import (
    FEATURE_NAMES,
    LABEL_BIB,
    _empty_table_cell_fraction,
    _isolated_ocr_glyph,
    _normalize_repetition,
    _scatter_heading_probability,
    typed_heading_barrier,
)
from .bibliography_positional_models import load_positional_table
from .bibliography_role_features import (
    candidate_window_mask,
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
from .bibliography_entry_models import load_table
from .bibliography_entry_dataset import MAX_PHYSICAL_GAP
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-candidate-table-v2"


def _load_connector_models(root: Path, n_folds: int) -> list[Any]:
    result = []
    for fold in range(n_folds):
        with (root / "models" / f"fold{fold}.pkl").open("rb") as handle:
            result.append(_HistoricalConnectorUnpickler(handle).load())
    return result


def _candidate_inventory(
    *, table: Any, texts_by_doc: Sequence[Sequence[str]],
    labels_by_doc: Sequence[Sequence[int]], entry_probability: np.ndarray,
    heading_probability: np.ndarray, thresholds: Sequence[float],
    deployment_threshold: float, heading_threshold: float,
    seed_length_limit: int, max_gap_lines: int,
) -> tuple[list[GapCandidate], dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    candidates: list[GapCandidate] = []
    document_reports: list[dict[str, Any]] = []
    heading_barriers: list[np.ndarray] = []
    scopes: list[np.ndarray] = []
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        local_heading = heading_probability[start:end]
        barrier = typed_heading_barrier(local_heading, threshold=heading_threshold)
        scope = np.asarray(auxiliary_scope_mask(texts_by_doc[document_index]), dtype=bool)
        heading_barriers.append(barrier)
        scopes.append(scope)
        context = CandidateContext(
            document_index=document_index,
            document_id=str(metadata["document_id"]),
            work_id=str(metadata["work_id"]),
            source=str(metadata["source"]),
            fold=int(table.folds[start]),
        )
        rows, report = enumerate_component_gaps(
            context=context,
            entry_probability=entry_probability[start:end],
            char_lengths=table.char_lengths[start:end],
            abs_indices=table.abs_indices[start:end],
            gold_bib=np.asarray(labels_by_doc[document_index], dtype=np.uint8) == LABEL_BIB,
            typed_heading_barrier=barrier,
            exact_scope=scope,
            thresholds=thresholds,
            deployment_threshold=deployment_threshold,
            seed_length_limit=seed_length_limit,
            max_gap_lines=max_gap_lines,
        )
        candidates.extend(rows)
        document_reports.append({"document_id": context.document_id, **report})

    natural = list(candidates)
    lengths_by_source_fold: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    lengths_by_source: defaultdict[str, list[int]] = defaultdict(list)
    all_positive_lengths = []
    for row in natural:
        if row.target != 1:
            continue
        lengths_by_source_fold[(row.context.source, row.context.fold)].append(row.gap_length)
        lengths_by_source[row.context.source].append(row.gap_length)
        all_positive_lengths.append(row.gap_length)
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        context = CandidateContext(
            document_index=document_index,
            document_id=str(metadata["document_id"]),
            work_id=str(metadata["work_id"]),
            source=str(metadata["source"]),
            fold=int(table.folds[start]),
        )
        lengths = tuple(lengths_by_source_fold[(context.source, context.fold)])
        if not lengths:
            lengths = tuple(lengths_by_source[context.source])
        if not lengths:
            lengths = tuple(all_positive_lengths)
        candidates.extend(sample_nonbib_spans(
            context=context,
            entry_probability=entry_probability[start:end],
            abs_indices=table.abs_indices[start:end],
            gold_bib=np.asarray(labels_by_doc[document_index], dtype=np.uint8) == LABEL_BIB,
            typed_heading_barrier=heading_barriers[document_index],
            exact_scope=scopes[document_index],
            target_lengths=lengths,
        ))
    candidates = normalize_boundary_weights(cap_nonbib_by_work(candidates))
    counts = Counter()
    negative_works: defaultdict[str, set[str]] = defaultdict(set)
    boundary_groups: defaultdict[str, set[str]] = defaultdict(set)
    for row in candidates:
        counts[f"{row.regime}:{'connect' if row.target else 'break'}"] += 1
        counts[f"source:{row.context.source}:{'connect' if row.target else 'break'}"] += 1
        boundary_groups[row.regime].add(row.boundary_group_id)
        if not row.target:
            negative_works[row.regime].add(row.context.work_id)
    report = {
        "candidate_counts": dict(sorted(counts.items())),
        "boundary_group_counts": {
            key: len(value) for key, value in sorted(boundary_groups.items())
        },
        "negative_work_counts": {
            key: len(value) for key, value in sorted(negative_works.items())
        },
        "document_inventory": document_reports,
    }
    return candidates, report, heading_barriers, scopes


def _base_rows(
    *, offsets: Sequence[int], texts: Sequence[str], counts: np.ndarray,
    gap_summaries: np.ndarray, abs_indices: np.ndarray,
    entry_probability: np.ndarray, heading_probability: np.ndarray,
    p0d_model: Any, connector_model: Any, entry_threshold: float,
) -> dict[int, np.ndarray]:
    if not offsets:
        return {}
    candidate = candidate_window_mask(
        entry_probability,
        np.any(heading_probability > 0, axis=1),
        abs_indices,
        entry_threshold=entry_threshold,
        radius=30,
    )

    def score_counts(values: np.ndarray) -> float:
        return float(p0d_model.predict_proba(p0d_matrix(values))[:, 1][0])

    base = np.stack([
        connector_feature_row(
            index=offset,
            texts=texts,
            counts=counts,
            gap_summaries=gap_summaries,
            abs_indices=abs_indices,
            entry_probability=entry_probability,
            heading_probability=heading_probability,
            candidate_mask=candidate,
            score_counts=score_counts,
            entry_threshold=entry_threshold,
        ).values
        for offset in offsets
    ]).astype(np.float32)
    connector_probability = connector_model.predict(base)
    repetitions = Counter(
        _normalize_repetition(text) for text in texts if _normalize_repetition(text)
    )
    result = {}
    for row, offset in enumerate(offsets):
        normalized = _normalize_repetition(texts[offset])
        repetition_count = repetitions.get(normalized, 0) if normalized else 0
        extras = np.asarray((
            float(entry_probability[offset]),
            *connector_probability[row].tolist(),
            math.log1p(repetition_count),
            repetition_count / max(1, len(texts)),
            _empty_table_cell_fraction(texts[offset]),
            _isolated_ocr_glyph(texts[offset]),
        ), dtype=np.float32)
        result[offset] = np.concatenate((base[row], extras))
    return result


def _work_balanced_weights(candidates: Sequence[GapCandidate]) -> np.ndarray:
    totals = Counter()
    for row in candidates:
        totals[row.context.work_id] += row.base_weight
    weights = np.asarray([
        row.base_weight / totals[row.context.work_id] for row in candidates
    ], dtype=np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("candidate work weights are invalid")
    return weights


def _gold_block_group_id(
    document_id: str, labels: np.ndarray, abs_indices: np.ndarray, left: int, right: int,
) -> str | None:
    if not np.all(labels[left : right + 1] == LABEL_BIB):
        return None
    start, end = left, right
    while (
        start > 0
        and labels[start - 1] == LABEL_BIB
        and int(abs_indices[start]) - int(abs_indices[start - 1]) <= MAX_PHYSICAL_GAP
    ):
        start -= 1
    while (
        end + 1 < len(labels)
        and labels[end + 1] == LABEL_BIB
        and int(abs_indices[end + 1]) - int(abs_indices[end]) <= MAX_PHYSICAL_GAP
    ):
        end += 1
    return f"{document_id}:gold:{start}:{end}"


def run(args: argparse.Namespace) -> Mapping[str, Any]:
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
    candidates, inventory, heading_barriers, _ = _candidate_inventory(
        table=table,
        texts_by_doc=texts_by_doc,
        labels_by_doc=labels_by_doc,
        entry_probability=entry_probability,
        heading_probability=heading_probability,
        thresholds=args.thresholds,
        deployment_threshold=args.deployment_threshold,
        heading_threshold=args.heading_threshold,
        seed_length_limit=args.seed_length_limit,
        max_gap_lines=args.max_gap_lines,
    )
    if not candidates or {row.target for row in candidates} != {0, 1}:
        raise ValueError("candidate universe requires both connect and break examples")

    by_document: defaultdict[int, list[GapCandidate]] = defaultdict(list)
    for row in candidates:
        by_document[row.context.document_index].append(row)
    n_folds = int(table.manifest["n_folds"])
    p0d_models = _load_p0d_models(entry_root, n_folds)
    connector_models = _load_connector_models(connector_root, n_folds)
    sequences: list[np.ndarray] = []
    line_targets: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    offsets = [0]

    for document_index, rows in sorted(by_document.items()):
        metadata = table.documents[document_index]
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        texts = texts_by_doc[document_index]
        local_counts = table.counts[start:end]
        local_gaps = positional.gap_summaries[start:end]
        local_abs = table.abs_indices[start:end]
        local_entry = entry_probability[start:end]
        local_heading = np.asarray(heading_probability[start:end], dtype=np.float32)
        fold = int(rows[0].context.fold)
        if any(row.context.fold != fold for row in rows):
            raise ValueError("candidate folds change inside one document")
        requested = sorted({
            index
            for row in rows
            for index in range(row.left + 1, row.right)
            if index not in row.removed_offsets
        })
        original_rows = _base_rows(
            offsets=requested,
            texts=texts,
            counts=local_counts,
            gap_summaries=local_gaps,
            abs_indices=local_abs,
            entry_probability=local_entry,
            heading_probability=local_heading,
            p0d_model=p0d_models[fold],
            connector_model=connector_models[fold],
            entry_threshold=args.deployment_threshold,
        )
        masked_lookup: dict[int, np.ndarray] | None = None
        if any(row.mask_heading_probability for row in rows):
            masked_heading = local_heading.copy()
            masked_heading[heading_barriers[document_index]] = 0.0
            masked_lookup = _base_rows(
                offsets=requested,
                texts=texts,
                counts=local_counts,
                gap_summaries=local_gaps,
                abs_indices=local_abs,
                entry_probability=local_entry,
                heading_probability=masked_heading,
                p0d_model=p0d_models[fold],
                connector_model=connector_models[fold],
                entry_threshold=args.deployment_threshold,
            )

        for row in rows:
            line_offsets = [
                index for index in range(row.left + 1, row.right)
                if index not in row.removed_offsets
            ]
            if not line_offsets:
                raise RuntimeError("candidate has no interior model rows")
            lookup = masked_lookup if row.mask_heading_probability else original_rows
            if lookup is None:
                raise RuntimeError("masked candidate lookup is absent")
            length = len(line_offsets)
            values = []
            for position, index in enumerate(line_offsets):
                relative = (position + 1) / (length + 1)
                position_features = np.asarray((
                    relative,
                    math.log1p(position + 1),
                    math.log1p(length - position),
                    math.log1p(length),
                ), dtype=np.float32)
                values.append(np.concatenate((lookup[index], position_features)))
            sequence = np.stack(values).astype(np.float32)
            if sequence.shape != (length, len(FEATURE_NAMES)) or not np.isfinite(sequence).all():
                raise RuntimeError("candidate feature sequence violates its contract")
            sequences.append(sequence)
            local_labels = np.asarray(labels_by_doc[document_index], dtype=np.uint8)
            line_targets.append(local_labels[line_offsets] == LABEL_BIB)
            offsets.append(offsets[-1] + length)
            metadata_rows.append({
                "boundary_group_id": row.boundary_group_id,
                "variant_id": row.variant_id,
                "document_id": row.context.document_id,
                "work_id": row.context.work_id,
                "source": row.context.source,
                "fold": row.context.fold,
                "left_local_index": row.left,
                "right_local_index": row.right,
                "left_abs_idx": int(local_abs[row.left]),
                "right_abs_idx": int(local_abs[row.right]),
                "original_gap_line_count": row.gap_length,
                "model_line_count": length,
                "target_connect": row.target,
                "gold_block_group_id": _gold_block_group_id(
                    row.context.document_id, local_labels, local_abs, row.left, row.right,
                ),
                "regime": row.regime,
                "generation_thresholds": row.generation_thresholds,
                "synthetic_kind": row.synthetic_kind,
                "removed_offsets": row.removed_offsets,
                "mask_heading_probability": row.mask_heading_probability,
                "virtual_boundaries": row.virtual_boundaries,
                "base_weight": row.base_weight,
                "entry_mean": row.entry_mean,
                "entry_max": row.entry_max,
                "label_tier": "LLM_silver_region",
                "genuine_deployment_candidate": row.regime == "deployment_real",
            })

    _prepare_output(output)
    sample_weights = _work_balanced_weights(candidates)
    arrays = {
        "features": np.concatenate(sequences).astype(np.float32),
        "line_targets": np.concatenate(line_targets).astype(np.uint8),
        "gap_offsets": np.asarray(offsets, dtype=np.uint64),
        "targets": np.asarray([row.target for row in candidates], dtype=np.uint8),
        "folds": np.asarray([row.context.fold for row in candidates], dtype=np.uint8),
        "gap_lengths": np.asarray([row.sequence_length for row in candidates], dtype=np.uint16),
        "sample_weights": sample_weights,
    }
    if len(metadata_rows) != len(candidates):
        raise RuntimeError("candidate metadata ordering diverged")
    for name, value in arrays.items():
        _save(output / f"{name}.npy", value)
    with (output / "gaps.jsonl").open("x", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts = Counter()
    for row in candidates:
        counts[f"{row.regime}:{'connect' if row.target else 'break'}"] += 1
    genuine = np.asarray([row.regime == "deployment_real" for row in candidates])
    targets = arrays["targets"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_multi_regime_gap_candidate_materialization",
        "validation_opened": False,
        "split": args.split,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "candidate_count": len(candidates),
        "model_line_count": int(arrays["gap_offsets"][-1]),
        "candidate_counts": dict(sorted(counts.items())),
        "genuine_candidate_count": int(np.count_nonzero(genuine)),
        "genuine_break_count": int(np.count_nonzero(genuine & (targets == 0))),
        "genuine_connect_count": int(np.count_nonzero(genuine & (targets == 1))),
        "independent_break_boundary_count": len({
            row.boundary_group_id for row in candidates if not row.target
        }),
        "negative_work_count": len({row.context.work_id for row in candidates if not row.target}),
        "inventory": inventory,
        "policy": {
            "regime_order": REGIME_ORDER,
            "threshold_ladder": list(args.thresholds),
            "deployment_threshold": args.deployment_threshold,
            "anchors_in_tensor": False,
            "hard_barriers": "physical discontinuity and exact scope are never ablated",
            "typed_heading": "hard in real regimes; two negative-only miss simulations",
            "header_removed_variant": "sequence-layer deletion; original-document neighbour features retained",
            "nonbib_cap": "at most one hard and one easy span per work",
            "weights": "boundary-capped base weights normalized to total one per work",
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
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLD_LADDER)
    parser.add_argument("--deployment-threshold", type=float, default=DEPLOYMENT_THRESHOLD)
    parser.add_argument("--heading-threshold", type=float, default=0.5)
    parser.add_argument("--seed-length-limit", type=int, default=330)
    parser.add_argument("--max-gap-lines", type=int, default=384)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

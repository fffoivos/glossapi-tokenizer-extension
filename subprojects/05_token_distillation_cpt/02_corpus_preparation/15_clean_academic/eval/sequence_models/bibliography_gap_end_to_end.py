#!/usr/bin/env python3
"""Decode gap-connection OOF probabilities into complete bibliography masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import evaluate_prediction
from .bibliography_entry_models import load_table
from .bibliography_gap_candidate_screen import CandidateTable
from .bibliography_gap_candidates import seed_components
from .bibliography_gap_connect_table import (
    _scatter_heading_probability,
    typed_heading_barrier,
)
from .bibliography_role_tables import _aligned_source
from .bibliography_scope_rules import auxiliary_scope_mask
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-end-to-end-oof-v1"


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def decode_components(
    components: Sequence[tuple[int, int]],
    edge_connect: Mapping[tuple[int, int], bool],
    *,
    line_count: int,
    minimum_anchor_lines: int = 2,
) -> np.ndarray:
    """Join adjacent seed components and emit only independently supported groups."""

    result = np.zeros(line_count, dtype=bool)
    if not components:
        return result
    groups: list[list[tuple[int, int]]] = [[components[0]]]
    for previous, current in zip(components[:-1], components[1:]):
        boundary = (int(previous[1]), int(current[0]))
        if bool(edge_connect.get(boundary, False)):
            groups[-1].append(current)
        else:
            groups.append([current])
    for group in groups:
        anchor_count = sum(end - start + 1 for start, end in group)
        if anchor_count < minimum_anchor_lines:
            continue
        result[group[0][0] : group[-1][1] + 1] = True
    return result


def attach_main_bib_headers(
    prediction: np.ndarray,
    heading_probability: np.ndarray,
    abs_indices: np.ndarray,
    exact_scope: np.ndarray,
    *,
    threshold: float,
    window: int = 2,
) -> np.ndarray:
    result = prediction.copy()
    non_bib = (
        (heading_probability[:, 2] >= threshold)
        & (heading_probability[:, 2] >= heading_probability[:, 0])
        & (heading_probability[:, 2] >= heading_probability[:, 1])
    )
    starts = np.flatnonzero(result & ~np.concatenate(([False], result[:-1])))
    for start in starts:
        for distance in range(1, window + 1):
            candidate = int(start) - distance
            if candidate < 0:
                break
            if int(abs_indices[start]) - int(abs_indices[candidate]) > 64:
                break
            main, sub, non = heading_probability[candidate]
            if main < threshold or main < max(sub, non) or exact_scope[candidate]:
                continue
            if np.any(exact_scope[candidate : int(start)] | non_bib[candidate : int(start)]):
                break
            result[candidate : int(start)] = True
            break
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    entry_root = Path(args.entry_oof_dir).resolve()
    heading_root = Path(args.heading_oof_dir).resolve()
    candidate_root = Path(args.candidate_table_dir).resolve()
    probability_path = Path(args.probability_npz).resolve()
    threshold_path = Path(args.threshold_npz).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    table = load_table(base_root, expected_split=args.split)
    candidates = CandidateTable(candidate_root)
    entry_probability = np.load(
        entry_root / "P0D.oof_probability.npy", mmap_mode="r", allow_pickle=False
    )
    heading_probability = _scatter_heading_probability(heading_root, len(table.targets))
    texts_by_doc, _, _, _ = _aligned_source(source, table, args.split)
    genuine_rows = np.asarray([
        index for index, row in enumerate(candidates.metadata)
        if bool(row["genuine_deployment_candidate"])
    ], dtype=np.int64)
    with np.load(probability_path) as archive:
        probability = np.asarray(archive[args.configuration_key], dtype=np.float32)
    with np.load(threshold_path) as archive:
        thresholds = np.asarray(archive[args.configuration_key], dtype=np.float32)
    if probability.shape != thresholds.shape or probability.shape != (len(genuine_rows),):
        raise ValueError("candidate OOF probability/threshold arrays are misaligned")
    edge_by_document: dict[str, dict[tuple[int, int], bool]] = {}
    for local, candidate_index in enumerate(genuine_rows):
        row = candidates.metadata[int(candidate_index)]
        edges = edge_by_document.setdefault(str(row["document_id"]), {})
        edge = (int(row["left_local_index"]), int(row["right_local_index"]))
        if edge in edges:
            raise ValueError("duplicate genuine deployment edge")
        edges[edge] = bool(probability[local] >= thresholds[local])

    prediction = np.zeros(len(table.targets), dtype=bool)
    hard_stop_crossings = 0
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        local_heading = heading_probability[start:end]
        scope = np.asarray(auxiliary_scope_mask(texts_by_doc[document_index]), dtype=bool)
        barrier = typed_heading_barrier(local_heading, threshold=args.heading_threshold)
        components = seed_components(
            entry_probability[start:end],
            table.char_lengths[start:end],
            table.abs_indices[start:end],
            barrier | scope,
            threshold=args.entry_threshold,
            seed_length_limit=args.seed_length_limit,
        )
        local = decode_components(
            components,
            edge_by_document.get(str(metadata["document_id"]), {}),
            line_count=end - start,
            minimum_anchor_lines=args.minimum_anchor_lines,
        )
        local = attach_main_bib_headers(
            local,
            local_heading,
            table.abs_indices[start:end],
            scope,
            threshold=args.heading_threshold,
            window=args.header_window,
        )
        non_bib = (
            (local_heading[:, 2] >= args.heading_threshold)
            & (local_heading[:, 2] >= local_heading[:, 0])
            & (local_heading[:, 2] >= local_heading[:, 1])
        )
        hard_stop_crossings += int(np.count_nonzero(local & (scope | non_bib)))
        prediction[start:end] = local

    metrics = evaluate_prediction(table, prediction)
    by_source = {}
    for source_name in sorted({str(row["source"]) for row in table.documents}):
        documents = {
            index for index, row in enumerate(table.documents)
            if str(row["source"]) == source_name
        }
        by_source[source_name] = evaluate_prediction(
            table, prediction, document_subset=documents
        )
    with (output / "prediction.npy").open("xb") as handle:
        np.save(handle, prediction.astype(np.uint8), allow_pickle=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_gap_end_to_end_oof",
        "validation_opened": False,
        "deployment_approved": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "configuration_key": args.configuration_key,
        "metrics": metrics,
        "trusted_hard_stop_crossings": hard_stop_crossings,
        "by_source": by_source,
        "gates": {
            "line_precision_at_least_0_99": metrics["line_precision"] >= 0.99,
            "line_recall_at_least_0_95": metrics["line_recall"] >= 0.95,
            "token_precision_at_least_0_99": metrics["token_precision"] >= 0.99,
            "token_recall_at_least_0_95": metrics["token_recall"] >= 0.95,
            "spurious_blocks_per_zero_doc_at_most_0_02": (
                metrics["spurious_blocks_per_zero_block_document"] <= 0.02
            ),
            "zero_trusted_hard_stop_crossings": hard_stop_crossings == 0,
        },
        "inputs": {
            "source_sha256": sha256_file(source),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "candidate_manifest_sha256": sha256_file(candidate_root / "manifest.json"),
            "probability_npz_sha256": sha256_file(probability_path),
            "threshold_npz_sha256": sha256_file(threshold_path),
        },
    }
    report["end_to_end_gate_passed"] = all(report["gates"].values())
    _write_json_new(output / "report.json", report)
    _write_json_new(output / "receipt.json", {**report, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--entry-oof-dir", required=True)
    parser.add_argument("--heading-oof-dir", required=True)
    parser.add_argument("--candidate-table-dir", required=True)
    parser.add_argument("--probability-npz", required=True)
    parser.add_argument("--threshold-npz", required=True)
    parser.add_argument("--configuration-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--entry-threshold", type=float, default=0.25)
    parser.add_argument("--heading-threshold", type=float, default=0.5)
    parser.add_argument("--seed-length-limit", type=int, default=330)
    parser.add_argument("--minimum-anchor-lines", type=int, default=2)
    parser.add_argument("--header-window", type=int, default=2)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explain the train-OOF oracle ceiling of the gap-connection policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask
from .bibliography_entry_models import load_table
from .bibliography_gap_connect_table import (
    LABEL_BIB,
    _scatter_heading_probability,
    typed_heading_barrier,
)
from .bibliography_role_tables import _aligned_source
from .bibliography_scope_rules import auxiliary_scope_mask
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-gap-reachability-audit-v1"
REASONS = (
    "no_eligible_seed_in_gold_block",
    "single_eligible_seed_in_gold_block",
    "leading_edge_outside_seed_extent",
    "trailing_edge_outside_seed_extent",
    "internal_unreachable_between_seeds",
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def classify_oracle_misses(
    gold: np.ndarray,
    oracle: np.ndarray,
    eligible_seed: np.ndarray,
    abs_indices: np.ndarray,
) -> np.ndarray:
    """Assign one mutually exclusive reason to every missed gold line."""

    if not (
        gold.shape == oracle.shape == eligible_seed.shape == abs_indices.shape
        and gold.ndim == 1
    ):
        raise ValueError("reachability arrays are not aligned")
    result = np.full(len(gold), -1, dtype=np.int8)
    for block_start, block_end in blocks_from_mask(gold.astype(bool), abs_indices):
        missed = np.flatnonzero(gold[block_start : block_end + 1] & ~oracle[block_start : block_end + 1])
        if not len(missed):
            continue
        missed = missed + block_start
        seeds = np.flatnonzero(eligible_seed[block_start : block_end + 1]) + block_start
        if not len(seeds):
            result[missed] = 0
            continue
        if len(seeds) == 1:
            result[missed] = 1
            continue
        first, last = int(seeds[0]), int(seeds[-1])
        result[missed[missed < first]] = 2
        result[missed[missed > last]] = 3
        result[missed[(missed >= first) & (missed <= last)]] = 4
    expected = gold.astype(bool) & ~oracle.astype(bool)
    if not np.array_equal(result >= 0, expected):
        raise RuntimeError("oracle misses were not partitioned exactly")
    return result


def _summarize(
    reason: np.ndarray,
    tokens: np.ndarray,
    document_ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    in_scope = np.zeros(len(reason), dtype=bool)
    for start, end in document_ranges:
        in_scope[start:end] = True
    result: dict[str, Any] = {}
    for reason_id, name in enumerate(REASONS):
        mask = (reason == reason_id) & in_scope
        result[name] = {
            "line_count": int(np.count_nonzero(mask)),
            "token_count": int(tokens[mask].astype(np.int64).sum()),
            "document_count": sum(bool(np.any(mask[start:end])) for start, end in document_ranges),
        }
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    entry_root = Path(args.entry_oof_dir).resolve()
    heading_root = Path(args.heading_oof_dir).resolve()
    oracle_path = Path(args.oracle_prediction).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    table = load_table(base_root, expected_split=args.split)
    entry_probability = np.load(
        entry_root / "P0D.oof_probability.npy", mmap_mode="r", allow_pickle=False
    )
    heading_probability = _scatter_heading_probability(heading_root, len(table.targets))
    oracle = np.load(oracle_path, mmap_mode="r", allow_pickle=False).astype(bool)
    if oracle.shape != (len(table.targets),):
        raise ValueError("oracle prediction is not aligned to the base table")
    texts_by_doc, _, labels_by_doc, _ = _aligned_source(source, table, args.split)

    all_reasons = np.full(len(table.targets), -1, dtype=np.int8)
    source_ranges: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    seed_block_counts = defaultdict(int)
    for document_index, metadata in enumerate(table.documents):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        scope = np.asarray(auxiliary_scope_mask(texts_by_doc[document_index]), dtype=bool)
        barrier = typed_heading_barrier(
            heading_probability[start:end], threshold=args.heading_threshold
        )
        eligible = (
            (entry_probability[start:end] >= args.entry_threshold)
            & (table.char_lengths[start:end] <= args.seed_length_limit)
            & ~(scope | barrier)
        )
        gold = np.asarray(labels_by_doc[document_index], dtype=np.uint8) == LABEL_BIB
        local_reason = classify_oracle_misses(
            gold, oracle[start:end], eligible, table.abs_indices[start:end]
        )
        all_reasons[start:end] = local_reason
        source_ranges[str(metadata["source"])].append((start, end))
        for block_start, block_end in blocks_from_mask(gold, table.abs_indices[start:end]):
            count = int(np.count_nonzero(eligible[block_start : block_end + 1]))
            bucket = "0" if count == 0 else "1" if count == 1 else "2_plus"
            seed_block_counts[bucket] += 1

    all_ranges = [
        (int(row["line_start"]), int(row["line_end"])) for row in table.documents
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_oracle_reachability_audit",
        "validation_opened": False,
        "deployment_approved": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "policy": {
            "entry_threshold": args.entry_threshold,
            "heading_threshold": args.heading_threshold,
            "seed_length_limit": args.seed_length_limit,
            "reason_partition": list(REASONS),
        },
        "gold_block_seed_inventory": dict(sorted(seed_block_counts.items())),
        "misses": _summarize(all_reasons, table.token_counts, all_ranges),
        "misses_by_source": {
            source_name: _summarize(all_reasons, table.token_counts, ranges)
            for source_name, ranges in sorted(source_ranges.items())
        },
        "total_missed_lines": int(np.count_nonzero(all_reasons >= 0)),
        "total_missed_tokens": int(
            table.token_counts[all_reasons >= 0].astype(np.int64).sum()
        ),
        "inputs": {
            "source_sha256": sha256_file(source),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "entry_oof_sha256": sha256_file(entry_root / "P0D.oof_probability.npy"),
            "heading_report_sha256": sha256_file(heading_root / "report.json"),
            "oracle_prediction_sha256": sha256_file(oracle_path),
        },
    }
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
    parser.add_argument("--oracle-prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--entry-threshold", type=float, default=0.25)
    parser.add_argument("--heading-threshold", type=float, default=0.5)
    parser.add_argument("--seed-length-limit", type=int, default=330)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

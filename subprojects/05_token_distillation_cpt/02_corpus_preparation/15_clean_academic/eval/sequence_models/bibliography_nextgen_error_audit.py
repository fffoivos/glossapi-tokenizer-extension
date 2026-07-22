#!/usr/bin/env python3
"""Audit exact OOF false-positive and false-negative bibliography regions."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import blocks_from_mask
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-error-audit-v1"


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _iter_jsonl(path: Path, split: str) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split") == split:
                yield value


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _false_positive_kind(
    block: tuple[int, int], gold_blocks: Sequence[tuple[int, int]], fp: np.ndarray
) -> str:
    overlapping = [gold for gold in gold_blocks if _overlaps(block, gold)]
    if not overlapping:
        return "spurious_block"
    first, last = min(row[0] for row in overlapping), max(row[1] for row in overlapping)
    positions = np.flatnonzero(fp[block[0] : block[1] + 1]) + block[0]
    if len(positions) and np.all(positions < first):
        return "left_boundary_overrun"
    if len(positions) and np.all(positions > last):
        return "right_boundary_overrun"
    return "internal_or_merged_overrun"


def _false_negative_kind(
    gold_block: tuple[int, int], prediction: np.ndarray, fn: np.ndarray
) -> str:
    start, end = gold_block
    if not prediction[start : end + 1].any():
        return "whole_block_missed"
    positions = np.flatnonzero(fn[start : end + 1]) + start
    if len(positions) and np.all(positions < np.flatnonzero(prediction[start : end + 1])[0] + start):
        return "left_boundary_miss"
    if len(positions) and np.all(positions > np.flatnonzero(prediction[start : end + 1])[-1] + start):
        return "right_boundary_miss"
    return "internal_or_split_miss"


def _context(lines: Sequence[Mapping[str, Any]], offsets: Sequence[int], radius: int) -> list[dict[str, Any]]:
    selected: set[int] = set()
    for offset in offsets:
        selected.update(range(max(0, offset - radius), min(len(lines), offset + radius + 1)))
    return [
        {
            "offset": index,
            "abs_idx": int(lines[index]["abs_idx"]),
            "text": str(lines[index].get("text", "")),
            "label": str(lines[index].get("label", "UNKNOWN")),
            "is_error": index in offsets,
        }
        for index in sorted(selected)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    prediction_path = Path(args.prediction).resolve()
    baseline_path = Path(args.baseline_prediction).resolve() if args.baseline_prediction else None
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    table = load_table(base_root, expected_split=args.split)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False).astype(bool)
    baseline = (
        np.load(baseline_path, mmap_mode="r", allow_pickle=False).astype(bool)
        if baseline_path
        else np.zeros(len(prediction), dtype=bool)
    )
    if prediction.shape != (len(table.targets),) or baseline.shape != prediction.shape:
        raise ValueError("prediction arrays are not aligned to the base table")
    documents = list(_iter_jsonl(source, args.split))
    if len(documents) != len(table.documents):
        raise ValueError("source/base document count mismatch")

    fp_counts: collections.Counter[str] = collections.Counter()
    fn_counts: collections.Counter[str] = collections.Counter()
    document_rows = []
    for document_index, (document, metadata) in enumerate(
        zip(documents, table.documents, strict=True)
    ):
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        if document.get("document_id") != metadata["document_id"]:
            raise ValueError(f"source/base identity mismatch at {document_index}")
        local_prediction = prediction[start:end]
        local_baseline = baseline[start:end]
        gold = table.original_labels[start:end] == LABEL_TO_ID["BIB"]
        fp = local_prediction & ~gold
        fn = ~local_prediction & gold
        if not fp.any() and not fn.any():
            continue
        abs_indices = table.abs_indices[start:end]
        predicted_blocks = blocks_from_mask(local_prediction, abs_indices)
        gold_blocks = blocks_from_mask(gold, abs_indices)
        fp_regions = []
        for block in predicted_blocks:
            offsets = np.flatnonzero(fp[block[0] : block[1] + 1]) + block[0]
            if not len(offsets):
                continue
            kind = _false_positive_kind(block, gold_blocks, fp)
            fp_counts[kind] += len(offsets)
            fp_regions.append(
                {
                    "kind": kind,
                    "predicted_block": block,
                    "error_offsets": offsets.tolist(),
                    "error_abs_indices": abs_indices[offsets].astype(int).tolist(),
                    "new_vs_baseline_offsets": offsets[~local_baseline[offsets]].tolist(),
                    "context": _context(document["lines"], offsets.tolist(), int(args.context_radius)),
                }
            )
        fn_regions = []
        for block in gold_blocks:
            offsets = np.flatnonzero(fn[block[0] : block[1] + 1]) + block[0]
            if not len(offsets):
                continue
            kind = _false_negative_kind(block, local_prediction, fn)
            fn_counts[kind] += len(offsets)
            fn_regions.append(
                {
                    "kind": kind,
                    "gold_block": block,
                    "error_offsets": offsets.tolist(),
                    "error_abs_indices": abs_indices[offsets].astype(int).tolist(),
                    "context": _context(document["lines"], offsets.tolist(), int(args.context_radius)),
                }
            )
        document_rows.append(
            {
                "document_id": metadata["document_id"],
                "work_id": metadata["work_id"],
                "source": metadata["source"],
                "fold": int(metadata["fold"]),
                "false_positive_line_count": int(np.count_nonzero(fp)),
                "false_negative_line_count": int(np.count_nonzero(fn)),
                "new_false_positive_line_count": int(np.count_nonzero(fp & ~local_baseline)),
                "false_positive_regions": fp_regions,
                "false_negative_regions": fn_regions,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_exact_oof_error_audit",
        "validation_opened": False,
        "test_opened": False,
        "false_positive_lines_by_topology": dict(sorted(fp_counts.items())),
        "false_negative_lines_by_topology": dict(sorted(fn_counts.items())),
        "document_count_with_errors": len(document_rows),
        "top_false_positive_documents": sorted(
            document_rows,
            key=lambda row: (-row["false_positive_line_count"], row["document_id"]),
        )[:25],
        "top_false_negative_documents": sorted(
            document_rows,
            key=lambda row: (-row["false_negative_line_count"], row["document_id"]),
        )[:25],
        "documents": document_rows,
        "inputs": {
            "source_sha256": sha256_file(source),
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "prediction_sha256": sha256_file(prediction_path),
            "baseline_prediction_sha256": sha256_file(baseline_path) if baseline_path else None,
        },
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
    }
    _write_json_new(output, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--baseline-prediction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

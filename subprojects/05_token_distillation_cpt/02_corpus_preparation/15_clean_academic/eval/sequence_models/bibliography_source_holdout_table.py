#!/usr/bin/env python3
"""Re-index a frozen train table so each source is one held-out fold.

The feature values and labels are not rebuilt.  Immutable arrays are hard-linked
and only the fold metadata is replaced.  This derived table is for a robustness
audit of the contextual bibliography model, not for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-entry-source-holdout-table-v1"
IMMUTABLE_ARRAYS = (
    "counts",
    "targets",
    "original_labels",
    "header_kinds",
    "abs_indices",
    "token_counts",
    "char_lengths",
    "block_indices",
    "document_indices",
)


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def assign_source_folds(
    documents: Sequence[Mapping[str, Any]], *, line_count: int
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    """Assign every document from one source to the same deterministic fold."""

    sources = sorted({str(row.get("source", "")).strip() for row in documents})
    if not documents or "" in sources or len(sources) < 2:
        raise ValueError("source holdout needs at least two non-empty sources")
    source_to_fold = {source: index for index, source in enumerate(sources)}
    if len(sources) > np.iinfo(np.uint8).max:
        raise ValueError("too many source folds for uint8 encoding")
    output_rows: list[dict[str, Any]] = []
    folds = np.full(int(line_count), 255, dtype=np.uint8)
    cursor = 0
    for document in documents:
        row = dict(document)
        start, end = int(row["line_start"]), int(row["line_end"])
        if start != cursor or end <= start or end > int(line_count):
            raise ValueError("documents do not form a complete contiguous table")
        fold = source_to_fold[str(row["source"]).strip()]
        row["fold"] = fold
        folds[start:end] = fold
        output_rows.append(row)
        cursor = end
    if cursor != int(line_count) or np.any(folds == 255):
        raise ValueError("source fold assignment did not cover every table line")
    return output_rows, folds, source_to_fold


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_table_dir).resolve()
    source_receipt = source_root / "receipt.json"
    if not source_receipt.is_file() or source_receipt.is_symlink():
        raise ValueError("source table receipt is absent or is a symlink")
    table = load_table(source_root, expected_split="train")
    documents, folds, source_to_fold = assign_source_folds(
        table.documents, line_count=len(table.targets)
    )
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    for name in IMMUTABLE_ARRAYS:
        os.link(source_root / f"{name}.npy", output_root / f"{name}.npy")
    with (output_root / "folds.npy").open("xb") as handle:
        np.save(handle, folds, allow_pickle=False)
    _write_jsonl(output_root / "documents.jsonl", documents)
    _write_json(
        output_root / "folds.json",
        {
            "schema_version": SCHEMA_VERSION,
            "strategy": "leave_one_complete_source_out",
            "source_to_fold": source_to_fold,
            "selection_eligible": False,
        },
    )
    manifest = {
        **table.manifest,
        "status": "passed_source_holdout_fold_materialization",
        "n_folds": len(source_to_fold),
        "fold_seed": "none:source-identity-determines-fold",
        "fold_strategy": "leave_one_complete_source_out",
        "source_to_fold": source_to_fold,
        "source_table_receipt_sha256": _sha256(source_receipt),
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "selection_eligible": False,
    }
    _write_json(output_root / "manifest.json", manifest)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    receipt = {**manifest, "outputs": outputs}
    _write_json(output_root / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Materialize work-clustered objective rows for paired evolution analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import _block_matches, blocks_from_mask, evaluate_prediction
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _selected(table: Any, path: Path) -> tuple[set[int], dict[str, set[int]]]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    ids = packet.get("document_ids")
    if not isinstance(ids, list) or len(ids) != 268 or len(ids) != len(set(ids)):
        raise ValueError("qualified development inventory must contain 268 unique ids")
    index = {str(row["document_id"]): number for number, row in enumerate(table.documents)}
    if not set(ids).issubset(index):
        raise ValueError("qualified development inventory is not in the table")
    selected = {index[value] for value in ids}
    by_source: dict[str, set[int]] = defaultdict(set)
    for value in selected:
        by_source[str(table.documents[value]["source"])].add(value)
    return selected, dict(by_source)


def _document_objectives(table: Any, prediction: np.ndarray, index: int) -> dict[str, Any]:
    document = table.documents[index]
    start, end = int(document["line_start"]), int(document["line_end"])
    gold = table.original_labels[start:end] == LABEL_TO_ID["BIB"]
    pred = prediction[start:end]
    tokens = table.token_counts[start:end].astype(np.int64)
    gold_blocks = blocks_from_mask(gold, table.abs_indices[start:end])
    predicted_blocks = blocks_from_mask(pred, table.abs_indices[start:end])
    matches = _block_matches(gold_blocks, predicted_blocks, 0.5)
    boundary_errors = []
    for gold_index, predicted_index, _ in matches:
        left, right = gold_blocks[gold_index], predicted_blocks[predicted_index]
        boundary_errors.append((abs(left[0] - right[0]) + abs(left[1] - right[1])) / 2)
    return {
        "work_id": str(document["work_id"]),
        "source": str(document["source"]),
        "token_fp": int(tokens[~gold & pred].sum()),
        "token_fn": int(tokens[gold & ~pred].sum()),
        "spurious_zero_blocks": len(predicted_blocks) if not gold_blocks else 0,
        "zero_doc_count": int(not gold_blocks),
        "boundary_error_sum": float(sum(boundary_errors)),
        "boundary_match_count": len(boundary_errors),
        "document_count": 1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="validation")
    prediction = np.load(args.prediction, allow_pickle=False).astype(bool)
    if prediction.shape != (len(table.targets),):
        raise ValueError("prediction does not align to the validation table")
    selected, by_source = _selected(table, Path(args.qualified_documents))
    work_rows: dict[str, dict[str, Any]] = {}
    for document_index in sorted(selected):
        row = _document_objectives(table, prediction, document_index)
        work_id = row["work_id"]
        if work_id not in work_rows:
            work_rows[work_id] = row
            continue
        target = work_rows[work_id]
        if target["source"] != row["source"]:
            raise ValueError("one work identity crosses source strata")
        for field in (
            "token_fp", "token_fn", "spurious_zero_blocks", "zero_doc_count",
            "boundary_error_sum", "boundary_match_count", "document_count",
        ):
            target[field] += row[field]
    rows = [work_rows[key] for key in sorted(work_rows)]
    with Path(args.output_rows).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema_version": "bibliography-evolution-work-objectives-v1",
        "status": "passed",
        "document_count": 268,
        "work_count": len(rows),
        "metrics": evaluate_prediction(table, prediction, document_subset=selected),
        "metrics_by_source": {
            source: evaluate_prediction(table, prediction, document_subset=documents)
            for source, documents in sorted(by_source.items())
        },
    }
    _write_json(Path(args.output_report), result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--qualified-documents", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

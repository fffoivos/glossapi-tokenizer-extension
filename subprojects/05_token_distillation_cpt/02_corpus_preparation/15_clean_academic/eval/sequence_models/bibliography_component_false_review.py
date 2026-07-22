#!/usr/bin/env python3
"""Materialize compact text evidence for train-OOF false components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .bibliography_entry_component_diagnostics import _chosen_rows
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-component-false-review-v1"


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sample_indices(start: int, end: int, total: int, context: int) -> list[int]:
    centres = (start, (start + end) // 2, end)
    selected: set[int] = set()
    for centre in centres:
        selected.update(
            range(max(0, centre - context), min(total, centre + context + 1))
        )
    return sorted(selected)


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    component_dir = Path(args.component_dir).resolve()
    variant = str(args.variant)
    prefix = component_dir / variant
    documents = np.load(
        prefix.with_suffix(".document_indices.npy"), mmap_mode="r", allow_pickle=False
    )
    starts = np.load(
        prefix.with_suffix(".starts.npy"), mmap_mode="r", allow_pickle=False
    )
    ends = np.load(
        prefix.with_suffix(".ends.npy"), mmap_mode="r", allow_pickle=False
    )
    scores = np.load(
        component_dir / f"{variant}.{args.model_arm}.oof_scores.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    chosen = _chosen_rows(documents, starts, ends, scores, float(args.threshold))
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    candidates: list[dict[str, Any]] = []
    for row in chosen:
        document_index = int(documents[row])
        document = table.documents[document_index]
        doc_start = int(document["line_start"])
        start, end = int(starts[row]), int(ends[row])
        purity = float(np.mean(gold[doc_start + start : doc_start + end + 1]))
        if purity > 0.2:
            continue
        candidates.append(
            {
                "candidate_row": int(row),
                "document_id": str(document["document_id"]),
                "source": str(document["source"]),
                "local_start": start,
                "local_end": end,
                "line_count": end - start + 1,
                "score": float(scores[row]),
                "gold_purity": purity,
            }
        )
    candidates.sort(key=lambda row: (-row["line_count"], -row["score"]))
    candidates = candidates[: int(args.limit)]
    requested = {row["document_id"] for row in candidates}
    raw_documents: dict[str, Mapping[str, Any]] = {}
    for document in _iter_rows(Path(args.input).resolve()):
        document_id = str(document.get("document_id"))
        if document_id in requested:
            if document.get("split") != "train":
                raise ValueError("requested false component is not in train")
            raw_documents[document_id] = document
            if raw_documents.keys() == requested:
                break
    if raw_documents.keys() != requested:
        raise ValueError("one or more requested train documents are absent")

    review_rows = []
    for candidate in candidates:
        lines = raw_documents[candidate["document_id"]].get("lines")
        if not isinstance(lines, list):
            raise ValueError("source document has no line inventory")
        indices = _sample_indices(
            candidate["local_start"],
            candidate["local_end"],
            len(lines),
            int(args.context),
        )
        review_rows.append(
            {
                **candidate,
                "sampled_lines": [
                    {
                        "local_index": index,
                        "abs_idx": int(lines[index]["abs_idx"]),
                        "label": str(lines[index]["label"]),
                        "inside_candidate": (
                            candidate["local_start"]
                            <= index
                            <= candidate["local_end"]
                        ),
                        "text": str(lines[index]["text"]),
                    }
                    for index in indices
                ],
            }
        )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_oof_false_component_review_validation_unopened",
        "variant": variant,
        "model_arm": str(args.model_arm),
        "threshold": float(args.threshold),
        "selection": "largest chosen candidates with at most 20% silver BIB lines",
        "review_count": len(review_rows),
        "components": review_rows,
        "validation_opened": False,
    }
    output_path = Path(args.output_path).resolve()
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--variant", default="no_length")
    parser.add_argument("--model-arm", default="logistic_l2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

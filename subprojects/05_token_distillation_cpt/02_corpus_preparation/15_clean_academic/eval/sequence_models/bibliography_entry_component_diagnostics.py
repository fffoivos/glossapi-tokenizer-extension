#!/usr/bin/env python3
"""Describe structural signals in train-OOF bibliography proposals.

This is a read-only diagnostic.  It never fits a model and never reads the
validation split.  Its purpose is to decide whether a proposed component
feature has a coherent, training-only justification before adding it to the
block gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .bibliography_entry_component_gate import _span_iou
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-entry-component-diagnostics-v1"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None, "mean": None}
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _load_candidates(component_dir: Path, variant: str) -> dict[str, np.ndarray]:
    prefix = component_dir / variant
    return {
        name: np.load(
            prefix.with_suffix(f".{name}.npy"), mmap_mode="r", allow_pickle=False
        )
        for name in ("features", "document_indices", "starts", "ends", "labels")
    }


def _chosen_rows(
    document_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply the production overlap rule but do not attach headers."""

    chosen: list[int] = []
    for document_index in np.unique(document_indices):
        rows = np.flatnonzero(
            (document_indices == document_index) & (scores >= threshold)
        )
        doc_chosen: list[tuple[int, int]] = []
        for row in sorted(
            rows,
            key=lambda index: (
                -float(scores[index]),
                -(int(ends[index]) - int(starts[index])),
                int(starts[index]),
            ),
        ):
            span = (int(starts[row]), int(ends[row]))
            if any(_span_iou(span, previous) > 0 for previous in doc_chosen):
                continue
            doc_chosen.append(span)
            chosen.append(int(row))
    return np.asarray(chosen, dtype=np.uint32)


def _line_fraction(
    counts: np.ndarray,
    doc_start: int,
    start: int,
    end: int,
    columns: Sequence[int],
) -> float:
    window = counts[doc_start + start : doc_start + end + 1, columns]
    if window.ndim == 1:
        present = window > 0
    else:
        present = np.any(window > 0, axis=1)
    return float(np.mean(present))


def _outside_context_probability(
    probability: np.ndarray,
    abs_indices: np.ndarray,
    doc_start: int,
    doc_end: int,
    start: int,
    end: int,
    *,
    physical_window: int = 8,
) -> float:
    """Median frozen probability immediately outside a component."""

    absolute_start = doc_start + start
    absolute_end = doc_start + end
    left = np.arange(max(doc_start, absolute_start - physical_window), absolute_start)
    right = np.arange(
        absolute_end + 1,
        min(doc_end, absolute_end + physical_window + 1),
    )
    if len(left):
        left = left[
            abs_indices[absolute_start] - abs_indices[left] <= physical_window
        ]
    if len(right):
        right = right[
            abs_indices[right] - abs_indices[absolute_end] <= physical_window
        ]
    context = np.concatenate((left, right))
    return float(np.median(probability[context])) if len(context) else 0.0


def _proposal_groups(starts: np.ndarray, ends: np.ndarray) -> int:
    """Count non-overlapping proposal islands in one document."""

    spans = sorted(zip(starts.tolist(), ends.tolist(), strict=True))
    groups = 0
    current_end = -1
    for start, end in spans:
        if int(start) > current_end + 1:
            groups += 1
            current_end = int(end)
        else:
            current_end = max(current_end, int(end))
    return groups


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    component_dir = Path(args.component_dir).resolve()
    report = json.loads(
        (component_dir / "component_gate_oof_report.json").read_text(encoding="utf-8")
    )
    if report.get("validation_opened") is not False:
        raise ValueError("diagnostic requires validation-isolated component outputs")
    variant = str(args.variant)
    candidates = _load_candidates(component_dir, variant)
    scores = np.load(
        component_dir / f"{variant}.{args.model_arm}.oof_scores.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    line_probability = np.load(
        Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if len(line_probability) != len(table.targets):
        raise ValueError("line OOF probability does not match the feature table")
    chosen = _chosen_rows(
        candidates["document_indices"],
        candidates["starts"],
        candidates["ends"],
        scores,
        float(args.threshold),
    )

    feature_index = {
        name: index for index, name in enumerate(table.manifest["feature_names"])
    }
    required = (
        "table_row_count",
        "prose_lead_count",
        "numbered_entry_count",
        "inverted_author_count",
        "name_initial_pair_count",
        "direct_author_count",
        "doi_count",
        "isbn_count",
        "issn_count",
        "volume_marker_count",
        "volume_shape_count",
        "journal_year_volume_count",
        "page_marker_count",
        "article_page_range_count",
        "page_range_count",
        "publisher_term_count",
        "place_publisher_shape_count",
    )
    missing = sorted(set(required) - set(feature_index))
    if missing:
        raise ValueError(f"feature table is missing {missing}")

    author_columns = [
        feature_index[name]
        for name in (
            "inverted_author_count",
            "name_initial_pair_count",
            "direct_author_count",
        )
    ]
    publication_columns = [
        feature_index[name]
        for name in (
            "doi_count",
            "isbn_count",
            "issn_count",
            "volume_marker_count",
            "volume_shape_count",
            "journal_year_volume_count",
            "page_marker_count",
            "article_page_range_count",
            "page_range_count",
            "publisher_term_count",
            "place_publisher_shape_count",
        )
    ]
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    doc_gold = np.asarray(
        [
            bool(np.any(gold[int(doc["line_start"]) : int(doc["line_end"])]))
            for doc in table.documents
        ],
        dtype=bool,
    )
    candidate_count_by_doc = np.bincount(
        candidates["document_indices"].astype(np.int64),
        minlength=len(table.documents),
    )
    proposal_groups_by_doc = np.zeros(len(table.documents), dtype=np.uint32)
    for document_index in np.unique(candidates["document_indices"]):
        rows = candidates["document_indices"] == document_index
        proposal_groups_by_doc[int(document_index)] = _proposal_groups(
            candidates["starts"][rows], candidates["ends"][rows]
        )

    rows: list[dict[str, Any]] = []
    for row in range(len(scores)):
        document_index = int(candidates["document_indices"][row])
        document = table.documents[document_index]
        doc_start = int(document["line_start"])
        doc_end = int(document["line_end"])
        start = int(candidates["starts"][row])
        end = int(candidates["ends"][row])
        line_count = end - start + 1
        local_gold = gold[doc_start + start : doc_start + end + 1]
        outside_probability = _outside_context_probability(
            line_probability,
            table.abs_indices,
            doc_start,
            doc_end,
            start,
            end,
        )
        inside_probability = float(
            np.median(line_probability[doc_start + start : doc_start + end + 1])
        )
        rows.append(
            {
                "row": row,
                "document_index": document_index,
                "document_id": str(document["document_id"]),
                "score": float(scores[row]),
                "supervision": int(candidates["labels"][row]),
                "gold_purity": float(np.mean(local_gold)),
                "document_has_gold_bib": bool(doc_gold[document_index]),
                "line_count": line_count,
                "document_coverage_fraction": line_count / (doc_end - doc_start),
                "table_row_fraction": _line_fraction(
                    table.counts,
                    doc_start,
                    start,
                    end,
                    [feature_index["table_row_count"]],
                ),
                "prose_lead_fraction": _line_fraction(
                    table.counts,
                    doc_start,
                    start,
                    end,
                    [feature_index["prose_lead_count"]],
                ),
                "numbered_entry_fraction": _line_fraction(
                    table.counts,
                    doc_start,
                    start,
                    end,
                    [feature_index["numbered_entry_count"]],
                ),
                "author_start_fraction": _line_fraction(
                    table.counts, doc_start, start, end, author_columns
                ),
                "publication_tail_fraction": _line_fraction(
                    table.counts, doc_start, start, end, publication_columns
                ),
                "document_candidate_count": int(
                    candidate_count_by_doc[document_index]
                ),
                "document_proposal_island_count": int(
                    proposal_groups_by_doc[document_index]
                ),
                "outside_context_median_probability": outside_probability,
                "inside_minus_outside_probability": (
                    inside_probability - outside_probability
                ),
            }
        )

    diagnostic_names = (
        "line_count",
        "document_coverage_fraction",
        "table_row_fraction",
        "prose_lead_fraction",
        "numbered_entry_fraction",
        "author_start_fraction",
        "publication_tail_fraction",
        "document_candidate_count",
        "document_proposal_island_count",
        "outside_context_median_probability",
        "inside_minus_outside_probability",
    )
    groups = {
        "supervised_positive": [row for row in rows if row["supervision"] == 1],
        "supervised_negative": [row for row in rows if row["supervision"] == 0],
        "chosen_positive": [
            rows[int(index)] for index in chosen if rows[int(index)]["gold_purity"] >= 0.8
        ],
        "chosen_mixed": [
            rows[int(index)]
            for index in chosen
            if 0.2 < rows[int(index)]["gold_purity"] < 0.8
        ],
        "chosen_negative": [
            rows[int(index)] for index in chosen if rows[int(index)]["gold_purity"] <= 0.2
        ],
        "chosen_zero_bib_document": [
            rows[int(index)]
            for index in chosen
            if not rows[int(index)]["document_has_gold_bib"]
        ],
    }
    summaries = {
        group: {
            name: _quantiles(row[name] for row in group_rows)
            for name in diagnostic_names
        }
        for group, group_rows in groups.items()
    }
    false_documents: dict[str, dict[str, Any]] = {}
    for row in groups["chosen_negative"]:
        entry = false_documents.setdefault(
            row["document_id"],
            {
                "document_id": row["document_id"],
                "document_has_gold_bib": row["document_has_gold_bib"],
                "chosen_negative_count": 0,
                "false_line_count": 0,
                "max_score": 0.0,
                "document_candidate_count": row["document_candidate_count"],
                "document_proposal_island_count": row[
                    "document_proposal_island_count"
                ],
            },
        )
        entry["chosen_negative_count"] += 1
        entry["false_line_count"] += row["line_count"]
        entry["max_score"] = max(entry["max_score"], row["score"])

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "train_oof_diagnostic_validation_unopened",
        "variant": variant,
        "model_arm": str(args.model_arm),
        "threshold": float(args.threshold),
        "candidate_count": len(rows),
        "chosen_component_count": int(len(chosen)),
        "group_counts": {name: len(group_rows) for name, group_rows in groups.items()},
        "feature_reference": {
            "table_row_fraction": "Share of component lines formatted as Markdown table rows; a negative block-format signal.",
            "prose_lead_fraction": "Share of component lines beginning with a frozen prose-lead pattern; a negative running-prose signal.",
            "numbered_entry_fraction": "Share of component lines beginning with a list number; repeated entry-boundary structure, not citation content.",
            "author_start_fraction": "Share of component lines beginning with one of the mutually exclusive frozen author-list shapes.",
            "publication_tail_fraction": "Share of component lines containing a specific identifier, volume, page, publisher, or place-publisher tail.",
            "document_candidate_count": "Number of distinct permissive spans proposed in the document across all frozen deletion biases.",
            "document_proposal_island_count": "Number of disconnected regions covered by any permissive proposal in the document; many islands indicate scattered citation-like text rather than one large bibliography region.",
            "document_coverage_fraction": "Share of all document lines covered by this one component.",
            "outside_context_median_probability": "Typical frozen entry probability in the emitted lines immediately outside the component and within eight physical lines; low values support a section boundary.",
            "inside_minus_outside_probability": "Difference between typical frozen entry probability inside the component and immediately outside it; positive values indicate a bibliography-like box rather than uniform running text.",
        },
        "summaries": summaries,
        "chosen_false_documents": sorted(
            false_documents.values(),
            key=lambda row: (-row["false_line_count"], -row["max_score"]),
        ),
        "validation_opened": False,
    }
    _write_json(Path(args.output_path).resolve(), output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--variant", default="no_length")
    parser.add_argument(
        "--model-arm", choices=("logistic_l2", "monotonic_hgb"), default="logistic_l2"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

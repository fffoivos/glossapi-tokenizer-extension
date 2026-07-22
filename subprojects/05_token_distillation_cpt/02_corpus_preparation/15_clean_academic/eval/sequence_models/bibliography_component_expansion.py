#!/usr/bin/env python3
"""Expand high-precision train-OOF bibliography cores through proposals.

Lower-scored proposals may extend an already accepted core but can never start
a block.  This tests block coherence directly without adding line regexes or
opening validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import _load_quality_exclusions, _span_iou
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-component-core-expansion-oof-v1"


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


def _save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def select_core_spans(
    starts: np.ndarray,
    ends: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
) -> list[tuple[int, int]]:
    """Select non-overlapping cores with the existing score-first rule."""

    chosen: list[tuple[int, int]] = []
    for row in sorted(
        rows,
        key=lambda index: (
            -float(scores[index]),
            -(int(ends[index]) - int(starts[index])),
            int(starts[index]),
        ),
    ):
        span = (int(starts[row]), int(ends[row]))
        if any(_span_iou(span, previous) > 0 for previous in chosen):
            continue
        chosen.append(span)
    return chosen


def expand_core_spans(
    cores: Sequence[tuple[int, int]],
    proposals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return proposal-graph components containing at least one core."""

    expanded: list[tuple[int, int]] = []
    proposal_rows = list(dict.fromkeys(proposals))
    for core in cores:
        start, end = core
        changed = True
        while changed:
            changed = False
            for proposal_start, proposal_end in proposal_rows:
                if proposal_end < start or proposal_start > end:
                    continue
                new_start, new_end = min(start, proposal_start), max(end, proposal_end)
                if (new_start, new_end) != (start, end):
                    start, end = new_start, new_end
                    changed = True
        expanded.append((start, end))
    if not expanded:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(expanded):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def decode_expansion(
    table: Any,
    document_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    scores: np.ndarray,
    probability: np.ndarray,
    config: BlockConfig,
    qualified_documents: set[int],
    *,
    core_threshold: float,
    expansion_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if expansion_threshold > core_threshold:
        raise ValueError("expansion threshold cannot exceed core threshold")
    core_prediction = np.zeros(len(table.targets), dtype=bool)
    expanded_prediction = np.zeros(len(table.targets), dtype=bool)
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        doc_rows = document_indices == document_index
        core_rows = np.flatnonzero(doc_rows & (scores >= core_threshold))
        pool_rows = np.flatnonzero(doc_rows & (scores >= expansion_threshold))
        cores = select_core_spans(starts, ends, scores, core_rows)
        proposals = [(int(starts[row]), int(ends[row])) for row in pool_rows]
        expanded = expand_core_spans(cores, proposals)
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local_core = np.zeros(doc_end - doc_start, dtype=bool)
        local_expanded = np.zeros(doc_end - doc_start, dtype=bool)
        for start, end in cores:
            local_core[start : end + 1] = True
        for start, end in expanded:
            local_expanded[start : end + 1] = True
        local_core = attach_h0_document(
            local_core,
            probability[doc_start:doc_end],
            table.header_kinds[doc_start:doc_end],
            table.abs_indices[doc_start:doc_end],
            config,
        )
        local_expanded = attach_h0_document(
            local_expanded,
            probability[doc_start:doc_end],
            table.header_kinds[doc_start:doc_end],
            table.abs_indices[doc_start:doc_end],
            config,
        )
        core_prediction[doc_start:doc_end] = local_core
        expanded_prediction[doc_start:doc_end] = local_expanded
    return core_prediction, expanded_prediction


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        row["model_arm"] == "logistic_l2",
        -float(row["core_threshold"]),
        -float(row["expansion_threshold"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    component_dir = Path(args.component_dir).resolve()
    component_report_path = component_dir / "component_gate_oof_report.json"
    component_report = json.loads(component_report_path.read_text(encoding="utf-8"))
    if component_report.get("validation_opened") is not False:
        raise ValueError("core expansion requires validation-isolated component scores")
    probability_path = Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    config = BlockConfig(**component_report["block_config"])
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}
    for variant in args.variants:
        prefix = component_dir / variant
        document_indices = np.load(
            prefix.with_suffix(".document_indices.npy"), mmap_mode="r", allow_pickle=False
        )
        starts = np.load(prefix.with_suffix(".starts.npy"), mmap_mode="r", allow_pickle=False)
        ends = np.load(prefix.with_suffix(".ends.npy"), mmap_mode="r", allow_pickle=False)
        for model_arm in args.model_arms:
            model_report = component_report["model_reports"][f"{variant}:{model_arm}"]
            if not bool(model_report["direction_contract_satisfied"]):
                continue
            scores = np.load(
                component_dir / f"{variant}.{model_arm}.oof_scores.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            arrays[(variant, model_arm)] = (
                document_indices, starts, ends, scores
            )
            for core_threshold in args.core_thresholds:
                for expansion_threshold in args.expansion_thresholds:
                    if expansion_threshold > core_threshold:
                        continue
                    core, expanded = decode_expansion(
                        table,
                        document_indices,
                        starts,
                        ends,
                        scores,
                        probability,
                        config,
                        qualified_documents,
                        core_threshold=float(core_threshold),
                        expansion_threshold=float(expansion_threshold),
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "model_arm": model_arm,
                            "core_threshold": float(core_threshold),
                            "expansion_threshold": float(expansion_threshold),
                            "added_line_count": int(np.count_nonzero(expanded & ~core)),
                            "core_metrics": evaluate_prediction(
                                table, core, document_subset=qualified_documents
                            ),
                            "metrics": evaluate_prediction(
                                table, expanded, document_subset=qualified_documents
                            ),
                        }
                    )
    safe_rows = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe_rows, key=_candidate_key) if safe_rows else None
    precision95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    highest = max(rows, key=_candidate_key)
    highest_p95 = max(precision95, key=_candidate_key) if precision95 else None
    if selected is not None:
        document_indices, starts, ends, scores = arrays[
            (selected["variant"], selected["model_arm"])
        ]
        _, prediction = decode_expansion(
            table,
            document_indices,
            starts,
            ends,
            scores,
            probability,
            config,
            qualified_documents,
            core_threshold=float(selected["core_threshold"]),
            expansion_threshold=float(selected["expansion_threshold"]),
        )
        _save_array(output_dir / "selected_expanded_oof_prediction.npy", prediction)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "algorithm_reference": {
            "core": "A component at or above the high core threshold may start a bibliography block.",
            "expansion": "A lower-scored proposal may only extend a core through an overlap-connected proposal chain; it can never start a block.",
            "attachment": "Exact H0 headers are attached only after core selection and expansion.",
        },
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "candidates": rows,
        "selected": selected,
        "diagnostic_highest_recall_candidate": highest,
        "diagnostic_highest_recall_at_line_precision_0_95": highest_p95,
        "selection_rule": "require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "component_gate_report": _sha256(component_report_path),
            "line_oof_probability": _sha256(probability_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "component_expansion_oof_report.json", report)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**report, "outputs": outputs})
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--variants", nargs="+", default=("no_length", "no_length_or_position"))
    parser.add_argument("--model-arms", nargs="+", default=("logistic_l2", "monotonic_hgb"))
    parser.add_argument("--core-thresholds", type=float, nargs="+", default=(0.95, 0.975, 0.99, 0.995))
    parser.add_argument("--expansion-thresholds", type=float, nargs="+", default=(0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

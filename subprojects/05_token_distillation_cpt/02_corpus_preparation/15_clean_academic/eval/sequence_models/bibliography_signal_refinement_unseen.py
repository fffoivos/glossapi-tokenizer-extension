#!/usr/bin/env python3
"""Apply frozen train-OOF refinements to the unseen human-review packet.

The review labels are used only after the refinement report has selected its
policies.  They are never passed to a model or threshold-selection routine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .bibliography_deterministic_roles import ROLE_NAMES, _analyze_document
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    blocks_from_mask,
    decode_b0_document,
)
from .bibliography_entry_dataset import HEADER_EXACT, HEADER_NONE, SUBHEADER_EXACT
from .bibliography_signal_refinement import (
    FEATURE_NAMES,
    SCHEMA_VERSION as REFINEMENT_SCHEMA,
    _split_span_at_headings,
    component_feature_vector,
    refine_outer_edges_asymmetric,
)
from .deterministic_structure import BibRole, analyze_bib_line


SCHEMA_VERSION = "bibliography-signal-refinement-unseen-diagnostic-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_kinds(lines: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.full(len(lines), HEADER_NONE, dtype=np.uint8)
    for index, line in enumerate(lines):
        evidence = analyze_bib_line(str(line["text"]), int(line["abs_idx"]))
        if evidence.role == BibRole.HEADING:
            result[index] = HEADER_EXACT
        elif evidence.role == BibRole.SUBHEADING:
            result[index] = SUBHEADER_EXACT
    return result


def _core_prediction(
    base: np.ndarray,
    signal: np.ndarray,
    entry: np.ndarray,
    headers: np.ndarray,
    absolute: np.ndarray,
    config: BlockConfig,
) -> np.ndarray:
    core = np.zeros(len(base), dtype=bool)
    core_config = replace(config, adjacent_expansion=0)
    for start, end in blocks_from_mask(base, absolute):
        local = decode_b0_document(
            signal[start : end + 1],
            np.zeros(end - start + 1, dtype=np.uint8),
            absolute[start : end + 1],
            core_config,
        )
        local = attach_h0_document(
            local,
            entry[start : end + 1],
            headers[start : end + 1],
            absolute[start : end + 1],
            core_config,
        )
        core[start : end + 1] = local
    return core


def _edge_prediction(
    base: np.ndarray,
    core: np.ndarray,
    roles: np.ndarray,
    absolute: np.ndarray,
    selected: Mapping[str, Any] | None,
) -> np.ndarray:
    if selected is None:
        return base.copy()
    table = SimpleNamespace(
        targets=np.zeros(len(base), dtype=np.int8),
        documents=({"line_start": 0, "line_end": len(base)},),
        abs_indices=absolute,
    )
    left_role_names = selected.get("left_role_names")
    right_role_names = selected.get("right_role_names")
    if left_role_names is None or right_role_names is None:
        raise ValueError("selected edge row lacks its frozen role-name expansions")
    return refine_outer_edges_asymmetric(
        table,
        base,
        core,
        roles,
        left_role_names=tuple(left_role_names),
        right_role_names=tuple(right_role_names),
        qualified_documents={0},
    )


def _component_prediction(
    base: np.ndarray,
    signal: np.ndarray,
    entry: np.ndarray,
    roles: np.ndarray,
    headers: np.ndarray,
    absolute: np.ndarray,
    config: BlockConfig,
    selected: Mapping[str, Any] | None,
    model: Any | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if selected is None or model is None:
        return base.copy(), []
    generic_index = ROLE_NAMES.index("generic_markdown_heading")
    spans: list[tuple[int, int]] = []
    for start, end in blocks_from_mask(base, absolute):
        spans.extend(
            [(start, end)]
            if selected["variant"] == "whole"
            else _split_span_at_headings(
                start, end, roles[:, generic_index] > 0
            )
        )
    features = np.stack(
        [
            component_feature_vector(
                signal,
                entry,
                roles,
                headers,
                start=start,
                end=end,
                config=config,
            )
            for start, end in spans
        ]
    ) if spans else np.zeros((0, 0), dtype=np.float32)
    scores = model.predict_proba(features)[:, 1] if len(spans) else np.asarray([])
    prediction = np.zeros(len(base), dtype=bool)
    decisions = []
    threshold = float(selected["threshold"])
    heading_threshold = float(selected["heading_threshold"])
    heading_feature = FEATURE_NAMES.index("starts_with_structural_non_bib_heading")
    for row, ((start, end), score) in enumerate(zip(spans, scores, strict=True)):
        minimum = heading_threshold if features[row, heading_feature] > 0.5 else threshold
        accepted = float(score) >= minimum
        if accepted:
            prediction[start : end + 1] = True
        decisions.append(
            {
                "local_start": start,
                "local_end": end,
                "score": float(score),
                "accepted": accepted,
            }
        )
    prediction = attach_h0_document(
        prediction, entry, headers, absolute, config
    )
    prediction &= base
    return prediction, decisions


def _line_keys(document_id: str, lines: Sequence[Mapping[str, Any]], mask: np.ndarray) -> set[str]:
    return {
        f"{document_id}:{int(line['abs_idx'])}"
        for line, enabled in zip(lines, mask, strict=True)
        if enabled
    }


def _summarize(
    original: set[str],
    candidate: set[str],
    wrong: set[str],
    weird_keys: set[str],
    whole_wrong: set[str],
) -> dict[str, Any]:
    removed = original - candidate
    marked_removed = removed & wrong
    unmarked_removed = removed - wrong
    ordinary_original = original - weird_keys
    ordinary_removed = removed - weird_keys
    return {
        "original_predicted_line_count": len(original),
        "retained_predicted_line_count": len(candidate),
        "removed_predicted_line_count": len(removed),
        "marked_wrong_removed": len(marked_removed),
        "marked_wrong_retained": len(wrong & candidate),
        "marked_whole_block_wrong_removed": len(marked_removed & whole_wrong),
        "marked_boundary_wrong_removed": len(marked_removed - whole_wrong),
        "unmarked_prediction_removed_review_risk": len(unmarked_removed),
        "ordinary_non_weird_original_prediction_count": len(ordinary_original),
        "ordinary_non_weird_removed_prediction_count": len(ordinary_removed),
        "ordinary_non_weird_marked_wrong_removed": len(marked_removed - weird_keys),
        "removed_keys": sorted(removed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet).resolve()
    review_path = Path(args.review).resolve()
    refinement_dir = Path(args.refinement_dir).resolve()
    refinement_path = refinement_dir / "signal_refinement_oof_report.json"
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    if (
        refinement.get("schema_version") != REFINEMENT_SCHEMA
        or refinement.get("validation_opened") is not False
        or refinement.get("unseen_review_used_for_fitting_or_selection") is not False
    ):
        raise ValueError("unseen diagnostic requires a review-blind train-OOF refinement")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    wrong = set(map(str, review["wrong_predicted_lines"]))
    weird_documents = set(map(str, review["weird_documents"]))
    config = BlockConfig(**refinement["frozen_decoder_config"])
    edge_selected = refinement["edge_experiment"].get("selected")
    if edge_selected is not None:
        edge_selected = {
            **edge_selected,
            "left_role_names": refinement["edge_experiment"]["role_arms"][edge_selected["left_role_arm"]],
            "right_role_names": refinement["edge_experiment"]["role_arms"][edge_selected["right_role_arm"]],
        }
    component_selected = refinement["component_experiment"].get("selected")
    model = None
    if component_selected is not None:
        model_path = refinement_dir / "models" / (
            f"{component_selected['variant']}.{component_selected['model_arm']}.full.pkl"
        )
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

    all_original: set[str] = set()
    all_edge: set[str] = set()
    all_component: set[str] = set()
    all_combined: set[str] = set()
    weird_keys: set[str] = set()
    whole_wrong: set[str] = set()
    document_rows = []
    for document in packet["documents"]:
        document_id = str(document["document_id"])
        lines = document["lines"]
        absolute = np.asarray([int(line["abs_idx"]) for line in lines], dtype=np.uint32)
        signal = np.asarray([float(line["signal_probability"]) for line in lines])
        entry = np.asarray([float(line["entry_probability"]) for line in lines])
        base = np.asarray([bool(line["predicted_bib"]) for line in lines])
        _doc, roles, _counts = _analyze_document((document_id, lines))
        headers = _header_kinds(lines)
        core = _core_prediction(base, signal, entry, headers, absolute, config)
        edge = _edge_prediction(base, core, roles, absolute, edge_selected)
        component, decisions = _component_prediction(
            base,
            signal,
            entry,
            roles,
            headers,
            absolute,
            config,
            component_selected,
            model,
        )
        combined = edge & component
        keys = {
            "original": _line_keys(document_id, lines, base),
            "edge": _line_keys(document_id, lines, edge),
            "component": _line_keys(document_id, lines, component),
            "combined": _line_keys(document_id, lines, combined),
        }
        all_original |= keys["original"]
        all_edge |= keys["edge"]
        all_component |= keys["component"]
        all_combined |= keys["combined"]
        if document_id in weird_documents:
            weird_keys |= keys["original"]
        for start, end in blocks_from_mask(base, absolute):
            block_keys = _line_keys(document_id, lines[start : end + 1], base[start : end + 1])
            if block_keys and block_keys <= wrong:
                whole_wrong |= block_keys
        document_rows.append(
            {
                "document_id": document_id,
                "source": document.get("source"),
                "weird": document_id in weird_documents,
                "original_prediction_count": len(keys["original"]),
                "marked_wrong_count": len(keys["original"] & wrong),
                "edge_removed": len(keys["original"] - keys["edge"]),
                "component_removed": len(keys["original"] - keys["component"]),
                "combined_removed": len(keys["original"] - keys["combined"]),
                "combined_marked_wrong_removed": len((keys["original"] - keys["combined"]) & wrong),
                "component_decisions": decisions,
            }
        )
    if not wrong <= all_original:
        raise ValueError("one or more human WRONG keys do not map to frozen predictions")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_frozen_refinement_on_unseen_human_review",
        "interpretation": "WRONG labels cover only user-marked false positives; unmarked removals are review risk, not established false negatives",
        "edge_selected": edge_selected,
        "component_selected": component_selected,
        "review_counts": {
            "wrong_predicted_lines": len(wrong),
            "whole_block_wrong_lines": len(whole_wrong),
            "boundary_wrong_lines": len(wrong - whole_wrong),
            "weird_documents": len(weird_documents),
        },
        "diagnostics": {
            "baseline": _summarize(all_original, all_original, wrong, weird_keys, whole_wrong),
            "edge": _summarize(all_original, all_edge, wrong, weird_keys, whole_wrong),
            "component": _summarize(all_original, all_component, wrong, weird_keys, whole_wrong),
            "combined": _summarize(all_original, all_combined, wrong, weird_keys, whole_wrong),
        },
        "documents": document_rows,
        "input_hashes": {
            "packet": _sha256(packet_path),
            "review": _sha256(review_path),
            "refinement_report": _sha256(refinement_path),
        },
        "human_review_used_for_fitting_or_selection": False,
        "development_review_informed_experiment_design": bool(
            refinement.get("development_review_informed_experiment_design")
        ),
        "independent_fresh_unseen_evaluation_required": True,
    }
    output_path = Path(args.output).resolve()
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--refinement-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

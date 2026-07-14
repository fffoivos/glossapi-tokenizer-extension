#!/usr/bin/env python3
"""Audit and apply an exact auxiliary-section veto to train-OOF components.

Only headings in the pre-existing exact auxiliary-scope lexicon are eligible.
Generic, body, CV, notes, and bibliography headings are not included.  A veto
can reject a proposed component but can never create or expand a deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_deterministic_roles import AUXILIARY_SCOPE_HEADINGS
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import (
    CandidateSet,
    _load_quality_exclusions,
    _span_iou,
)
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .deterministic_structure import _heading_key


SCHEMA_VERSION = "bibliography-auxiliary-scope-veto-oof-v1"


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


def materialize_auxiliary_headings(table: Any, input_path: Path) -> np.ndarray:
    expected = {
        str(document["document_id"]): (
            int(document["line_start"]),
            int(document["line_end"]),
        )
        for document in table.documents
    }
    result = np.zeros(len(table.targets), dtype=bool)
    completed: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            # Fail closed before parsing: validation rows are never deserialized.
            if '"split": "train"' not in raw:
                continue
            row = json.loads(raw)
            document_id = str(row.get("document_id"))
            if document_id not in expected:
                continue
            if document_id in completed:
                raise ValueError("source contains a duplicate train document")
            start, end = expected[document_id]
            lines = row.get("lines")
            if not isinstance(lines, list) or len(lines) != end - start:
                raise ValueError(f"{document_id}: source/table line alignment failure")
            for offset, line in enumerate(lines):
                text = line.get("text") if isinstance(line, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"{document_id}: invalid source line")
                result[start + offset] = (
                    _heading_key(text) in AUXILIARY_SCOPE_HEADINGS
                )
            completed.add(document_id)
    if completed != set(expected):
        raise ValueError("auxiliary-scope materialization is incomplete")
    return result


def has_auxiliary_scope(
    auxiliary: np.ndarray,
    abs_indices: np.ndarray,
    start: int,
    *,
    window: int,
) -> bool:
    """Return whether an exact auxiliary heading immediately scopes a span."""

    if not 0 <= start < len(auxiliary) or len(auxiliary) != len(abs_indices):
        raise ValueError("auxiliary scope arrays or start are invalid")
    for index in range(max(0, start - window), start + 1):
        if (
            bool(auxiliary[index])
            and 0 <= int(abs_indices[start]) - int(abs_indices[index]) <= window
        ):
            return True
    return False


def _load_candidates(root: Path, variant: str) -> CandidateSet:
    prefix = root / variant
    return CandidateSet(
        features=np.load(prefix.with_suffix(".features.npy"), mmap_mode="r", allow_pickle=False),
        document_indices=np.load(prefix.with_suffix(".document_indices.npy"), mmap_mode="r", allow_pickle=False),
        starts=np.load(prefix.with_suffix(".starts.npy"), mmap_mode="r", allow_pickle=False),
        ends=np.load(prefix.with_suffix(".ends.npy"), mmap_mode="r", allow_pickle=False),
        labels=np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r", allow_pickle=False),
    )


def decode_with_auxiliary_veto(
    table: Any,
    candidates: CandidateSet,
    scores: np.ndarray,
    probability: np.ndarray,
    auxiliary: np.ndarray,
    config: BlockConfig,
    *,
    threshold: float,
    qualified_documents: set[int],
    apply_veto: bool,
) -> tuple[np.ndarray, list[int]]:
    prediction = np.zeros(len(table.targets), dtype=bool)
    vetoed_rows: list[int] = []
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        document_rows = np.flatnonzero(
            (candidates.document_indices == document_index) & (scores >= threshold)
        )
        doc_start, doc_end = int(document["line_start"]), int(document["line_end"])
        local_auxiliary = auxiliary[doc_start:doc_end]
        local_absolute = table.abs_indices[doc_start:doc_end]
        chosen: list[tuple[int, int]] = []
        for row in sorted(
            document_rows,
            key=lambda index: (
                -float(scores[index]),
                -(int(candidates.ends[index]) - int(candidates.starts[index])),
                int(candidates.starts[index]),
            ),
        ):
            span = (int(candidates.starts[row]), int(candidates.ends[row]))
            if apply_veto and has_auxiliary_scope(
                local_auxiliary,
                local_absolute,
                span[0],
                window=config.header_window,
            ):
                vetoed_rows.append(int(row))
                continue
            if any(_span_iou(span, previous) > 0 for previous in chosen):
                continue
            chosen.append(span)
        local = np.zeros(doc_end - doc_start, dtype=bool)
        for start, end in chosen:
            local[start : end + 1] = True
        local = attach_h0_document(
            local,
            probability[doc_start:doc_end],
            table.header_kinds[doc_start:doc_end],
            local_absolute,
            config,
        )
        prediction[doc_start:doc_end] = local
    return prediction, vetoed_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    component_root = Path(args.component_dir).resolve()
    component_report_path = component_root / args.report_name
    component_report = json.loads(
        component_report_path.read_text(encoding="utf-8")
    )
    if component_report.get("validation_opened") is not False:
        raise ValueError("auxiliary veto requires validation-isolated components")
    model_key = f"{args.variant}:{args.model_arm}"
    if not component_report["model_reports"][model_key][
        "direction_contract_satisfied"
    ]:
        raise ValueError("auxiliary veto rejects a direction-unstable model")
    config = BlockConfig(**component_report["block_config"])
    candidates = _load_candidates(component_root, args.variant)
    scores_path = component_root / f"{args.variant}.{args.model_arm}.oof_scores.npy"
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    probability_path = (
        Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    )
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    auxiliary = materialize_auxiliary_headings(table, Path(args.input).resolve())
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
    _save_array(output_dir / "auxiliary_scope_heading.npy", auxiliary)

    rows = []
    selected_predictions: dict[float, np.ndarray] = {}
    all_vetoed: set[int] = set()
    for threshold in sorted(set(float(value) for value in args.thresholds)):
        baseline, _ = decode_with_auxiliary_veto(
            table,
            candidates,
            scores,
            probability,
            auxiliary,
            config,
            threshold=threshold,
            qualified_documents=qualified_documents,
            apply_veto=False,
        )
        prediction, vetoed = decode_with_auxiliary_veto(
            table,
            candidates,
            scores,
            probability,
            auxiliary,
            config,
            threshold=threshold,
            qualified_documents=qualified_documents,
            apply_veto=True,
        )
        all_vetoed.update(vetoed)
        selected_predictions[threshold] = prediction
        rows.append(
            {
                "threshold": threshold,
                "vetoed_candidate_count": len(vetoed),
                "baseline_metrics": evaluate_prediction(
                    table, baseline, document_subset=qualified_documents
                ),
                "metrics": evaluate_prediction(
                    table, prediction, document_subset=qualified_documents
                ),
            }
        )
    safe = [row for row in rows if is_safe_candidate(row)]
    selected = max(
        safe,
        key=lambda row: (
            row["metrics"]["token_recall"],
            row["metrics"]["line_recall"],
            row["metrics"]["token_precision"],
        ),
    ) if safe else None
    if selected is not None:
        _save_array(
            output_dir / "selected_oof_prediction.npy",
            selected_predictions[float(selected["threshold"])],
        )
    vetoed_labels = candidates.labels[np.asarray(sorted(all_vetoed), dtype=np.int64)]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "variant": args.variant,
        "model_arm": args.model_arm,
        "scope_rule": "veto only when a pre-existing exact auxiliary-scope heading occurs on the candidate start or within the preceding two physical lines",
        "auxiliary_scope_headings": sorted(AUXILIARY_SCOPE_HEADINGS),
        "auxiliary_heading_line_count": int(np.count_nonzero(auxiliary)),
        "auxiliary_heading_inside_silver_bib_count": int(
            np.count_nonzero(
                auxiliary & (table.original_labels == LABEL_TO_ID["BIB"])
            )
        ),
        "unique_vetoed_candidate_count": len(all_vetoed),
        "vetoed_candidate_supervision": {
            "positive": int(np.count_nonzero(vetoed_labels == 1)),
            "mixed_masked": int(np.count_nonzero(vetoed_labels == -1)),
            "negative": int(np.count_nonzero(vetoed_labels == 0)),
        },
        "candidates": rows,
        "safe_candidate_count": len(safe),
        "selected": selected,
        "selection_rule": "require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "source": _sha256(Path(args.input).resolve()),
            "component_report": _sha256(component_report_path),
            "component_scores": _sha256(scores_path),
            "line_oof_probability": _sha256(probability_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "auxiliary_scope_veto_oof_report.json", result)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--component-dir", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-arm", required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995),
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

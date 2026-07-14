#!/usr/bin/env python3
"""Refine one frozen component-score threshold on grouped train OOF data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import (
    CandidateSet,
    _load_quality_exclusions,
    decode_candidates,
)
from .bibliography_entry_models import load_table


SCHEMA_VERSION = "bibliography-component-threshold-calibration-oof-v1"


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


def _load_candidates(root: Path, variant: str) -> CandidateSet:
    prefix = root / variant
    return CandidateSet(
        features=np.load(prefix.with_suffix(".features.npy"), mmap_mode="r", allow_pickle=False),
        document_indices=np.load(prefix.with_suffix(".document_indices.npy"), mmap_mode="r", allow_pickle=False),
        starts=np.load(prefix.with_suffix(".starts.npy"), mmap_mode="r", allow_pickle=False),
        ends=np.load(prefix.with_suffix(".ends.npy"), mmap_mode="r", allow_pickle=False),
        labels=np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r", allow_pickle=False),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    component_root = Path(args.component_dir).resolve()
    report_path = component_root / args.report_name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validation_opened") is not False:
        raise ValueError("threshold calibration requires validation-isolated scores")
    model_key = f"{args.variant}:{args.model_arm}"
    model_report = report["model_reports"][model_key]
    if not bool(model_report["direction_contract_satisfied"]):
        raise ValueError("threshold calibration rejects a direction-unstable model")
    candidates = _load_candidates(component_root, args.variant)
    scores_path = component_root / f"{args.variant}.{args.model_arm}.oof_scores.npy"
    scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
    if len(scores) != len(candidates.labels):
        raise ValueError("component scores do not align with candidate rows")
    line_probability_path = (
        Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    )
    probability = np.load(line_probability_path, mmap_mode="r", allow_pickle=False)
    config = BlockConfig(**report["block_config"])
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

    rows = []
    predictions: dict[float, np.ndarray] = {}
    for threshold in sorted(set(float(value) for value in args.thresholds)):
        prediction = decode_candidates(
            table,
            candidates,
            scores,
            probability,
            config,
            threshold=threshold,
            qualified_documents=qualified_documents,
        )
        predictions[threshold] = prediction
        rows.append(
            {
                "threshold": threshold,
                "metrics": evaluate_prediction(
                    table, prediction, document_subset=qualified_documents
                ),
            }
        )
    safe = [row for row in rows if is_safe_candidate(row)]
    selected = (
        min(safe, key=lambda row: float(row["threshold"])) if safe else None
    )
    if selected is not None:
        _save_array(
            output_dir / "selected_oof_prediction.npy",
            predictions[float(selected["threshold"])],
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_threshold_met_safety_gate"
        ),
        "variant": args.variant,
        "model_arm": args.model_arm,
        "thresholds": rows,
        "safe_threshold_count": len(safe),
        "selected": selected,
        "selection_rule": "choose the lowest predeclared threshold satisfying line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document",
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "component_report": _sha256(report_path),
            "component_scores": _sha256(scores_path),
            "line_oof_probability": _sha256(line_probability_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "component_calibration_oof_report.json", result)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_json(output_dir / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=tuple(round(0.975 + index * 0.001, 3) for index in range(21)),
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

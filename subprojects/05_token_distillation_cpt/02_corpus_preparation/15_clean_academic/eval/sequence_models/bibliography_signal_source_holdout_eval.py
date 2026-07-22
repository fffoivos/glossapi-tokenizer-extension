#!/usr/bin/env python3
"""Evaluate the frozen bibliography block decoder with source-held-out TCN scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_signal_block_decode import (
    SCHEMA_VERSION as BLOCK_SCHEMA,
    decode_signal_blocks,
)
from .bibliography_signal_tcn import SCHEMA_VERSION as SIGNAL_SCHEMA
from .bibliography_signal_validation import select_train_recall_candidate
from .bibliography_source_holdout_table import SCHEMA_VERSION as TABLE_SCHEMA


SCHEMA_VERSION = "bibliography-signal-source-holdout-eval-v1"


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


def validate_source_folds(table: Any, folds_packet: dict[str, Any]) -> dict[str, int]:
    if (
        folds_packet.get("schema_version") != TABLE_SCHEMA
        or folds_packet.get("strategy") != "leave_one_complete_source_out"
    ):
        raise ValueError("table is not a source-held-out table")
    expected = {str(key): int(value) for key, value in folds_packet["source_to_fold"].items()}
    observed: dict[str, set[int]] = {}
    fold_sources: dict[int, set[str]] = {}
    for document in table.documents:
        source, fold = str(document["source"]), int(document["fold"])
        observed.setdefault(source, set()).add(fold)
        fold_sources.setdefault(fold, set()).add(source)
    if any(len(values) != 1 for values in observed.values()) or any(
        len(values) != 1 for values in fold_sources.values()
    ):
        raise ValueError("a source is split across folds or a fold mixes sources")
    flattened = {source: next(iter(values)) for source, values in observed.items()}
    if flattened != expected or len(expected) != int(table.manifest["n_folds"]):
        raise ValueError("source fold mapping disagrees with the table manifest")
    return expected


def run(args: argparse.Namespace) -> dict[str, Any]:
    table_root = Path(args.table_dir).resolve()
    table = load_table(table_root, expected_split="train")
    folds_path = table_root / "folds.json"
    source_to_fold = validate_source_folds(
        table, json.loads(folds_path.read_text(encoding="utf-8"))
    )
    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    if (
        signal_report.get("schema_version") != SIGNAL_SCHEMA
        or signal_report.get("validation_opened") is not False
        or len(signal_report.get("folds", ())) != len(source_to_fold)
        or signal_report.get("input_hashes", {}).get("table_receipt")
        != _sha256(table_root / "receipt.json")
    ):
        raise ValueError("signal scores are not aligned source-held-out predictions")
    signal_path = signal_root / "signal_tcn_oof_probability.npy"
    signal_probability = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    line_path = Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    line_probability = np.load(line_path, mmap_mode="r", allow_pickle=False)
    if not (len(signal_probability) == len(line_probability) == len(table.targets)):
        raise ValueError("source-held-out signal arrays are not line aligned")
    block_report_path = (
        Path(args.frozen_recall_block_dir).resolve()
        / "signal_block_decode_oof_report.json"
    )
    block_report = json.loads(block_report_path.read_text(encoding="utf-8"))
    if (
        block_report.get("schema_version") != BLOCK_SCHEMA
        or block_report.get("validation_opened") is not False
    ):
        raise ValueError("decoder selection is not a frozen train-OOF report")
    frozen_row = select_train_recall_candidate(block_report)
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    qualified = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    input_path = Path(args.input).resolve()
    headings, scope = materialize_auxiliary_headings(table, input_path)
    if np.any(scope & (table.original_labels == LABEL_TO_ID["BIB"])):
        raise ValueError("source-held-out auxiliary scope overlaps silver bibliography")
    prediction, barrier_intervals = decode_signal_blocks(
        table,
        signal_probability,
        line_probability,
        scope,
        BlockConfig(**frozen_row["config"]),
        qualified_documents=qualified,
        apply_veto=True,
    )
    by_source = {}
    for source in sorted(source_to_fold):
        subset = {
            index
            for index in qualified
            if str(table.documents[index]["source"]) == source
        }
        by_source[source] = {
            "held_out_fold": source_to_fold[source],
            "metrics": evaluate_prediction(table, prediction, document_subset=subset),
        }
    output_root = Path(args.output_dir).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    _save_array(output_root / "source_holdout_prediction.npy", prediction)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_frozen_decoder_source_holdout_robustness_audit",
        "evidence_scope": "signal_TCN_leaves_one_complete_source_out; frozen entry probability remains document-OOF rather than source-held-out",
        "selection_eligible": False,
        "decoder_selection": "unchanged >=0.90-line-precision candidate selected on the original grouped train OOF report",
        "source_to_fold": source_to_fold,
        "frozen_decoder_row": frozen_row,
        "metrics": evaluate_prediction(table, prediction, document_subset=qualified),
        "by_held_out_source": by_source,
        "scope_barrier_interval_count": barrier_intervals,
        "auxiliary_heading_line_count": int(np.count_nonzero(headings)),
        "auxiliary_scope_line_count": int(np.count_nonzero(scope)),
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified),
        },
        "input_hashes": {
            "source": _sha256(input_path),
            "table_receipt": _sha256(table_root / "receipt.json"),
            "folds": _sha256(folds_path),
            "signal_report": _sha256(signal_report_path),
            "signal_probability": _sha256(signal_path),
            "line_probability": _sha256(line_path),
            "frozen_decoder_report": _sha256(block_report_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_root / "source_holdout_report.json", result)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    _write_json(output_root / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--frozen-recall-block-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

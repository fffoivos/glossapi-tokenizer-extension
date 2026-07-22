#!/usr/bin/env python3
"""Decode train-OOF signal-TCN scores as anchored bibliography blocks.

A high score on one line is insufficient.  Several high-confidence lines must
establish a local region; weak lines are then included only between or directly
beside established anchors.  This is the explicit second level requested for
long/weak bibliography entries.  No model is refitted and validation is not
accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_auxiliary_scope_veto import materialize_auxiliary_headings
from .bibliography_entry_blocks import (
    BlockConfig,
    attach_h0_document,
    decode_b0_document,
    evaluate_prediction,
)
from .bibliography_entry_coherence import is_safe_candidate
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_signal_tcn import SCHEMA_VERSION as SIGNAL_SCHEMA


SCHEMA_VERSION = "bibliography-signal-block-decode-oof-v2"


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


def decode_signal_blocks(
    table: Any,
    signal_probability: np.ndarray,
    frozen_entry_probability: np.ndarray,
    auxiliary_scope: np.ndarray,
    config: BlockConfig,
    *,
    qualified_documents: set[int],
    apply_veto: bool,
) -> tuple[np.ndarray, int]:
    """Establish anchored regions between exact scope barriers and attach H0."""

    prediction = np.zeros(len(table.targets), dtype=bool)
    scope_barrier_intervals = 0
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local_absolute = table.abs_indices[start:end]
        local_scope = (
            np.asarray(auxiliary_scope[start:end], dtype=bool)
            if apply_veto
            else np.zeros(end - start, dtype=bool)
        )
        local = np.zeros(end - start, dtype=bool)
        segment_start = 0
        while segment_start < len(local):
            if local_scope[segment_start]:
                scope_barrier_intervals += 1
                while (
                    segment_start < len(local) and local_scope[segment_start]
                ):
                    segment_start += 1
                continue
            segment_end = segment_start + 1
            while segment_end < len(local) and not local_scope[segment_end]:
                segment_end += 1
            # Length is intentionally absent from this decoder.  The all-zero
            # placeholder makes every line eligible as an anchor by length,
            # while the TCN score and multi-anchor rule establish the block.
            segment_prediction = decode_b0_document(
                signal_probability[start + segment_start : start + segment_end],
                np.zeros(segment_end - segment_start, dtype=np.uint8),
                local_absolute[segment_start:segment_end],
                config,
            )
            # Header attachment is also segment-local.  Running H0 on the
            # whole document would let a heading jump across a short scope
            # interval even though bibliography anchors cannot cross it.
            local[segment_start:segment_end] = attach_h0_document(
                segment_prediction,
                frozen_entry_probability[
                    start + segment_start : start + segment_end
                ],
                table.header_kinds[start + segment_start : start + segment_end],
                local_absolute[segment_start:segment_end],
                config,
            )
            segment_start = segment_end
        # This assertion-by-assignment also protects against a future decoder
        # implementation accidentally emitting a scope line.
        local[local_scope] = False
        prediction[start:end] = local
    return prediction, scope_barrier_intervals


def _evaluate_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        table_dir,
        signal_probability_path,
        frozen_probability_path,
        scope_path,
        qualified_documents,
        config_payload,
    ) = task
    table = load_table(table_dir, expected_split="train")
    signal_probability = np.load(
        signal_probability_path, mmap_mode="r", allow_pickle=False
    )
    frozen_probability = np.load(
        frozen_probability_path, mmap_mode="r", allow_pickle=False
    )
    auxiliary_scope = np.load(scope_path, mmap_mode="r", allow_pickle=False)
    config = BlockConfig(**config_payload)
    prediction, scope_barrier_intervals = decode_signal_blocks(
        table,
        signal_probability,
        frozen_probability,
        auxiliary_scope,
        config,
        qualified_documents=set(qualified_documents),
        apply_veto=True,
    )
    return {
        "config": asdict(config),
        "scope_barrier_interval_count": scope_barrier_intervals,
        "metrics": evaluate_prediction(
            table, prediction, document_subset=set(qualified_documents)
        ),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["metrics"]
    config = row["config"]
    return (
        float(metrics["token_recall"]),
        float(metrics["line_recall"]),
        float(metrics["token_precision"]),
        -float(config["anchor_probability"]),
        -float(config["inside_probability"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    signal_root = Path(args.signal_tcn_dir).resolve()
    signal_report_path = signal_root / "signal_tcn_oof_report.json"
    signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    if (
        signal_report.get("schema_version") != SIGNAL_SCHEMA
        or signal_report.get("validation_opened") is not False
    ):
        raise ValueError("block decode requires validation-isolated signal-TCN scores")
    signal_probability_path = signal_root / "signal_tcn_oof_probability.npy"
    line_root = Path(args.line_oof_dir).resolve()
    frozen_probability_path = line_root / f"{args.line_arm}.oof_probability.npy"
    signal_probability = np.load(
        signal_probability_path, mmap_mode="r", allow_pickle=False
    )
    frozen_probability = np.load(
        frozen_probability_path, mmap_mode="r", allow_pickle=False
    )
    if not (
        len(signal_probability) == len(frozen_probability) == len(table.targets)
    ):
        raise ValueError("line probability arrays are not aligned")
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
    auxiliary_headings, auxiliary_scope = materialize_auxiliary_headings(
        table, Path(args.input).resolve()
    )
    if np.any(
        auxiliary_scope & (table.original_labels == LABEL_TO_ID["BIB"])
    ):
        raise ValueError("audited auxiliary scope now overlaps silver bibliography")
    scope_path = output_dir / "auxiliary_scope_active.npy"
    _save_array(scope_path, auxiliary_scope)

    configs = [
        BlockConfig(
            anchor_probability=float(anchor),
            seed_length_limit=1,
            anchors_required=int(required),
            anchor_window=int(window),
            maximum_bridge_gap=int(bridge),
            inside_probability=float(inside),
            adjacent_expansion=int(expansion),
            header_window=2,
        )
        for anchor, required, window, bridge, inside, expansion in itertools.product(
            args.anchor_probabilities,
            args.anchors_required,
            args.anchor_windows,
            args.maximum_bridge_gaps,
            args.inside_probabilities,
            args.adjacent_expansions,
        )
        if float(inside) < float(anchor)
    ]
    tasks = [
        (
            str(Path(args.table_dir).resolve()),
            str(signal_probability_path),
            str(frozen_probability_path),
            str(scope_path),
            tuple(sorted(qualified_documents)),
            asdict(config),
        )
        for config in configs
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        rows = list(executor.map(_evaluate_task, tasks, chunksize=1))
    safe = [row for row in rows if is_safe_candidate(row)]
    selected = max(safe, key=_selection_key) if safe else None
    p95 = [row for row in rows if row["metrics"]["line_precision"] >= 0.95]
    diagnostic_p95 = max(p95, key=_selection_key) if p95 else None
    diagnostic_highest = max(rows, key=_selection_key)
    if selected is not None:
        selected_config = BlockConfig(**selected["config"])
        selected_prediction, _ = decode_signal_blocks(
            table,
            signal_probability,
            frozen_probability,
            auxiliary_scope,
            selected_config,
            qualified_documents=qualified_documents,
            apply_veto=True,
        )
        baseline_prediction, _ = decode_signal_blocks(
            table,
            signal_probability,
            frozen_probability,
            auxiliary_scope,
            selected_config,
            qualified_documents=qualified_documents,
            apply_veto=False,
        )
        _save_array(output_dir / "selected_oof_prediction.npy", selected_prediction)
        selected["baseline_metrics_without_scope_veto"] = evaluate_prediction(
            table, baseline_prediction, document_subset=qualified_documents
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed_train_oof_safety_gate_validation_unopened"
            if selected is not None
            else "research_only_no_candidate_met_safety_gate"
        ),
        "decoder_contract": "multiple high-confidence signal lines establish a block; weak/long lines may be filled only between or directly beside established anchors; exact negative scopes are hard walls decoded independently on each side; line length is not an input",
        "grid": {
            "anchor_probabilities": list(args.anchor_probabilities),
            "anchors_required": list(args.anchors_required),
            "anchor_windows": list(args.anchor_windows),
            "maximum_bridge_gaps": list(args.maximum_bridge_gaps),
            "inside_probabilities": list(args.inside_probabilities),
            "adjacent_expansions": list(args.adjacent_expansions),
            "candidate_count": len(configs),
        },
        "candidates": rows,
        "safe_candidate_count": len(safe),
        "selected": selected,
        "diagnostic_highest_recall_at_line_precision_0_95": diagnostic_p95,
        "diagnostic_highest_recall_candidate": diagnostic_highest,
        "selection_rule": "require line precision>=0.99 and <=0.02 spurious blocks per zero-BIB document; then maximize token and line recall",
        "auxiliary_heading_line_count": int(np.count_nonzero(auxiliary_headings)),
        "auxiliary_scope_line_count": int(np.count_nonzero(auxiliary_scope)),
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
            "qualified_document_count": len(qualified_documents),
        },
        "input_hashes": {
            "source": _sha256(Path(args.input).resolve()),
            "signal_report": _sha256(signal_report_path),
            "signal_probability": _sha256(signal_probability_path),
            "frozen_probability": _sha256(frozen_probability_path),
            "quality_decisions": _sha256(Path(args.quality_decisions).resolve()),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
        "production_eligible": False,
    }
    _write_json(output_dir / "signal_block_decode_oof_report.json", result)
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
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument(
        "--anchor-probabilities",
        type=float,
        nargs="+",
        default=(0.90, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975),
    )
    parser.add_argument("--anchors-required", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--anchor-windows", type=int, nargs="+", default=(8, 16))
    parser.add_argument(
        "--maximum-bridge-gaps", type=int, nargs="+", default=(8, 16)
    )
    parser.add_argument(
        "--inside-probabilities", type=float, nargs="+", default=(0.20, 0.40, 0.60)
    )
    parser.add_argument(
        "--adjacent-expansions", type=int, nargs="+", default=(1, 2)
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

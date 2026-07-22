#!/usr/bin/env python3
"""Materialize complete text contexts for false signal-block predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import BlockConfig, blocks_from_mask
from .bibliography_entry_component_gate import _load_quality_exclusions
from .bibliography_entry_dataset import LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_signal_block_decode import (
    SCHEMA_VERSION as BLOCK_SCHEMA,
    decode_signal_blocks,
)


SCHEMA_VERSION = "bibliography-signal-block-false-review-v1"


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="train")
    block_root = Path(args.signal_block_dir).resolve()
    report_path = block_root / "signal_block_decode_oof_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != BLOCK_SCHEMA
        or report.get("validation_opened") is not False
    ):
        raise ValueError("review requires validation-isolated signal blocks")
    row = report[str(args.candidate)]
    config = BlockConfig(**row["config"])
    signal_path = Path(args.signal_tcn_dir).resolve() / "signal_tcn_oof_probability.npy"
    frozen_path = Path(args.line_oof_dir).resolve() / f"{args.line_arm}.oof_probability.npy"
    scope_path = block_root / "auxiliary_scope_active.npy"
    signal_probability = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    frozen_probability = np.load(frozen_path, mmap_mode="r", allow_pickle=False)
    auxiliary_scope = np.load(scope_path, mmap_mode="r", allow_pickle=False)
    excluded_ids, quality_packet = _load_quality_exclusions(
        Path(args.quality_decisions).resolve()
    )
    qualified_documents = {
        index
        for index, document in enumerate(table.documents)
        if str(document["document_id"]) not in excluded_ids
    }
    prediction, _ = decode_signal_blocks(
        table,
        signal_probability,
        frozen_probability,
        auxiliary_scope,
        config,
        qualified_documents=qualified_documents,
        apply_veto=True,
    )
    gold = table.original_labels == LABEL_TO_ID["BIB"]
    candidates: list[dict[str, Any]] = []
    wanted_ids: set[str] = set()
    for document_index in sorted(qualified_documents):
        document = table.documents[document_index]
        start, end = int(document["line_start"]), int(document["line_end"])
        local_gold = gold[start:end]
        local_prediction = prediction[start:end]
        zero_bib = not bool(np.any(local_gold))
        for block_start, block_end in blocks_from_mask(
            local_prediction, table.abs_indices[start:end]
        ):
            false = ~local_gold[block_start : block_end + 1]
            false_lines = int(np.count_nonzero(false))
            if not false_lines:
                continue
            false_tokens = int(
                table.token_counts[start + block_start : start + block_end + 1][
                    false
                ].sum()
            )
            document_id = str(document["document_id"])
            wanted_ids.add(document_id)
            candidates.append(
                {
                    "document_index": document_index,
                    "document_id": document_id,
                    "source": document.get("source"),
                    "block_start": block_start,
                    "block_end": block_end,
                    "false_line_count": false_lines,
                    "false_token_count": false_tokens,
                    "zero_bib_document": zero_bib,
                }
            )
    candidates.sort(
        key=lambda item: (
            not item["zero_bib_document"],
            -item["false_token_count"],
            -item["false_line_count"],
            item["document_id"],
            item["block_start"],
        )
    )
    selected = candidates[: int(args.limit)]
    selected_ids = {item["document_id"] for item in selected}
    source_rows: dict[str, dict[str, Any]] = {}
    with Path(args.input).resolve().open("r", encoding="utf-8") as handle:
        for raw in handle:
            if '"split": "train"' not in raw:
                continue
            if not any(document_id in raw for document_id in selected_ids):
                continue
            source = json.loads(raw)
            document_id = str(source.get("document_id"))
            if document_id in selected_ids:
                source_rows[document_id] = source
    if source_rows.keys() != selected_ids:
        raise ValueError("could not recover every selected source document")
    for item in selected:
        document = table.documents[int(item["document_index"])]
        start, end = int(document["line_start"]), int(document["line_end"])
        lines = source_rows[item["document_id"]].get("lines")
        if not isinstance(lines, list) or len(lines) != end - start:
            raise ValueError("source/table line alignment failed")
        context_start = max(0, int(item["block_start"]) - int(args.context))
        context_end = min(end - start, int(item["block_end"]) + int(args.context) + 1)
        item["context_start"] = context_start
        item["context_end"] = context_end - 1
        item["lines"] = [
            {
                "local_index": local_index,
                "abs_index": int(table.abs_indices[start + local_index]),
                "text": lines[local_index].get("text", ""),
                "silver_bib": bool(gold[start + local_index]),
                "predicted_bib": bool(prediction[start + local_index]),
                "signal_probability": float(signal_probability[start + local_index]),
            }
            for local_index in range(context_start, context_end)
        ]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate": str(args.candidate),
        "config": row["config"],
        "metrics": row["metrics"],
        "false_component_count": len(candidates),
        "zero_bib_false_component_count": sum(
            bool(item["zero_bib_document"]) for item in candidates
        ),
        "selected_count": len(selected),
        "contexts": selected,
        "quality_filter": {
            "decision_schema": quality_packet["schema_version"],
            "excluded_document_count": len(excluded_ids),
        },
        "input_hashes": {
            "source": _sha256(Path(args.input).resolve()),
            "block_report": _sha256(report_path),
            "signal_probability": _sha256(signal_path),
            "frozen_probability": _sha256(frozen_path),
            "scope": _sha256(scope_path),
        },
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "validation_opened": False,
    }
    packet_path = output_dir / "false_block_contexts.json"
    _write_json(packet_path, result)
    _write_json(
        output_dir / "receipt.json",
        {
            **result,
            "packet": {
                "bytes": packet_path.stat().st_size,
                "sha256": _sha256(packet_path),
            },
        },
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--line-oof-dir", required=True)
    parser.add_argument("--signal-tcn-dir", required=True)
    parser.add_argument("--signal-block-dir", required=True)
    parser.add_argument("--quality-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-arm", default="D1")
    parser.add_argument(
        "--candidate",
        choices=(
            "selected",
            "diagnostic_highest_recall_at_line_precision_0_95",
            "diagnostic_highest_recall_candidate",
        ),
        default="diagnostic_highest_recall_at_line_precision_0_95",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

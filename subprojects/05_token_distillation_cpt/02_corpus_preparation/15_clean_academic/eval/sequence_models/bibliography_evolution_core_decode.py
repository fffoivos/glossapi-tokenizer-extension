#!/usr/bin/env python3
"""Evaluate one anchored-core decoder configuration on the fixed dev set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_blocks import BlockConfig, evaluate_prediction
from .bibliography_entry_models import load_table
from .bibliography_signal_block_decode import decode_signal_blocks


SCHEMA_VERSION = "bibliography-evolution-core-decode-v1"


def decoding_document_subset(
    document_count: int, selected: set[int], *, decode_all_documents: bool
) -> set[int]:
    return set(range(document_count)) if decode_all_documents else set(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _selected(table: Any, path: Path) -> tuple[set[int], dict[str, set[int]]]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    ids = packet.get("document_ids")
    if not isinstance(ids, list) or len(ids) != 268 or len(ids) != len(set(ids)):
        raise ValueError("qualified inventory must contain 268 unique documents")
    index = {str(row["document_id"]): number for number, row in enumerate(table.documents)}
    if not set(ids).issubset(index):
        raise ValueError("qualified inventory is not represented in the table")
    selected = {index[value] for value in ids}
    by_source: dict[str, set[int]] = {}
    for document_index in selected:
        by_source.setdefault(str(table.documents[document_index]["source"]), set()).add(document_index)
    return selected, by_source


def run(args: argparse.Namespace) -> dict[str, Any]:
    table = load_table(args.table_dir, expected_split="validation")
    signal = np.load(args.signal_probability, allow_pickle=False)
    line = np.load(args.line_probability, allow_pickle=False)
    scope = np.load(args.scope_mask, allow_pickle=False).astype(bool)
    if signal.shape != line.shape or signal.shape != scope.shape or signal.shape != (len(table.targets),):
        raise ValueError("validation decoder inputs do not align")
    selected, by_source = _selected(table, Path(args.qualified_documents))
    config = BlockConfig(
        anchor_probability=float(args.anchor_probability),
        seed_length_limit=1,
        anchors_required=int(args.anchors_required),
        anchor_window=int(args.anchor_window),
        maximum_bridge_gap=int(args.maximum_bridge_gap),
        inside_probability=float(args.inside_probability),
        adjacent_expansion=int(args.adjacent_expansion),
        header_window=int(args.header_window),
    )
    decode_documents = decoding_document_subset(
        len(table.documents), selected, decode_all_documents=bool(args.decode_all_documents)
    )
    prediction, scope_intervals = decode_signal_blocks(
        table,
        signal,
        line,
        scope,
        config,
        qualified_documents=decode_documents,
        apply_veto=True,
    )
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    prediction_path = output / "prediction.npy"
    with prediction_path.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    with (output / "combined_barriers.npz").open("xb") as handle:
        np.savez(
            handle,
            hard_wall=scope.astype(bool),
            upward_stop=np.zeros(len(scope), dtype=bool),
            downward_stop=np.zeros(len(scope), dtype=bool),
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_fixed_development_decode",
        "validation_opened": True,
        "final_test_opened": False,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "config": config.__dict__,
        "scope_barrier_interval_count": scope_intervals,
        "decoded_document_count": len(decode_documents),
        "headline_document_count": len(selected),
        "inputs": {
            "table_receipt": _sha256(Path(args.table_dir) / "receipt.json"),
            "signal_probability": _sha256(Path(args.signal_probability)),
            "line_probability": _sha256(Path(args.line_probability)),
            "scope_mask": _sha256(Path(args.scope_mask)),
            "qualified_documents": _sha256(Path(args.qualified_documents)),
        },
        "metrics": evaluate_prediction(table, prediction, document_subset=selected),
        "metrics_by_source": {
            source: evaluate_prediction(table, prediction, document_subset=documents)
            for source, documents in sorted(by_source.items())
        },
        "prediction_sha256": _sha256(prediction_path),
    }
    _write(output / "report.json", result)
    outputs = {
        path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    _write(output / "receipt.json", {**result, "outputs": outputs})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--signal-probability", required=True)
    parser.add_argument("--line-probability", required=True)
    parser.add_argument("--scope-mask", required=True)
    parser.add_argument("--qualified-documents", required=True)
    parser.add_argument("--anchor-probability", type=float, required=True)
    parser.add_argument("--anchors-required", type=int, required=True)
    parser.add_argument("--anchor-window", type=int, required=True)
    parser.add_argument("--maximum-bridge-gap", type=int, required=True)
    parser.add_argument("--inside-probability", type=float, required=True)
    parser.add_argument("--adjacent-expansion", type=int, required=True)
    parser.add_argument("--header-window", type=int, default=2)
    parser.add_argument("--decode-all-documents", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

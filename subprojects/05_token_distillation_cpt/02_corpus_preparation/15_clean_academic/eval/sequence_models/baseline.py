#!/usr/bin/env python3
"""Frozen C0 Python reproduction of the deployed LR + hysteresis heads.

This module loads the tracked model artifacts; it never fits or tunes anything.
Joint BIB+ToC decoding retains overlaps as fail-closed conflicts. The current
BIB-only replay instead follows the frozen BIB head and records whether the
inactive ToC head also fired; that observation does not suppress a BIB output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bib_ladder import configure_runtime, peak_rss_bytes, verify_selection_bundle
from .contract import GoldDocument, sha256_file
from .features import EVAL_DIR, existing_features, span_signals


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _scores(rows: Sequence[Mapping[str, float]], model: Mapping[str, Any]) -> list[float]:
    scores: list[float] = []
    for row in rows:
        value = float(model["bias"])
        for name, mean, scale, weight in zip(
            model["features"], model["mu"], model["sd"], model["weight"]
        ):
            value += float(weight) * ((float(row[name]) - float(mean)) / float(scale))
        scores.append(_sigmoid(value))
    return scores


def hysteresis(probabilities: Sequence[float], params: Mapping[str, Any]) -> list[tuple[int, int]]:
    hi, lo = float(params["theta_hi"]), float(params["theta_lo"])
    gap, minimum = int(params["gap"]), int(params["lmin"])
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, probability in enumerate(probabilities):
        if start is None and probability >= hi:
            start = index
        elif start is not None and probability < lo:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(probabilities) - 1))
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start <= merged[-1][1] + gap + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(start, end) for start, end in merged if end - start + 1 >= minimum]


def predict_document(
    document: GoldDocument,
    bib_model: Mapping[str, Any],
    toc_model: Mapping[str, Any],
    decoder: Mapping[str, Any],
    *,
    active_classes: Sequence[str] = ("BIB", "TOC"),
) -> tuple[list[str], list[bool]]:
    active = frozenset(active_classes)
    if active not in {frozenset(("BIB",)), frozenset(("BIB", "TOC"))}:
        raise ValueError(f"invalid active classes: {sorted(active)!r}")
    legacy = {
        "lines": [(line.abs_idx, line.text) for line in document.lines],
        "N": document.n_physical_lines,
    }
    base = existing_features.doc_features(legacy)
    toc_rows = []
    for line, row in zip(document.lines, base):
        merged = dict(row)
        merged.update(span_signals.toc_signals(line.text))
        toc_rows.append(merged)
    bib_probability = _scores(base, bib_model)
    toc_probability = _scores(toc_rows, toc_model)
    cut = min(300, int(0.30 * document.n_physical_lines))
    toc_probability = [
        probability if line.abs_idx < cut else 0.0
        for line, probability in zip(document.lines, toc_probability)
    ]
    bib = set()
    toc = set()
    for start, end in hysteresis(bib_probability, decoder["bib"]):
        bib.update(range(start, end + 1))
    for start, end in hysteresis(toc_probability, decoder["toc"]):
        toc.update(range(start, end + 1))
    conflicts = [index in bib and index in toc for index in range(len(document.lines))]
    if active == {"BIB"}:
        # SPAN evidence supervises only the BIB task. Compare with the frozen
        # BIB head itself; absence of a ToC label is not a ToC-negative label.
        predictions = ["BIB" if index in bib else "O" for index in range(len(document.lines))]
    else:
        predictions = [
            "O"
            if conflicts[index]
            else "BIB"
            if index in bib
            else "TOC"
            if index in toc
            else "O"
            for index in range(len(document.lines))
        ]
    return predictions, conflicts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-silver", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--validation-silver", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--uenv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runtime = configure_runtime(config, uenv=args.uenv, effective_seed=None)
    _selection, documents, selection_receipt = verify_selection_bundle(
        selection_silver_path=args.selection_silver,
        selection_manifest_path=args.selection_manifest,
        validation_silver_path=args.validation_silver,
        selection_receipt_path=args.selection_receipt,
        config_path=args.config,
    )
    bib_path = EVAL_DIR / "span_line_lr_struct_model.json"
    toc_path = EVAL_DIR / "toc_line_lr_model.json"
    decoder_path = EVAL_DIR / "struct_smooth_params.json"
    bib_model, toc_model, decoder = map(_load_json, (bib_path, toc_path, decoder_path))
    output = Path(args.output)
    receipt_output = Path(args.receipt_out)
    for path in (output, receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing immutable output overwrite: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for document in documents:
                predictions, conflicts = predict_document(
                    document,
                    bib_model,
                    toc_model,
                    decoder,
                    active_classes=("BIB",),
                )
                row = {
                    "schema_version": "academic-structure-predictions-v1",
                    "model_id": "c0-rust-lr-hysteresis-python-bib-head",
                    "document_id": document.document_id,
                    "work_id": document.work_id,
                    "source": document.source,
                    "split": document.split,
                    "artifacts": {
                        path.name: sha256_file(path)
                        for path in (bib_path, toc_path, decoder_path)
                    },
                    "lines": [
                        {
                            "line_id": line.line_id,
                            "abs_idx": line.abs_idx,
                            "prediction": prediction,
                            "other_head_overlap_observed": conflict,
                        }
                        for line, prediction, conflict in zip(
                            document.lines, predictions, conflicts
                        )
                    ],
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    runtime["wall_seconds"] = time.perf_counter() - started
    runtime["peak_rss_bytes"] = peak_rss_bytes()
    receipt = {
        "schema_version": "academic-structure-c0-reference-v2",
        "status": "passed_descriptive_reference_prediction",
        "architecture_id": "c0-rust-lr-hysteresis",
        "comparison_role": "historical_reference_only",
        "overlap_caveat": (
            "STRUCT-2K training overlap with SPAN cannot be excluded; this is not an "
            "independent held-out baseline and differences are descriptive only"
        ),
        "production_eligible": False,
        "inputs": {
            "selection_silver_sha256": sha256_file(args.selection_silver),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "validation_silver_sha256": sha256_file(args.validation_silver),
            "selection_receipt_sha256": sha256_file(args.selection_receipt),
            "config_sha256": sha256_file(args.config),
            "source_rehydration_receipt_sha256": selection_receipt["source"][
                "rehydration_receipt_sha256"
            ],
        },
        "model_artifacts": {
            path.name: sha256_file(path) for path in (bib_path, toc_path, decoder_path)
        },
        "execution": runtime,
        "effective_seed": None,
        "outputs": {"validation_predictions_sha256": sha256_file(output)},
        "historically_named_test_partition": {
            "documents_loaded": 0,
            "predictions_written": 0,
            "semantics": "sealed_retrospective_comparison_not_unbiased_test",
        },
    }
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{receipt_output.name}.", suffix=".partial", dir=receipt_output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, receipt_output)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen C0 Python reproduction of the deployed LR + hysteresis heads.

This module loads the tracked model artifacts; it never fits or tunes anything.
Lines selected by both heads are emitted as conflicts and retained (fail closed).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import GoldDocument, read_gold, sha256_file
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
) -> tuple[list[str], list[bool]]:
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
    predictions = [
        "O" if conflicts[index] else "BIB" if index in bib else "TOC" if index in toc else "O"
        for index in range(len(document.lines))
    ]
    return predictions, conflicts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    bib_path = EVAL_DIR / "span_line_lr_struct_model.json"
    toc_path = EVAL_DIR / "toc_line_lr_model.json"
    decoder_path = EVAL_DIR / "struct_smooth_params.json"
    bib_model, toc_model, decoder = map(_load_json, (bib_path, toc_path, decoder_path))
    documents = read_gold(args.silver)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for document in documents:
            predictions, conflicts = predict_document(document, bib_model, toc_model, decoder)
            row = {
                "schema_version": "academic-structure-predictions-v1",
                "model_id": "c0-rust-lr-hysteresis-python-parity",
                "document_id": document.document_id,
                "work_id": document.work_id,
                "source": document.source,
                "split": document.split,
                "artifacts": {
                    path.name: sha256_file(path) for path in (bib_path, toc_path, decoder_path)
                },
                "lines": [
                    {
                        "line_id": line.line_id,
                        "abs_idx": line.abs_idx,
                        "prediction": prediction,
                        "head_conflict_retained": conflict,
                    }
                    for line, prediction, conflict in zip(document.lines, predictions, conflicts)
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

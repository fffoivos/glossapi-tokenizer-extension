#!/usr/bin/env python3
"""Prediction- and label-blind extraction-quality screen for train documents."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_validation_quality import (
    CRITERIA_VERSION,
    _render_html,
    _score_rust,
    analyze_text,
    candidate_reasons,
)


SCHEMA_VERSION = "bibliography-training-quality-screen-v1"


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


def _iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"input row {row_number} is not an object")
            yield row


def _table_document_ids(table_dir: Path) -> set[str]:
    manifest = json.loads((table_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "bibliography-entry-feature-table-v1"
        or manifest.get("split") != "train"
    ):
        raise ValueError("quality screen requires the train feature-table inventory")
    with (table_dir / "documents.jsonl").open(encoding="utf-8") as handle:
        ids = {str(json.loads(line)["document_id"]) for line in handle if line.strip()}
    if len(ids) != int(manifest["document_count"]):
        raise ValueError("train document inventory is not unique")
    return ids


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    table_dir = Path(args.table_dir).resolve()
    expected_ids = _table_document_ids(table_dir)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    rust_module = importlib.import_module("glossapi_rs_noise")

    candidates: list[dict[str, Any]] = []
    observed: set[str] = set()
    for row in _iter_rows(input_path):
        if row.get("split") != "train":
            continue
        document_id = str(row.get("document_id", ""))
        if document_id in observed:
            raise ValueError(f"duplicate train document: {document_id}")
        if document_id not in expected_ids:
            raise ValueError(f"train input/table mismatch: {document_id}")
        raw_lines = row.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(f"{document_id}: missing line inventory")
        lines = [
            {"abs_idx": int(line["abs_idx"]), "text": str(line["text"])}
            for line in raw_lines
        ]
        text_lines = [line["text"] for line in lines]
        quality = analyze_text(text_lines)
        rust = _score_rust("\n".join(text_lines), rust_module)
        reasons = candidate_reasons(
            quality, float(rust["greek_badness_score"])
        )
        if reasons:
            candidates.append(
                {
                    "document_id": document_id,
                    "source": str(row.get("source", "")),
                    "text_quality": asdict(quality),
                    "rust_metrics": rust,
                    "candidate_reasons": reasons,
                    "decision": None,
                    "decision_reason": None,
                    "lines": lines,
                }
            )
        observed.add(document_id)
    if observed != expected_ids:
        raise ValueError("train input/table document inventory mismatch")

    # Do not reserve the immutable output path until all source validation and
    # scoring have succeeded.  A failed run can then be retried safely.
    output_dir.mkdir(parents=True)
    candidates.sort(
        key=lambda row: (
            -len(row["candidate_reasons"]),
            -int(row["text_quality"]["glyph_placeholder_count"]),
            row["document_id"],
        )
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "criteria_version": CRITERIA_VERSION,
        "input_sha256": _sha256(input_path),
        "table_manifest_sha256": _sha256(table_dir / "manifest.json"),
        "document_count": len(observed),
        "candidate_count": len(candidates),
        "prediction_blind_quality_screen": True,
        "labels_read": False,
        "candidates": candidates,
    }
    _write_json(output_dir / "quality_screen.json", packet)
    (output_dir / "index.html").write_text(
        _render_html(packet), encoding="utf-8"
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_prediction_and_label_blind_train_quality_screen",
        "code_commit": str(args.code_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "document_count": len(observed),
        "candidate_count": len(candidates),
        "prediction_blind_quality_screen": True,
        "labels_read": False,
        "quality_screen_sha256": _sha256(output_dir / "quality_screen.json"),
        "index_sha256": _sha256(output_dir / "index.html"),
        "production_eligible": False,
    }
    _write_json(output_dir / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

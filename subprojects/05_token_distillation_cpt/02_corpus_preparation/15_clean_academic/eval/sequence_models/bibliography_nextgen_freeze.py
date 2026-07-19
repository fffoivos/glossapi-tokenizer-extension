#!/usr/bin/env python3
"""Freeze development-selected nextgen candidates before sealed-test access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bibliography_nextgen_decode import SCHEMA_VERSION as DECODER_SCHEMA
from .bibliography_nextgen_models import SCHEMA_VERSION as MODEL_SCHEMA
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-candidate-freeze-v1"


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_candidate(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"invalid candidate: {raw!r}")
    paths = parts[1].split("::", 1)
    if len(paths) != 2:
        raise ValueError("candidate format is NAME=MODEL_DIR::DECODER_DIR")
    return parts[0], Path(paths[0]).resolve(), Path(paths[1]).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    seal_path = Path(args.test_seal).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("status") != "frozen_posthoc_consensus_silver_evaluation_set"
        or seal.get("human_gold") is not False
    ):
        raise ValueError("expected the frozen post-hoc consensus-silver test seal")
    rows = []
    names: set[str] = set()
    for raw in args.candidate:
        name, model_root, decoder_root = _parse_candidate(raw)
        if name in names:
            raise ValueError(f"duplicate candidate name: {name}")
        names.add(name)
        model = json.loads((model_root / "report.json").read_text(encoding="utf-8"))
        decoder = json.loads((decoder_root / "report.json").read_text(encoding="utf-8"))
        selected = decoder.get("selected")
        if (
            model.get("schema_version") != MODEL_SCHEMA
            or model.get("test_opened") is not False
            or decoder.get("schema_version") != DECODER_SCHEMA
            or decoder.get("test_opened") is not False
            or not isinstance(selected, Mapping)
        ):
            raise ValueError(f"candidate is not a selected development-only model: {name}")
        rows.append(
            {
                "name": name,
                "model_dir": str(model_root),
                "model_kind": model["kind"],
                "model_receipt_sha256": sha256_file(model_root / "receipt.json"),
                "model_probability_sha256": sha256_file(
                    model_root / "oof_probability.npy"
                ),
                "decoder_dir": str(decoder_root),
                "decoder_receipt_sha256": sha256_file(decoder_root / "receipt.json"),
                "decoder_config": selected["config"],
                "development_metrics": selected["metrics"],
            }
        )
    if not rows:
        raise ValueError("at least one frozen candidate is required")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_test_open",
        "test_opened": False,
        "test_labels_opened": False,
        "selection_protocol": (
            "one-shot joint test of candidates frozen from grouped development OOF; "
            "no post-test threshold or topology selection"
        ),
        "selection_rule": args.selection_rule,
        "candidate_count": len(rows),
        "candidates": rows,
        "test_seal": {
            "path": str(seal_path),
            "sha256": sha256_file(seal_path),
            "schema_version": seal["schema_version"],
            "document_count": seal["document_count"],
            "line_count": seal["line_count"],
            "documents_sha256": seal["sealed_hashes"]["documents_sha256"],
            "labels_sha256": seal["sealed_hashes"]["labels_sha256"],
            "line_key_sha256": seal["sealed_hashes"]["line_key_sha256"],
            "human_gold": False,
        },
        "code_commit": args.code_commit,
    }
    _write_json_new(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--test-seal", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

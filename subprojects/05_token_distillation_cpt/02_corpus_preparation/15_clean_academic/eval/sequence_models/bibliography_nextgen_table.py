#!/usr/bin/env python3
"""Materialize full-document features for next-generation bibliography models.

The earlier role-aware block learner used only fully reviewed role windows.  This
table keeps those out-of-fold expert signals, but aligns them to every line in
the 1,118-document binary bibliography training corpus.  The source BIB/O/TOC
labels remain the only block target; detailed roles are features, not inferred
targets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .bibliography_entry_dataset import FEATURE_NAMES, LABEL_TO_ID
from .bibliography_entry_models import load_table
from .bibliography_positional_models import load_positional_table
from .bibliography_role_experts import (
    CONNECTOR_PROBABILITY_COLUMNS,
    HEADING_PROBABILITY_COLUMNS,
)
from .bibliography_role_features import GAP_SUMMARY_NAMES, LINE_SHAPE_NAMES, line_shape
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-nextgen-full-table-v1"
ROLE_PROBABILITY_NAMES = (
    "entry",
    "continuation",
    "filler",
    "bib_header",
    "bib_subheader",
    "non_bib_header",
    "other",
)
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_IMAGE_MARKER = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.IGNORECASE)


def feature_names() -> tuple[str, ...]:
    return (
        "probability:entry",
        "probability:signal_tcn",
        "probability:connector",
        "probability:continuation_specialist",
        *(f"probability:{name}" for name in ROLE_PROBABILITY_NAMES[1:]),
        *(f"presence:{name}" for name in FEATURE_NAMES),
        *(f"log1p:{name}" for name in FEATURE_NAMES),
        *(f"shape:{name}" for name in LINE_SHAPE_NAMES),
        *(f"gap:{name}" for name in GAP_SUMMARY_NAMES),
        "structure:markdown_heading",
        "structure:image_marker",
        "structure:table_row",
    )


def _iter_jsonl(path: Path, split: str) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            if value.get("split") == split:
                yield value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _scatter(
    root: Path,
    *,
    line_count: int,
    columns: int,
    default: Sequence[float],
) -> np.ndarray:
    indices = np.load(root / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    probability = np.load(root / "oof_probability.npy", mmap_mode="r", allow_pickle=False)
    if probability.shape != (len(indices), columns):
        raise ValueError(f"malformed probability artifact: {root}")
    if len(indices) and (
        len(np.unique(indices)) != len(indices) or int(indices.max()) >= line_count
    ):
        raise ValueError(f"malformed row inventory: {root}")
    result = np.tile(np.asarray(default, dtype=np.float32), (line_count, 1))
    result[indices] = probability
    return result


def _load_specialist(
    root: Path | None,
    connector_root: Path,
    fallback: np.ndarray,
    line_count: int,
    filename: str,
) -> tuple[np.ndarray, str | None]:
    result = np.asarray(fallback, dtype=np.float32).copy()
    if root is None:
        return result, None
    indices = np.load(connector_root / "row_indices.npy", mmap_mode="r", allow_pickle=False)
    probability_path = root / filename
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probability.shape != (len(indices),) or np.any(indices >= line_count):
        raise ValueError("continuation specialist is not aligned to connector candidates")
    result[indices] = probability
    return result, sha256_file(probability_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    base_root = Path(args.base_table_dir).resolve()
    positional_root = Path(args.positional_table_dir).resolve()
    entry_root = Path(args.entry_oof_dir).resolve()
    heading_root = Path(args.heading_oof_dir).resolve()
    connector_root = Path(args.connector_oof_dir).resolve()
    continuation_root = (
        Path(args.continuation_oof_dir).resolve() if args.continuation_oof_dir else None
    )
    signal_path = Path(args.signal_probability).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    table = load_table(base_root, expected_split=args.split)
    n = len(table.targets)
    positional = load_positional_table(
        positional_root, sha256_file(base_root / "manifest.json"), n
    )
    entry_path = entry_root / "P0D.oof_probability.npy"
    entry = np.load(entry_path, mmap_mode="r", allow_pickle=False)
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    if entry.shape != (n,) or signal.shape != (n,):
        raise ValueError("entry/signal probabilities do not cover the full base table")
    if not np.isfinite(entry).all() or not np.isfinite(signal).all():
        raise ValueError("entry/signal probabilities contain non-finite values")

    heading = _scatter(
        heading_root,
        line_count=n,
        columns=len(HEADING_PROBABILITY_COLUMNS),
        default=(0.0, 0.0, 0.0, 0.0),
    )
    connector = _scatter(
        connector_root,
        line_count=n,
        columns=len(CONNECTOR_PROBABILITY_COLUMNS),
        default=(0.0, 0.0, 0.0, 1.0),
    )
    continuation, continuation_sha = _load_specialist(
        continuation_root,
        connector_root,
        connector[:, 1],
        n,
        args.continuation_probability_file,
    )

    names = feature_names()
    output.mkdir(parents=True)
    features = open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32, shape=(n, len(names))
    )
    documents = list(_iter_jsonl(source, args.split))
    if len(documents) != len(table.documents):
        raise ValueError("source/base document count mismatch")
    table_feature = FEATURE_NAMES.index("table_row_count")
    cursor = 0
    markdown_count = image_count = table_count = 0
    for document_index, (document, metadata) in enumerate(
        zip(documents, table.documents, strict=True)
    ):
        lines = document.get("lines")
        start, end = int(metadata["line_start"]), int(metadata["line_end"])
        if (
            document.get("document_id") != metadata["document_id"]
            or not isinstance(lines, list)
            or len(lines) != end - start
        ):
            raise ValueError(f"source/base alignment failure at document {document_index}")
        for offset, line in enumerate(lines):
            absolute = start + offset
            text = line.get("text")
            if not isinstance(text, str) or int(line.get("abs_idx", -1)) != int(
                table.abs_indices[absolute]
            ):
                raise ValueError(f"source/base line mismatch at {document_index}:{offset}")
            markdown = int(bool(_MARKDOWN_HEADING.match(text)))
            image = int(bool(_IMAGE_MARKER.match(text)))
            table_row = int(bool(table.counts[absolute, table_feature]))
            markdown_count += markdown
            image_count += image
            table_count += table_row
            counts = np.asarray(table.counts[absolute], dtype=np.float32)
            row = np.concatenate(
                (
                    np.asarray(
                        (
                            entry[absolute],
                            signal[absolute],
                            connector[absolute, 0],
                            continuation[absolute],
                            connector[absolute, 1],
                            connector[absolute, 2],
                            heading[absolute, 1],
                            heading[absolute, 2],
                            heading[absolute, 3],
                            connector[absolute, 3],
                        ),
                        dtype=np.float32,
                    ),
                    (counts > 0).astype(np.float32),
                    np.log1p(counts),
                    line_shape(text),
                    np.asarray(positional.gap_summaries[absolute], dtype=np.float32),
                    np.asarray((markdown, image, table_row), dtype=np.float32),
                )
            )
            if row.shape != (len(names),) or not np.isfinite(row).all():
                raise RuntimeError(f"nextgen feature contract failure at line {absolute}")
            features[absolute] = row
            cursor += 1
    if cursor != n:
        raise ValueError("source traversal did not cover the base table")
    features.flush()
    del features

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_full_document_feature_materialization",
        "split": args.split,
        "validation_opened": False,
        "test_opened": False,
        "document_count": len(documents),
        "line_count": n,
        "feature_count": len(names),
        "feature_names": names,
        "markdown_heading_count": markdown_count,
        "image_marker_count": image_count,
        "table_row_count": table_count,
        "binary_bibliography_line_count": int(
            np.count_nonzero(table.original_labels == LABEL_TO_ID["BIB"])
        ),
        "unknown_line_count": int(
            np.count_nonzero(table.original_labels == LABEL_TO_ID["UNKNOWN"])
        ),
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "inputs": {
            "source": {"path": str(source), "sha256": sha256_file(source)},
            "base_manifest_sha256": sha256_file(base_root / "manifest.json"),
            "positional_manifest_sha256": sha256_file(positional_root / "manifest.json"),
            "entry_probability_sha256": sha256_file(entry_path),
            "signal_probability_sha256": sha256_file(signal_path),
            "heading_receipt_sha256": sha256_file(heading_root / "receipt.json"),
            "connector_receipt_sha256": sha256_file(connector_root / "receipt.json"),
            "continuation_probability_sha256": continuation_sha,
        },
    }
    _write_json_new(output / "manifest.json", manifest)
    _write_json_new(
        output / "receipt.json",
        {
            **manifest,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        },
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--positional-table-dir", required=True)
    parser.add_argument("--entry-oof-dir", required=True)
    parser.add_argument("--heading-oof-dir", required=True)
    parser.add_argument("--connector-oof-dir", required=True)
    parser.add_argument("--continuation-oof-dir")
    parser.add_argument(
        "--continuation-probability-file",
        default="compact_plus_directional_join.oof_probability.npy",
    )
    parser.add_argument("--signal-probability", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

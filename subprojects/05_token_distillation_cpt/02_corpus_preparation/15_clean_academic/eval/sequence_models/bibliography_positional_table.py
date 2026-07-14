#!/usr/bin/env python3
"""Materialize a lossless positional extension for the frozen entry table.

This artifact contains only feature geometry.  Corrected role targets are a
separate, cheap overlay so annotations can change without re-extracting all
939k lines.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .bibliography_entry_models import load_table
from .bibliography_positional_features import (
    FEATURE_NAMES,
    GAP_SUMMARY_NAMES,
    NONMATCH_CATEGORIES,
    POSITION_SUMMARY_NAMES,
    extract_positional_line,
)
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-positional-feature-table-v2"


def _iter_rows(path: Path, split: str) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"source row {row_number} is not an object")
            if row.get("split") == split:
                yield row


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _chunk_task(task: tuple[int, Mapping[str, Any], str]) -> dict[str, Any]:
    document_index, row, chunk_root = task
    document_id = str(row.get("document_id", ""))
    lines = row.get("lines")
    if not document_id or not isinstance(lines, list) or not lines:
        raise ValueError(f"source document {document_index}: missing identity/lines")
    counts: list[np.ndarray] = []
    nfkc_lengths: list[int] = []
    position_summaries: list[np.ndarray] = []
    gap_summaries: list[np.ndarray] = []
    match_ptr = [0]
    match_features: list[np.ndarray] = []
    match_starts: list[np.ndarray] = []
    match_ends: list[np.ndarray] = []
    nonmatch_ptr = [0]
    nonmatch_categories: list[np.ndarray] = []
    nonmatch_starts: list[np.ndarray] = []
    nonmatch_ends: list[np.ndarray] = []
    for offset, line in enumerate(lines):
        text = line.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{document_id}: line {offset} has no text")
        encoding = extract_positional_line(text)
        counts.append(encoding.counts)
        nfkc_lengths.append(encoding.nfkc_length)
        position_summaries.append(encoding.position_summaries)
        gap_summaries.append(encoding.gap_summaries)
        match_features.append(encoding.match_feature)
        match_starts.append(encoding.match_start)
        match_ends.append(encoding.match_end)
        match_ptr.append(match_ptr[-1] + len(encoding.match_feature))
        nonmatch_categories.append(encoding.nonmatch_category)
        nonmatch_starts.append(encoding.nonmatch_start)
        nonmatch_ends.append(encoding.nonmatch_end)
        nonmatch_ptr.append(nonmatch_ptr[-1] + len(encoding.nonmatch_category))

    def concatenate(parts: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
        return np.concatenate(parts).astype(dtype, copy=False) if any(len(part) for part in parts) else np.empty(0, dtype=dtype)

    chunk_path = Path(chunk_root) / f"{document_index:06d}.npz"
    with chunk_path.open("xb") as handle:
        np.savez(
            handle,
            counts=np.stack(counts),
            nfkc_lengths=np.asarray(nfkc_lengths, dtype=np.uint32),
            position_summaries=np.stack(position_summaries),
            gap_summaries=np.stack(gap_summaries),
            match_ptr=np.asarray(match_ptr, dtype=np.uint64),
            match_feature=concatenate(match_features, np.dtype(np.uint8)),
            match_start=concatenate(match_starts, np.dtype(np.uint32)),
            match_end=concatenate(match_ends, np.dtype(np.uint32)),
            nonmatch_ptr=np.asarray(nonmatch_ptr, dtype=np.uint64),
            nonmatch_category=concatenate(nonmatch_categories, np.dtype(np.uint8)),
            nonmatch_start=concatenate(nonmatch_starts, np.dtype(np.uint32)),
            nonmatch_end=concatenate(nonmatch_ends, np.dtype(np.uint32)),
        )
    return {
        "document_index": document_index,
        "document_id": document_id,
        "line_count": len(lines),
        "match_count": match_ptr[-1],
        "nonmatch_run_count": nonmatch_ptr[-1],
        "chunk_sha256": sha256_file(chunk_path),
    }


def _allocate(root: Path, name: str, dtype: Any, shape: tuple[int, ...]) -> np.memmap:
    return open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.input).resolve()
    source_sha = sha256_file(source_path)
    if args.expected_input_sha256 and source_sha != args.expected_input_sha256:
        raise ValueError("source input SHA-256 differs from the pin")
    base = load_table(args.base_table_dir, expected_split=args.split)
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    chunks = output / ".chunks"
    chunks.mkdir()
    source_rows = list(_iter_rows(source_path, args.split))
    if len(source_rows) != len(base.documents):
        raise ValueError("source/base document count mismatch")
    for index, (row, document) in enumerate(zip(source_rows, base.documents, strict=True)):
        start, end = int(document["line_start"]), int(document["line_end"])
        source_abs = np.asarray([line.get("abs_idx") for line in row.get("lines", [])])
        if (
            row.get("document_id") != document["document_id"]
            or len(row.get("lines", [])) != int(document["line_count"])
            or not np.array_equal(source_abs, base.abs_indices[start:end])
        ):
            raise ValueError(f"source/base document alignment mismatch at {index}")
    tasks = [(index, row, str(chunks)) for index, row in enumerate(source_rows)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        reports = list(executor.map(_chunk_task, tasks, chunksize=1))
    reports.sort(key=lambda row: int(row["document_index"]))
    line_count = sum(int(row["line_count"]) for row in reports)
    match_count = sum(int(row["match_count"]) for row in reports)
    nonmatch_count = sum(int(row["nonmatch_run_count"]) for row in reports)
    if line_count != len(base.targets):
        raise ValueError("source/base line count mismatch")

    arrays = {
        "match_ptr": _allocate(output, "match_ptr", np.uint64, (line_count + 1,)),
        "match_feature": _allocate(output, "match_feature", np.uint8, (match_count,)),
        "match_start": _allocate(output, "match_start", np.uint32, (match_count,)),
        "match_end": _allocate(output, "match_end", np.uint32, (match_count,)),
        "nonmatch_ptr": _allocate(output, "nonmatch_ptr", np.uint64, (line_count + 1,)),
        "nonmatch_category": _allocate(output, "nonmatch_category", np.uint8, (nonmatch_count,)),
        "nonmatch_start": _allocate(output, "nonmatch_start", np.uint32, (nonmatch_count,)),
        "nonmatch_end": _allocate(output, "nonmatch_end", np.uint32, (nonmatch_count,)),
        "nfkc_lengths": _allocate(output, "nfkc_lengths", np.uint32, (line_count,)),
        "position_summaries": _allocate(
            output, "position_summaries", np.float32,
            (line_count, len(FEATURE_NAMES), len(POSITION_SUMMARY_NAMES)),
        ),
        "gap_summaries": _allocate(
            output, "gap_summaries", np.float32, (line_count, len(GAP_SUMMARY_NAMES)),
        ),
    }
    line_cursor = match_cursor = nonmatch_cursor = 0
    arrays["match_ptr"][0] = arrays["nonmatch_ptr"][0] = 0
    for report in reports:
        path = chunks / f"{int(report['document_index']):06d}.npz"
        if sha256_file(path) != report["chunk_sha256"]:
            raise ValueError(f"chunk hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as chunk:
            n = int(report["line_count"])
            m, q = int(report["match_count"]), int(report["nonmatch_run_count"])
            line_slice = slice(line_cursor, line_cursor + n)
            if not np.array_equal(chunk["counts"], base.counts[line_slice]):
                raise ValueError(f"feature count/span parity with base failed: {path.name}")
            arrays["nfkc_lengths"][line_slice] = chunk["nfkc_lengths"]
            arrays["position_summaries"][line_slice] = chunk["position_summaries"]
            arrays["gap_summaries"][line_slice] = chunk["gap_summaries"]
            arrays["match_ptr"][line_cursor : line_cursor + n + 1] = chunk["match_ptr"] + match_cursor
            arrays["nonmatch_ptr"][line_cursor : line_cursor + n + 1] = chunk["nonmatch_ptr"] + nonmatch_cursor
            for name in ("match_feature", "match_start", "match_end"):
                arrays[name][match_cursor : match_cursor + m] = chunk[name]
            for name in ("nonmatch_category", "nonmatch_start", "nonmatch_end"):
                arrays[name][nonmatch_cursor : nonmatch_cursor + q] = chunk[name]
        line_cursor += n
        match_cursor += m
        nonmatch_cursor += q
    if (line_cursor, match_cursor, nonmatch_cursor) != (line_count, match_count, nonmatch_count):
        raise AssertionError("positional cursor mismatch")
    for value in arrays.values():
        value.flush()
    del arrays
    shutil.rmtree(chunks)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_geometry_only_targets_external",
        "split": args.split,
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "source": {"path": str(source_path), "sha256": source_sha},
        "base_table": {
            "path": str(Path(args.base_table_dir).resolve()),
            "manifest_sha256": sha256_file(Path(args.base_table_dir).resolve() / "manifest.json"),
            "counts_sha256": sha256_file(Path(args.base_table_dir).resolve() / "counts.npy"),
        },
        "document_count": len(reports),
        "line_count": line_count,
        "feature_names": list(FEATURE_NAMES),
        "position_summary_names": list(POSITION_SUMMARY_NAMES),
        "gap_summary_names": list(GAP_SUMMARY_NAMES),
        "nonmatch_categories": list(NONMATCH_CATEGORIES),
        "match_count": match_count,
        "nonmatch_run_count": nonmatch_count,
        "count_span_parity_with_base": True,
        "target_policy": "no targets; join a provenance-bound role target view",
        "document_chunk_counts": dict(collections.Counter(int(row["line_count"]) for row in reports)),
    }
    _write_json(output / "manifest.json", manifest)
    outputs = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }
    receipt = {**manifest, "outputs": outputs}
    _write_json(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

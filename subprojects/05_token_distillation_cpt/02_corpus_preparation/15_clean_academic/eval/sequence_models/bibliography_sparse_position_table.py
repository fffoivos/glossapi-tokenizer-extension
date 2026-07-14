#!/usr/bin/env python3
"""Materialize target-aligned sparse 32-bin bibliography position maps."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bibliography_entry_models import load_table
from .bibliography_positional_features import FEATURE_NAMES, NONMATCH_CATEGORIES
from .bibliography_positional_models import load_targets
from .contract import sha256_file


SCHEMA_VERSION = "bibliography-sparse-position-table-v1"


def _scipy() -> Any:
    import scipy.sparse as sparse
    return sparse


def _load_array(root: Path, name: str) -> np.ndarray:
    return np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def _span_matrix(
    root: Path, prefix: str, line_start: int, line_end: int,
    row_by_local_line: np.ndarray, lengths: np.ndarray, bins: int,
) -> Any:
    sparse = _scipy()
    ptr = _load_array(root, f"{prefix}_ptr")
    channel_name = "match_feature" if prefix == "match" else "nonmatch_category"
    channels = _load_array(root, channel_name)
    starts, ends = _load_array(root, f"{prefix}_start"), _load_array(root, f"{prefix}_end")
    left, right = int(ptr[line_start]), int(ptr[line_end])
    counts = np.diff(np.asarray(ptr[line_start : line_end + 1], dtype=np.int64))
    local_lines = np.repeat(np.arange(line_end - line_start, dtype=np.int32), counts)
    if len(local_lines) != right - left:
        raise ValueError("span pointer expansion mismatch")
    rows = row_by_local_line[local_lines]
    span_channels = np.asarray(channels[left:right], dtype=np.int32)
    span_starts = np.asarray(starts[left:right], dtype=np.float64)
    span_ends = np.asarray(ends[left:right], dtype=np.float64)
    span_lengths = np.asarray(lengths[local_lines], dtype=np.float64)
    valid = (rows >= 0) & (span_lengths > 0) & (span_starts < span_ends)
    rows, span_channels = rows[valid], span_channels[valid]
    span_starts, span_ends, span_lengths = (
        span_starts[valid], span_ends[valid], span_lengths[valid]
    )
    if not len(rows):
        channel_count = len(FEATURE_NAMES) if prefix == "match" else len(NONMATCH_CATEGORIES)
        return sparse.csr_matrix((len(np.flatnonzero(row_by_local_line >= 0)), channel_count * bins), dtype=np.float32)
    first = np.minimum(bins - 1, np.floor(span_starts * bins / span_lengths).astype(np.int32))
    last = np.minimum(bins - 1, np.ceil(span_ends * bins / span_lengths).astype(np.int32) - 1)
    widths = last - first + 1
    repeated_rows = np.repeat(rows, widths)
    repeated_channels = np.repeat(span_channels, widths)
    repeated_first = np.repeat(first, widths)
    group_starts = np.repeat(np.cumsum(widths) - widths, widths)
    relative = np.arange(int(widths.sum()), dtype=np.int64) - group_starts
    columns_within = repeated_first + relative
    repeated_lengths = np.repeat(span_lengths, widths)
    repeated_starts = np.repeat(span_starts, widths)
    repeated_ends = np.repeat(span_ends, widths)
    bin_left = columns_within * repeated_lengths / bins
    bin_right = (columns_within + 1) * repeated_lengths / bins
    overlap = np.maximum(
        0.0, np.minimum(repeated_ends, bin_right) - np.maximum(repeated_starts, bin_left)
    )
    values = (overlap * bins / repeated_lengths).astype(np.float32)
    columns = repeated_channels * bins + columns_within
    channel_count = len(FEATURE_NAMES) if prefix == "match" else len(NONMATCH_CATEGORIES)
    matrix = sparse.coo_matrix(
        (values, (repeated_rows, columns)),
        shape=(len(np.flatnonzero(row_by_local_line >= 0)), channel_count * bins),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    np.minimum(matrix.data, 1.0, out=matrix.data)
    matrix.eliminate_zeros()
    return matrix


def _chunk(task: tuple[str, str, int, int, int, str]) -> dict[str, Any]:
    positional_raw, target_raw, start, end, bins, output_raw = task
    positional, target_root, output = Path(positional_raw), Path(target_raw), Path(output_raw)
    targets = _load_array(target_root, "entry_targets")
    lengths = np.asarray(_load_array(positional, "nfkc_lengths")[start:end])
    labelled_local = np.flatnonzero(np.asarray(targets[start:end]) != -1)
    row_by_line = np.full(end - start, -1, dtype=np.int32)
    row_by_line[labelled_local] = np.arange(len(labelled_local), dtype=np.int32)
    match = _span_matrix(positional, "match", start, end, row_by_line, lengths, bins)
    nonmatch = _span_matrix(positional, "nonmatch", start, end, row_by_line, lengths, bins)
    matrix = _scipy().hstack((match, nonmatch), format="csr", dtype=np.float32)
    matrix_path, index_path = output / f"map-{start:07d}.npz", output / f"index-{start:07d}.npy"
    _scipy().save_npz(matrix_path, matrix, compressed=True)
    with index_path.open("xb") as handle:
        np.save(handle, labelled_local.astype(np.int64) + start, allow_pickle=False)
    return {
        "start": start, "end": end, "rows": len(labelled_local), "nnz": matrix.nnz,
        "map": matrix_path.name, "index": index_path.name,
    }


def _write_json_new(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse = _scipy()
    base = load_table(args.base_table_dir, expected_split="train")
    base_sha = sha256_file(base.root / "manifest.json")
    targets, target_manifest = load_targets(args.role_target_dir, base_sha, len(base.targets))
    positional = Path(args.positional_table_dir).resolve()
    positional_manifest = json.loads((positional / "manifest.json").read_text())
    if positional_manifest.get("base_table", {}).get("manifest_sha256") != base_sha:
        raise ValueError("positional/base provenance mismatch")
    bins = int(args.bins)
    if bins != 32:
        raise ValueError("v1 materializes the predeclared maximum of 32 bins")
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    chunks = output / ".chunks"
    chunks.mkdir()
    n = len(targets)
    bounds = list(range(0, n, int(args.chunk_lines))) + [n]
    tasks = [
        (str(positional), str(Path(args.role_target_dir).resolve()), start, end, bins, str(chunks))
        for start, end in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        reports = list(executor.map(_chunk, tasks, chunksize=1))
    reports.sort(key=lambda row: int(row["start"]))
    maps = [sparse.load_npz(chunks / row["map"]) for row in reports]
    indices = [np.load(chunks / row["index"], allow_pickle=False) for row in reports]
    matrix = sparse.vstack(maps, format="csr", dtype=np.float32)
    labelled_indices = np.concatenate(indices)
    expected_indices = np.flatnonzero(np.asarray(targets) != -1)
    if not np.array_equal(labelled_indices, expected_indices):
        raise ValueError("sparse map/target row alignment mismatch")
    summaries = _load_array(positional, "position_summaries")
    maximum_match_integral_error = 0.0
    for channel in range(len(FEATURE_NAMES)):
        observed = np.asarray(
            matrix[:, channel * bins : (channel + 1) * bins].sum(axis=1)
        ).ravel() / bins
        expected = np.asarray(summaries[labelled_indices, channel, 3])
        maximum_match_integral_error = max(
            maximum_match_integral_error,
            float(np.max(np.abs(observed - expected), initial=0.0)),
        )
    nonmatch_start = len(FEATURE_NAMES) * bins
    observed_unmatched = np.asarray(matrix[:, nonmatch_start:].sum(axis=1)).ravel() / bins
    expected_unmatched = np.asarray(
        _load_array(positional, "gap_summaries")[labelled_indices, 0]
    )
    maximum_nonmatch_integral_error = float(
        np.max(np.abs(observed_unmatched - expected_unmatched), initial=0.0)
    )
    if maximum_match_integral_error > 1.0e-5 or maximum_nonmatch_integral_error > 1.0e-5:
        raise ValueError("sparse raster integral parity failed")
    counts = np.asarray(base.counts[labelled_indices], dtype=np.float32)
    gaps = np.asarray(_load_array(positional, "gap_summaries")[labelled_indices], dtype=np.float32)
    gaps[:, 5] = np.log1p(gaps[:, 5])
    scalars = sparse.csr_matrix(
        np.concatenate(((counts > 0).astype(np.float32), np.log1p(counts), gaps), axis=1)
    )
    sparse.save_npz(output / "position_map_32.npz", matrix, compressed=True)
    sparse.save_npz(output / "count_gap_scalars.npz", scalars, compressed=True)
    with (output / "labelled_indices.npy").open("xb") as handle:
        np.save(handle, labelled_indices, allow_pickle=False)
    shutil.rmtree(chunks)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_target_aligned_sparse_geometry",
        "code_commit": args.code_commit,
        "slurm_job_id": args.slurm_job_id,
        "bins": bins,
        "row_count": len(labelled_indices),
        "map_shape": list(matrix.shape),
        "map_nnz": int(matrix.nnz),
        "scalar_shape": list(scalars.shape),
        "scalar_nnz": int(scalars.nnz),
        "maximum_match_integral_error": maximum_match_integral_error,
        "maximum_nonmatch_integral_error": maximum_nonmatch_integral_error,
        "raster_integral_parity": True,
        "base_table_manifest_sha256": base_sha,
        "positional_table_manifest_sha256": sha256_file(positional / "manifest.json"),
        "role_target_manifest_sha256": sha256_file(Path(args.role_target_dir).resolve() / "manifest.json"),
        "overlay_sha256": target_manifest["overlay"]["sha256"],
        "feature_channels": list(FEATURE_NAMES),
        "nonmatch_channels": list(NONMATCH_CATEGORIES),
        "lower_resolution_policy": "derive 8/16-bin coverage by adjacent-bin means",
        "validation_opened": False,
    }
    _write_json_new(output / "manifest.json", manifest)
    receipt = {**manifest, "outputs": {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output.iterdir()) if path.is_file()
    }}
    _write_json_new(output / "receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-table-dir", required=True)
    parser.add_argument("--positional-table-dir", required=True)
    parser.add_argument("--role-target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--chunk-lines", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

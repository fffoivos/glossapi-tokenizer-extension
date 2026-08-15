#!/usr/bin/env python3
"""Audit explicitly selected polytonic source datasets inside the HF release.

This is a read-only source-level audit.  It verifies that the selected named
sources are present in the pinned anonymized release and measures distinctive
polytonic Unicode evidence in every selected row.  It never extracts,
filters, rewrites, reconstructs, or deduplicates corpus rows.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, write_json_atomic


POLYTONIC_COMBINING_MARKS = {0x0300, 0x0313, 0x0314, 0x0342, 0x0345}


def inspect_text(text: object) -> tuple[int, int, int, bool]:
    """Return chars, Greek chars, distinctive polytonic signals and presence."""

    value = "" if text is None else str(text)
    greek = 0
    distinctive = 0
    for char in value:
        codepoint = ord(char)
        if 0x0370 <= codepoint <= 0x03FF or 0x1F00 <= codepoint <= 0x1FFF:
            greek += 1
        if 0x1F00 <= codepoint <= 0x1FFF or codepoint in POLYTONIC_COMBINING_MARKS:
            distinctive += 1
    return len(value), greek, distinctive, distinctive > 0


def inspect_file(path: Path, sources: pa.Array) -> dict[str, dict[str, int]]:
    """Read selected rows only and return per-source evidence counters."""

    result: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    parquet = pq.ParquetFile(path)
    require({"source_dataset", "text"}.issubset(parquet.schema_arrow.names), f"{path}: release schema drift")
    for batch in parquet.iter_batches(columns=["source_dataset", "text"], batch_size=256, use_threads=False):
        source_values = batch.column(0)
        mask = pc.is_in(source_values, value_set=sources)
        if not bool(pc.any(mask).as_py()):
            continue
        filtered_sources = source_values.filter(mask).to_pylist()
        filtered_texts = batch.column(1).filter(mask).to_pylist()
        for source, text in zip(filtered_sources, filtered_texts, strict=True):
            chars, greek, distinctive, present = inspect_text(text)
            stats = result[str(source)]
            stats["rows"] += 1
            stats["text_chars"] += chars
            stats["greek_chars"] += greek
            stats["distinctive_polytonic_signals"] += distinctive
            stats["rows_with_distinctive_polytonic_signal"] += int(present)
    return {source: dict(stats) for source, stats in result.items()}


def merge_rows(rows: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    total: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        for source, stats in row.items():
            total[source].update(stats)
    return {source: dict(total[source]) for source in sorted(total)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable polytonic source audit exists: {args.output}")
    require(args.workers >= 1, "--workers must be positive")
    sources = list(dict.fromkeys(args.source))
    require(len(sources) == len(args.source), "duplicate --source value")

    release_root = args.release_root.resolve()
    anonymization = release_root / "release/manifests/anonymization_manifest.json"
    token_counts = release_root / "release/manifests/token_counts.json"
    manifest = read_json(anonymization)
    counts = read_json(token_counts)
    require(manifest.get("status") == "passed", "release anonymization manifest drift")
    require(counts.get("status") == "passed", "release token-count manifest drift")
    source_rows = counts.get("source_rows", {})
    missing = [source for source in sources if source not in source_rows]
    require(not missing, f"selected polytonic source missing from release: {missing}")
    paths = [release_root / "release" / row["path"] for row in manifest["files"]]
    require(paths and all(path.is_file() for path in paths), "release data file inventory drift")

    value_set = pa.array(sources, type=pa.string())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        per_file = list(executor.map(lambda path: inspect_file(path, value_set), paths))
    observed = merge_rows(per_file)
    require(set(observed) == set(sources), "some selected polytonic source had no release rows")
    for source in sources:
        require(int(observed[source]["rows"]) == int(source_rows[source]), f"{source}: release row count drift")
        require(
            int(observed[source]["distinctive_polytonic_signals"]) > 0,
            f"{source}: no distinctive polytonic Greek evidence",
        )
    total = collections.Counter()
    for stats in observed.values():
        total.update(stats)
    payload: dict[str, Any] = {
        "schema_version": "targeted_8b_release_polytonic_source_audit_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_authority": "pinned_hf_release_only",
        "release_root": str(release_root),
        "anonymization_manifest": file_binding(anonymization),
        "token_counts_manifest": file_binding(token_counts),
        "source_datasets": sources,
        "source_evidence": observed,
        "totals": dict(total),
        "files_scanned": len(paths),
        "signal_definition": {
            "greek_extended_codepoints": "U+1F00..U+1FFF",
            "combining_marks": [f"U+{value:04X}" for value in sorted(POLYTONIC_COMBINING_MARKS)],
            "source_rows_filtered_or_removed": False,
        },
        "invariants": {
            "all_selected_sources_are_present_in_pinned_release": True,
            "all_selected_sources_have_distinctive_polytonic_greek_evidence": True,
            "external_dataset_used": False,
            "rows_written": 0,
            "global_deduplication_performed": False,
        },
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

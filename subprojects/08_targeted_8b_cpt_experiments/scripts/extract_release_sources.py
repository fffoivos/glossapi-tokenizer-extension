#!/usr/bin/env python3
"""Extract named sources without reordering, transforming, or deduplicating rows."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, write_json_atomic


def deterministic_bucket(seed: int, source: str, doc_id: str, denominator: int) -> int:
    payload = f"targeted8b-source-sample-v1\0{seed}\0{source}\0{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % denominator


def inspect(
    path: Path,
    sources: pa.Array,
    *,
    numerator: int,
    denominator: int,
    seed: int,
) -> tuple[Path, int, dict[str, int], pa.Array]:
    identity = pq.read_table(path, columns=["source_dataset", "source_doc_id"])
    column = identity["source_dataset"]
    source_mask = pc.is_in(column, value_set=sources)
    if numerator < denominator:
        source_values = column.to_pylist()
        doc_ids = identity["source_doc_id"].to_pylist()
        sampled = [
            bool(in_source) and deterministic_bucket(seed, str(source), str(doc_id), denominator) < numerator
            for in_source, source, doc_id in zip(source_mask.to_pylist(), source_values, doc_ids, strict=True)
        ]
        mask = pa.array(sampled, type=pa.bool_())
    else:
        mask = source_mask
    count = int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)
    found: dict[str, int] = collections.Counter(
        str(value) for value in column.filter(mask).to_pylist()
    )
    return path, count, dict(found), mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fraction-numerator", type=int, default=1)
    parser.add_argument("--fraction-denominator", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite extraction root: {args.output_root}")
    wanted = tuple(dict.fromkeys(args.source))
    require(len(wanted) == len(args.source), "duplicate --source value")
    require(
        1 <= args.fraction_numerator <= args.fraction_denominator,
        "sampling fraction must satisfy 1 <= numerator <= denominator",
    )
    manifest_path = args.release_root / "release/manifests/anonymization_manifest.json"
    counts_path = args.release_root / "release/manifests/token_counts.json"
    upstream = read_json(manifest_path)
    counts = read_json(counts_path)
    expected = {source: int(counts["source_rows"][source]) for source in wanted}
    files = [args.release_root / "release" / row["path"] for row in upstream["files"]]
    value_set = pa.array(wanted, type=pa.string())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspections = list(
            executor.map(
                lambda path: inspect(
                    path,
                    value_set,
                    numerator=args.fraction_numerator,
                    denominator=args.fraction_denominator,
                    seed=args.seed,
                ),
                files,
            )
        )
    observed: collections.Counter[str] = collections.Counter()
    selected = [(path, count, found, mask) for path, count, found, mask in inspections if count]
    for _, _, found, _ in selected:
        observed.update(found)
    if args.fraction_numerator == args.fraction_denominator:
        require(dict(observed) == expected, f"selected source rows drift: {dict(observed)} != {expected}")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", suffix=".partial", dir=args.output_root.parent)
    )
    try:
        data_root = temporary_root / "data"
        data_root.mkdir()
        outputs: list[dict[str, Any]] = []
        for path, expected_rows, _, mask in selected:
            table = pq.read_table(path)
            filtered = table.filter(mask)
            require(filtered.num_rows == expected_rows, f"two-pass selection drift: {path}")
            output = data_root / path.name
            pq.write_table(filtered, output, compression="zstd", use_dictionary=True)
            reread = pq.read_table(output)
            require(reread.equals(filtered), f"Parquet round-trip changed selected rows: {output}")
            output_binding = file_binding(output)
            output_binding["path"] = str((args.output_root / "data" / path.name).resolve())
            outputs.append({
                "input": file_binding(path),
                "output": output_binding,
                "rows": filtered.num_rows,
            })
        payload = {
            "schema_version": "targeted_8b_source_extraction_v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "release_root": str(args.release_root.resolve()),
            "upstream": {"anonymization_manifest": file_binding(manifest_path), "token_counts": file_binding(counts_path)},
            "sources": list(wanted),
            "source_rows": dict(observed),
            "rows": sum(observed.values()),
            "files_scanned": len(files),
            "files_with_selected_rows": len(outputs),
            "outputs": outputs,
            "selection": {
                "method": "sha256_u64_modulo" if args.fraction_numerator < args.fraction_denominator else "all_named_source_rows",
                "seed": args.seed,
                "fraction_numerator": args.fraction_numerator,
                "fraction_denominator": args.fraction_denominator,
                "identity": ["source_dataset", "source_doc_id"],
            },
            "invariants": {
                "row_order_preserved_within_each_input_shard": True,
                "row_multiplicity_preserved": True,
                "all_values_unchanged": True,
                "global_deduplication_performed": False,
                "near_deduplication_performed": False,
                "only_named_source_filter_applied": True,
            },
        }
        write_json_atomic(temporary_root / "extraction_manifest.json", payload)
        os.rename(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(args.output_root / "extraction_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

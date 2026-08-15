#!/usr/bin/env python3
"""Freeze Parquet metadata and value inventories for the selected replay file.

This is deliberately a metadata/schema operation plus bounded column scans. It
does not materialize or rewrite replay rows. The resulting receipt is the
authority used to write the heterogeneous scanner adapter instead of guessing
column names from historical scripts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, require, write_json_atomic


def canonical_type(field: pa.Field) -> str:
    return str(field.type)


def digest_counts(counts: Counter[str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(counts.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inventory-column",
        action="append",
        default=[],
        help="scan a low-cardinality column and record exact value counts",
    )
    args = parser.parse_args()
    require(args.input_parquet.is_file(), f"replay parquet missing: {args.input_parquet}")
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    parquet = pq.ParquetFile(args.input_parquet)
    schema = parquet.schema_arrow
    requested = tuple(dict.fromkeys(args.inventory_column))
    require(len(requested) == len(args.inventory_column), "duplicate inventory column")
    for name in requested:
        require(name in schema.names, f"inventory column missing: {name}")

    inventories: dict[str, Any] = {}
    for name in requested:
        counts: Counter[str] = Counter()
        nulls = 0
        for batch in parquet.iter_batches(batch_size=262_144, columns=[name]):
            for value in batch.column(0).to_pylist():
                if value is None:
                    nulls += 1
                else:
                    counts[str(value)] += 1
        require(sum(counts.values()) + nulls == parquet.metadata.num_rows, f"{name}: row accounting drift")
        inventories[name] = {
            "null_rows": nulls,
            "distinct_non_null_values": len(counts),
            "value_counts": dict(sorted(counts.items())),
            "value_counts_sha256": digest_counts(counts),
        }

    payload = {
        "schema_version": "apertus_replay_parquet_inspection_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "input": file_binding(args.input_parquet),
        "parquet": {
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "columns": [
                {"name": field.name, "type": canonical_type(field), "nullable": field.nullable}
                for field in schema
            ],
            "created_by": parquet.metadata.created_by,
        },
        "inventories": inventories,
        "data_rows_written": 0,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

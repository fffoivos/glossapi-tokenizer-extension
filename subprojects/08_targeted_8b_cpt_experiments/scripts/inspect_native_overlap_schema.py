#!/usr/bin/env python3
"""Freeze Parquet schema and safe identifier-only samples for overlap evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, require, sha256_file, write_json_atomic


EXPECTED_SHA256 = "1b23a9dc14a6175c18e0530210cc47795e24e3841bb1b3229c666877ac4b4b19"
SAFE_COLUMNS = {
    "benchmark", "benchmark_id", "benchmark_family", "task", "task_id",
    "example_id", "query_id", "match_class", "match_kind", "strict",
    "strict_match", "recommended_exclusion", "dataset_shard", "shard",
    "dataset_row", "row_index", "source_dataset", "source_doc_id",
    "document_id", "normalized_text_sha256", "text_sha256",
}


def scalar(value):
    return value.as_py() if hasattr(value, "as_py") else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable output exists: {args.output}")
    require(args.match_table.is_file(), f"match table missing: {args.match_table}")
    digest = sha256_file(args.match_table)
    require(digest == EXPECTED_SHA256, "match-table SHA-256 drift")
    parquet = pq.ParquetFile(args.match_table)
    names = parquet.schema_arrow.names
    safe = [name for name in names if name in SAFE_COLUMNS]
    sample = []
    if safe and parquet.metadata.num_rows:
        table = parquet.read_row_group(0, columns=safe).slice(0, 5)
        for index in range(table.num_rows):
            sample.append({name: scalar(table[name][index]) for name in safe})
    row_groups = []
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        row_groups.append({"index": index, "rows": group.num_rows, "bytes": group.total_byte_size})
    payload = {
        "schema_version": "apertus_native_overlap_schema_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "match_table": {
            "path": str(args.match_table.resolve()),
            "bytes": args.match_table.stat().st_size,
            "sha256": digest,
            "rows": parquet.metadata.num_rows,
            "row_groups": row_groups,
            "schema": str(parquet.schema_arrow),
            "columns": names,
        },
        "safe_identifier_columns": safe,
        "identifier_samples": sample,
        "text_or_snippet_materialized": False,
        "data_rows_written": 0,
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

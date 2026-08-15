#!/usr/bin/env python3
"""Flatten the complete benchmark-clean OpenArchives source view for Phase 3.

This is not a second mix and does not alter the Phase-2 selection.  It exposes
all rows surviving the pinned v2/native-suite/heldout/Greek-replay exclusions
so the post-3218 gate can remove actually realized Phase-2 documents and prove
enough one-pass extension capacity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view-receipt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_jsonl.exists() and not args.output_receipt.exists(), "immutable Phase-3 OpenArchives candidate output exists")
    source = read_json(args.source_view_receipt)
    require(source.get("schema_version") == "apertus_hard_h_to_g_source_views_v1" and source.get("status") == "passed", "source-view receipt drift")
    pool = source.get("pools", {}).get("openarchives")
    require(isinstance(pool, dict), "OpenArchives source view missing")
    files = pool.get("output_files")
    require(isinstance(files, list) and files, "OpenArchives source-view file inventory missing")
    expected_rows = int(pool.get("counts", {}).get("kept_rows", -1))
    require(expected_rows > 0, "OpenArchives source view is empty")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{args.output_jsonl.name}.", suffix=".partial", dir=args.output_jsonl.parent)
    temporary = Path(name)
    rows = 0
    source_dataset_counts: dict[str, int] = {}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for item in files:
                path = Path(str(item.get("output", {}).get("path", "")))
                require(path.is_file() and file_binding(path) == item.get("output"), f"source-view payload drift: {path}")
                parquet = pq.ParquetFile(path)
                required = {"text", "source_dataset", "source_doc_id", "_source_release_shard", "_source_release_row_index"}
                require(required.issubset(parquet.schema_arrow.names), f"source-view columns drift: {path}")
                columns = sorted(required | ({"source_metadata_json"} if "source_metadata_json" in parquet.schema_arrow.names else set()))
                for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
                    for row in batch.to_pylist():
                        text = row.get("text")
                        source_dataset = row.get("source_dataset")
                        source_doc_id = row.get("source_doc_id")
                        require(isinstance(text, str) and text and source_dataset not in (None, "") and source_doc_id not in (None, ""), f"invalid OpenArchives candidate row: {path}:{rows}")
                        normalized = {
                            "text": text,
                            "source": str(source_dataset),
                            "doc_id": str(source_doc_id),
                            "source_dataset": str(source_dataset),
                            "source_doc_id": str(source_doc_id),
                            "source_metadata_json": row.get("source_metadata_json"),
                            "_source_release_shard": str(row["_source_release_shard"]),
                            "_source_release_row_index": int(row["_source_release_row_index"]),
                        }
                        output.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
                        rows += 1
                        source_dataset_counts[str(source_dataset)] = source_dataset_counts.get(str(source_dataset), 0) + 1
            output.flush(); os.fsync(output.fileno())
        require(rows == expected_rows, f"OpenArchives candidate row drift: {rows} != {expected_rows}")
        os.link(temporary, args.output_jsonl); temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": "apertus_hard_h_to_g_openarchives_candidate_source_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "pool": "openarchives",
        "source_view_receipt": file_binding(args.source_view_receipt),
        "actual_rows": rows,
        "output": {**file_binding(args.output_jsonl), "rows": rows},
        "source_dataset_rows": dict(sorted(source_dataset_counts.items())),
        "invariants": {
            "all_benchmark_clean_source_view_rows_included": True,
            "row_order_preserved": True,
            "text_transformed": False,
            "additional_deduplication": False,
            "phase2_selection_unchanged": True,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

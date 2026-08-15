#!/usr/bin/env python3
"""Build exact native-suite training-document exclusions from published matches.

Only rows with ``recommended_exclusion == true`` are selected. Every selected
coordinate is verified against the audited HF release using shard, zero-based
row, source identifiers, and SHA-256 of the stored UTF-8 text. No text scan,
near-duplicate expansion, or additional deduplication is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, require, sha256_file, write_json_atomic


EXPECTED_MATCH_SHA256 = "1b23a9dc14a6175c18e0530210cc47795e24e3841bb1b3229c666877ac4b4b19"
EXPECTED_DATASET_REVISION = "987b8955fcd395c6219e39df9e64715457f69065"
MATCH_COLUMNS = [
    "benchmark", "dataset_row_index", "dataset_shard", "document_key_sha256",
    "document_text_sha256", "example_id", "evaluation_unit_id",
    "match_category", "match_strength", "recommended_exclusion",
    "source_dataset", "source_doc_id",
]


def value_at(batch: pa.RecordBatch, name: str, index: int):
    return batch.column(batch.schema.get_field_index(name))[index].as_py()


def load_exact_exclusions(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    parquet = pq.ParquetFile(path)
    require(set(MATCH_COLUMNS).issubset(parquet.schema_arrow.names), "match-table schema drift")
    exclusions: dict[tuple[str, int], dict[str, Any]] = {}
    trace_counts: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=131_072, columns=MATCH_COLUMNS):
        mask = pc.fill_null(batch.column(batch.schema.get_field_index("recommended_exclusion")), False)
        selected = batch.filter(mask)
        for index in range(selected.num_rows):
            shard = str(value_at(selected, "dataset_shard", index))
            row_index = int(value_at(selected, "dataset_row_index", index))
            key = (shard, row_index)
            benchmark = str(value_at(selected, "benchmark", index))
            trace_counts[benchmark] += 1
            identity = {
                "dataset_shard": shard,
                "dataset_row_index": row_index,
                "source_dataset": value_at(selected, "source_dataset", index),
                "source_doc_id": value_at(selected, "source_doc_id", index),
                "document_key_sha256": value_at(selected, "document_key_sha256", index),
                "document_text_sha256": value_at(selected, "document_text_sha256", index),
            }
            current = exclusions.get(key)
            if current is None:
                exclusions[key] = {
                    **identity,
                    "benchmarks": {benchmark},
                    "example_ids": {str(value_at(selected, "example_id", index))},
                    "evaluation_unit_ids": {str(value_at(selected, "evaluation_unit_id", index))},
                    "match_categories": {str(value_at(selected, "match_category", index))},
                    "match_strengths": {str(value_at(selected, "match_strength", index))},
                    "published_trace_rows": 1,
                }
            else:
                for field, expected in identity.items():
                    require(current[field] == expected, f"published identity disagreement for {key}: {field}")
                current["benchmarks"].add(benchmark)
                current["example_ids"].add(str(value_at(selected, "example_id", index)))
                current["evaluation_unit_ids"].add(str(value_at(selected, "evaluation_unit_id", index)))
                current["match_categories"].add(str(value_at(selected, "match_category", index)))
                current["match_strengths"].add(str(value_at(selected, "match_strength", index)))
                current["published_trace_rows"] += 1
    require(exclusions, "published match table produced zero recommended exclusions")
    return exclusions, dict(sorted(trace_counts.items()))


def raw_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_release_rows(release_root: Path, exclusions: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    by_shard: dict[str, list[int]] = {}
    for shard, row_index in exclusions:
        by_shard.setdefault(shard, []).append(row_index)
    source_counts: Counter[str] = Counter()
    verified = 0
    for shard, indices in sorted(by_shard.items()):
        path = release_root / "release" / shard
        require(path.is_file(), f"audited release shard missing: {path}")
        table = pq.read_table(path, columns=["text", "source_dataset", "source_doc_id"])
        for row_index in sorted(set(indices)):
            require(0 <= row_index < table.num_rows, f"published row out of range: {shard}:{row_index}")
            expected = exclusions[(shard, row_index)]
            text = table["text"][row_index].as_py()
            source_dataset = table["source_dataset"][row_index].as_py()
            source_doc_id = table["source_doc_id"][row_index].as_py()
            require(source_dataset == expected["source_dataset"], f"source_dataset drift: {shard}:{row_index}")
            require(source_doc_id == expected["source_doc_id"], f"source_doc_id drift: {shard}:{row_index}")
            require(isinstance(text, str), f"non-text release row: {shard}:{row_index}")
            require(raw_text_sha256(text) == expected["document_text_sha256"], f"text SHA-256 drift: {shard}:{row_index}")
            source_counts[str(source_dataset)] += 1
            verified += 1
    require(verified == len(exclusions), "verified-row accounting drift")
    return {
        "documents": verified,
        "shards": len(by_shard),
        "source_datasets": dict(sorted(source_counts.items())),
    }


def write_manifest(path: Path, exclusions: dict[tuple[str, int], dict[str, Any]]) -> None:
    require(not path.exists(), f"immutable exclusion manifest exists: {path}")
    rows = []
    for key in sorted(exclusions):
        row = exclusions[key]
        rows.append({
            "dataset_revision": EXPECTED_DATASET_REVISION,
            "dataset_shard": row["dataset_shard"],
            "dataset_row_index": row["dataset_row_index"],
            "source_dataset": row["source_dataset"],
            "source_doc_id": row["source_doc_id"],
            "document_key_sha256": row["document_key_sha256"],
            "document_text_sha256": row["document_text_sha256"],
            "benchmarks": sorted(row["benchmarks"]),
            "example_ids": sorted(row["example_ids"]),
            "evaluation_unit_ids": sorted(row["evaluation_unit_ids"]),
            "match_categories": sorted(row["match_categories"]),
            "match_strengths": sorted(row["match_strengths"]),
            "published_trace_rows": row["published_trace_rows"],
            "exclusion_authority": "published_recommended_exclusion_true",
        })
    table = pa.Table.from_pylist(rows)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-table", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_receipt.exists(), f"immutable receipt exists: {args.output_receipt}")
    require(sha256_file(args.match_table) == EXPECTED_MATCH_SHA256, "match-table SHA-256 drift")
    exclusions, trace_counts = load_exact_exclusions(args.match_table)
    verification = verify_release_rows(args.release_root, exclusions)
    write_manifest(args.output_manifest, exclusions)
    payload = {
        "schema_version": "apertus_native_suite_training_exclusions_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "nodes": int(os.environ.get("SLURM_NNODES", "0")),
        },
        "executing_code_bundle": executing_code_bundle(),
        "dataset": {"revision": EXPECTED_DATASET_REVISION, "release_root": str(args.release_root.resolve())},
        "published_match_table": {
            "path": str(args.match_table.resolve()),
            "bytes": args.match_table.stat().st_size,
            "sha256": EXPECTED_MATCH_SHA256,
            "recommended_trace_rows_by_benchmark": trace_counts,
        },
        "exclusion_manifest": {
            "path": str(args.output_manifest.resolve()),
            "bytes": args.output_manifest.stat().st_size,
            "sha256": sha256_file(args.output_manifest),
            "documents": len(exclusions),
        },
        "release_coordinate_verification": verification,
        "policy": {
            "recommended_exclusion_true_only": True,
            "question_only_candidates_excluded": False,
            "near_duplicate_expansion": False,
            "additional_global_deduplication": False,
            "full_dataset_rescan": False,
        },
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

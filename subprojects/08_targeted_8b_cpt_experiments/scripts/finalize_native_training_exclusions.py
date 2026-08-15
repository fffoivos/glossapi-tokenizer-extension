#!/usr/bin/env python3
"""Validate canonical scanner shards and freeze exact strong-match document exclusions."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--query-authority", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_parquet, args.output_receipt):
        require(not output.exists(), f"immutable native exclusion output exists: {output}")
    query_authority = read_json(args.query_authority)
    require(query_authority.get("schema_version") == "apertus_native_suite_scan_authority_v1", "query authority schema drift")
    require(query_authority.get("status") == "frozen", "query authority did not freeze")
    queries_sha = sha256_file(args.queries_jsonl)
    require(query_authority.get("queries", {}).get("sha256") == queries_sha, "query authority binding drift")
    manifest = read_json(args.corpus_manifest)
    publication = read_json(args.publication_receipt)
    manifest_sha = sha256_file(args.corpus_manifest)
    publication_sha = sha256_file(args.publication_receipt)
    require(manifest.get("status") == "passed" and publication.get("status") == "passed", "scan corpus authority failed")
    require(publication.get("manifest", {}).get("sha256") == manifest_sha, "publication/manifest binding drift")

    excluded: dict[tuple[str, str, str], dict[str, object]] = {}
    trace_counts: Counter[str] = Counter()
    match_rows = strong_rows = 0
    shard_receipts = []
    for item in manifest["files"]:
        stem = Path(item["path"]).stem
        receipt_path = args.scan_root / "shards" / f"{stem}.receipt.json"
        matches_path = args.scan_root / "shards" / f"{stem}.matches.jsonl"
        require(receipt_path.is_file() and matches_path.is_file(), f"scan shard artifacts missing: {stem}")
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "passed", f"scan shard failed: {stem}")
        require(receipt.get("queries_sha256") == queries_sha, f"scan query binding drift: {stem}")
        require(receipt.get("corpus_manifest_sha256") == manifest_sha, f"scan manifest binding drift: {stem}")
        require(receipt.get("publication_receipt_sha256") == publication_sha, f"scan publication binding drift: {stem}")
        require(receipt.get("input", {}).get("path") == item["path"], f"scan input path drift: {stem}")
        require(receipt.get("input", {}).get("sha256") == item["sha256"], f"scan input SHA drift: {stem}")
        require(receipt.get("output", {}).get("sha256") == sha256_file(matches_path), f"scan output SHA drift: {stem}")
        shard_receipts.append(file_binding(receipt_path))
        with matches_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                match_rows += 1
                if row.get("recommended_exclusion") is not True:
                    continue
                strong_rows += 1
                key = (str(row["source_dataset"]), str(row["source_doc_id"]), str(row["document_text_sha256"]))
                block = excluded.setdefault(key, {
                    "source_dataset": key[0],
                    "source_doc_id": key[1],
                    "document_text_sha256": key[2],
                    "benchmarks": set(),
                    "evaluation_unit_ids": set(),
                    "strong_match_rows": 0,
                })
                block["benchmarks"].add(str(row["benchmark"]))  # type: ignore[union-attr]
                block["evaluation_unit_ids"].add(str(row["evaluation_unit_id"]))  # type: ignore[union-attr]
                block["strong_match_rows"] = int(block["strong_match_rows"]) + 1
                trace_counts[str(row["benchmark"])] += 1
    rows = []
    for key in sorted(excluded):
        block = excluded[key]
        rows.append({
            "source_dataset": block["source_dataset"],
            "source_doc_id": block["source_doc_id"],
            "document_text_sha256": block["document_text_sha256"],
            "benchmarks": sorted(block["benchmarks"]),
            "evaluation_unit_ids": sorted(block["evaluation_unit_ids"]),
            "strong_match_rows": block["strong_match_rows"],
        })
    schema = pa.schema([
        ("source_dataset", pa.string()), ("source_doc_id", pa.string()),
        ("document_text_sha256", pa.string()), ("benchmarks", pa.list_(pa.string())),
        ("evaluation_unit_ids", pa.list_(pa.string())), ("strong_match_rows", pa.int64()),
    ])
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_parquet.with_name(args.output_parquet.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    os.replace(temporary, args.output_parquet)
    payload = {
        "schema_version": "apertus_native_suite_training_scan_exclusions_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodes": int(os.environ.get("SLURM_NNODES", "0"))},
        "executing_code_bundle": executing_code_bundle(),
        "queries": file_binding(args.queries_jsonl),
        "query_authority": file_binding(args.query_authority),
        "corpus_manifest": file_binding(args.corpus_manifest),
        "publication_receipt": file_binding(args.publication_receipt),
        "scan_root": str(args.scan_root.resolve()),
        "scan_shard_receipts": shard_receipts,
        "counts": {"match_rows": match_rows, "strong_match_rows": strong_rows, "excluded_documents": len(rows), "strong_rows_by_benchmark": dict(sorted(trace_counts.items()))},
        "exclusions": {**file_binding(args.output_parquet), "rows": len(rows)},
        "policy": {"recommended_exclusion_true_only": True, "candidate_only_excluded": False, "cluster_expansion": False, "additional_deduplication": False},
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

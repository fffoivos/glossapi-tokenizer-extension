#!/usr/bin/env python3
"""Compact strong-match evidence to one auditable row per evaluation unit."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-parquet", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def rank(row: dict[str, Any]) -> tuple[int, str]:
    identity = "\0".join(
        str(row.get(key) or "")
        for key in (
            "benchmark",
            "evaluation_unit_id",
            "source_dataset",
            "source_doc_id",
            "dataset_shard",
            "dataset_row_index",
        )
    )
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big"), identity


def main() -> int:
    import pyarrow.parquet as pq

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output_jsonl.exists() or args.summary_json.exists() or args.receipt.exists():
        raise FileExistsError("adjudication output already exists")
    queries: dict[tuple[str, str], dict[str, Any]] = {}
    with args.queries_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["benchmark"]), str(row["evaluation_unit_id"]))
            if key in queries:
                raise ValueError(f"duplicate query identity: {key}")
            queries[key] = row

    columns = [
        "benchmark",
        "evaluation_unit_id",
        "match_category",
        "recommended_exclusion",
        "source_dataset",
        "source_doc_id",
        "document_url",
        "dataset_shard",
        "dataset_row_index",
        "evidence_line_start",
        "evidence_line_end",
        "evidence_snippet",
    ]
    units: dict[tuple[str, str], dict[str, Any]] = {}
    strong_rows = 0
    for batch in pq.ParquetFile(args.matches_parquet).iter_batches(batch_size=args.batch_size, columns=columns):
        for row in batch.to_pylist():
            if not row["recommended_exclusion"]:
                continue
            strong_rows += 1
            key = (str(row["benchmark"]), str(row["evaluation_unit_id"]))
            unit = units.setdefault(
                key,
                {
                    "strong_match_rows": 0,
                    "source_datasets": set(),
                    "source_documents": set(),
                    "match_categories": collections.Counter(),
                    "representative_rank": None,
                    "representative": None,
                },
            )
            unit["strong_match_rows"] += 1
            unit["source_datasets"].add(str(row["source_dataset"]))
            unit["source_documents"].add((str(row["source_dataset"]), str(row["source_doc_id"])))
            unit["match_categories"][str(row["match_category"])] += 1
            measured_rank = rank(row)
            if unit["representative_rank"] is None or measured_rank < unit["representative_rank"]:
                unit["representative_rank"] = measured_rank
                unit["representative"] = row

    output: list[dict[str, Any]] = []
    benchmark_units: collections.Counter[str] = collections.Counter()
    benchmark_scored_examples: collections.Counter[str] = collections.Counter()
    benchmark_source_units: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for key in sorted(units):
        benchmark, evaluation_unit_id = key
        query = queries[key]
        unit = units[key]
        answer_index = int(query["answer_index"])
        for source_dataset in unit["source_datasets"]:
            benchmark_source_units[benchmark][source_dataset] += 1
        benchmark_units[benchmark] += 1
        benchmark_scored_examples[benchmark] += len(query["discount_example_ids"])
        representative = dict(unit["representative"])
        output.append(
            {
                "benchmark": benchmark,
                "evaluation_unit_id": evaluation_unit_id,
                "discount_example_ids": query["discount_example_ids"],
                "query_kind": query["query_kind"],
                "question_or_first_surface": query["question"],
                "correct_answer_or_second_surface": query["choices"][answer_index],
                "strong_match_rows": unit["strong_match_rows"],
                "strong_source_documents": len(unit["source_documents"]),
                "strong_source_datasets": sorted(unit["source_datasets"]),
                "match_categories": dict(sorted(unit["match_categories"].items())),
                "representative_evidence": representative,
            }
        )
    atomic_write(
        args.output_jsonl,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
    )
    summary = {
        "schema_version": "greek_benchmark_contamination_adjudication_packet_v1",
        "status": "passed",
        "policy": "one deterministic representative for every two-surface strong-match unit",
        "strong_match_rows": strong_rows,
        "strong_evaluation_units": len(output),
        "recommended_excluded_scored_examples": sum(benchmark_scored_examples.values()),
        "benchmarks": {
            benchmark: {
                "strong_evaluation_units": benchmark_units[benchmark],
                "recommended_excluded_scored_examples": benchmark_scored_examples[benchmark],
                "source_dataset_units": dict(benchmark_source_units[benchmark].most_common()),
            }
            for benchmark in sorted(benchmark_units)
        },
    }
    atomic_write(args.summary_json, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "greek_benchmark_contamination_adjudication_packet_receipt_v1",
        "status": "passed",
        "inputs": {
            "matches_parquet": {"path": str(args.matches_parquet), "sha256": sha256(args.matches_parquet)},
            "queries_jsonl": {"path": str(args.queries_jsonl), "sha256": sha256(args.queries_jsonl)},
        },
        "outputs": {
            "jsonl": {"path": str(args.output_jsonl), "rows": len(output), "sha256": sha256(args.output_jsonl)},
            "summary": {"path": str(args.summary_json), "sha256": sha256(args.summary_json)},
        },
    }
    atomic_write(args.receipt, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "units": len(output), "scored_examples": sum(benchmark_scored_examples.values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

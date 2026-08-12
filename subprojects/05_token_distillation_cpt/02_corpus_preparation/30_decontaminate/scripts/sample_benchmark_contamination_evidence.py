#!/usr/bin/env python3
"""Create a deterministic, query-joined review sample from match evidence."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-parquet", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--per-benchmark-category", type=int, default=20)
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


def stable_rank(row: dict[str, Any]) -> tuple[int, str]:
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
    if args.per_benchmark_category < 1 or args.batch_size < 1:
        raise ValueError("sampling and batch sizes must be positive")
    if args.output_jsonl.exists() or args.receipt.exists():
        raise FileExistsError("review output already exists")
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
        "match_strength",
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
    # Each heap retains the rows with the smallest stable hashes. Negating the
    # rank makes the largest retained rank the root, so replacement is O(log n).
    samples: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = {}
    scanned = 0
    parquet = pq.ParquetFile(args.matches_parquet)
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
        for row in batch.to_pylist():
            scanned += 1
            group = (str(row["benchmark"]), str(row["match_category"]))
            rank, identity = stable_rank(row)
            heap = samples.setdefault(group, [])
            item = (-rank, identity, row)
            if len(heap) < args.per_benchmark_category:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    output: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for (benchmark, category), heap in sorted(samples.items()):
        selected = sorted(((-negative_rank, identity, row) for negative_rank, identity, row in heap))
        group_counts[f"{benchmark}:{category}"] = len(selected)
        for rank, _, row in selected:
            query = queries[(benchmark, str(row["evaluation_unit_id"]))]
            answer_index = int(query["answer_index"])
            output.append(
                {
                    **row,
                    "stable_sample_rank": str(rank),
                    "query_kind": query["query_kind"],
                    "question_or_first_surface": query["question"],
                    "correct_answer_or_second_surface": query["choices"][answer_index],
                }
            )
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output)
    atomic_write(args.output_jsonl, payload)
    receipt = {
        "schema_version": "greek_benchmark_contamination_review_sample_v1",
        "status": "passed",
        "method": "smallest stable SHA-256 ranks per benchmark and match category",
        "per_benchmark_category": args.per_benchmark_category,
        "match_rows_scanned": scanned,
        "query_units": len(queries),
        "group_counts": group_counts,
        "inputs": {
            "matches_parquet": {"path": str(args.matches_parquet), "sha256": sha256(args.matches_parquet)},
            "queries_jsonl": {"path": str(args.queries_jsonl), "sha256": sha256(args.queries_jsonl)},
        },
        "output": {
            "path": str(args.output_jsonl),
            "rows": len(output),
            "sha256": sha256(args.output_jsonl),
        },
    }
    atomic_write(args.receipt, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "rows": len(output), "match_rows_scanned": scanned}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

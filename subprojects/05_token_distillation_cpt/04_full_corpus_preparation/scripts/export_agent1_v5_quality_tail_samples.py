#!/usr/bin/env python3
"""Export bounded deterministic samples of Agent-1 v5 quality-filter failures."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "agent1_v5_quality_tail_sample_v1"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _score(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return result if math.isfinite(result) else float("-inf")


def _bounded_push(
    heap: list[tuple[int, str, dict[str, Any]]],
    key: int,
    tie: str,
    row: dict[str, Any],
    limit: int,
) -> None:
    entry = (-key, tie, row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def scan_file(payload: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = Path(str(payload["path"]))
    limit = int(payload["limit"])
    random_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    worst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(
        columns=[
            "source_dataset",
            "source_doc_id",
            "title",
            "text",
            "filter",
            "greek_percentage",
            "latin_percentage",
            "greek_badness_score",
            "mojibake_badness_score",
            "chars",
            "approx_word_count",
        ],
        batch_size=2048,
    ):
        for row in batch.to_pylist():
            row_index += 1
            filter_value = str(row.get("filter") or "")
            if filter_value == "ok":
                continue
            source = str(row.get("source_dataset") or "")
            document_id = str(row.get("source_doc_id") or "")
            text = str(row.get("text") or "")
            counts[f"{source}\0{filter_value}"] += 1
            digest = hashlib.sha256(f"{source}\0{document_id}".encode("utf-8")).digest()
            key = int.from_bytes(digest[:8], "big")
            sample = {
                "sample_key": digest[:8].hex(),
                "source_dataset": source,
                "source_doc_id_sha256": hashlib.sha256(
                    document_id.encode("utf-8")
                ).hexdigest(),
                "filter": filter_value,
                "title": str(row.get("title") or "")[:300] or None,
                "chars": int(row.get("chars") or len(text)),
                "approx_word_count": int(row.get("approx_word_count") or 0),
                "greek_percentage": row.get("greek_percentage"),
                "latin_percentage": row.get("latin_percentage"),
                "greek_badness_score": row.get("greek_badness_score"),
                "mojibake_badness_score": row.get("mojibake_badness_score"),
                "text_preview": text[:2400],
            }
            # The path/row suffix keeps the heap key unique even if a source
            # accidentally contains duplicate document IDs.  Without it,
            # Python may try to compare the trailing dictionaries.
            tie = f"{path.name}:{row_index:012d}:{sample['source_doc_id_sha256']}"
            _bounded_push(random_heaps[source], key, tie, sample, limit)
            worst[source].append(sample)
            worst[source] = sorted(
                worst[source],
                key=lambda value: _score(value.get("greek_badness_score")),
                reverse=True,
            )[:limit]
    return {
        "path": str(path),
        "counts": dict(counts),
        "random": {
            source: [row for _, _, row in heap] for source, heap in random_heaps.items()
        },
        "worst": dict(worst),
    }


def merge_bounded(
    results: Sequence[Mapping[str, Any]], name: str, limit: int, *, worst: bool = False
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for source, rows in result[name].items():
            merged[source].extend(rows)
    if worst:
        return {
            source: sorted(
                rows,
                key=lambda row: _score(row.get("greek_badness_score")),
                reverse=True,
            )[:limit]
            for source, rows in sorted(merged.items())
        }
    return {
        source: sorted(rows, key=lambda row: row["sample_key"])[:limit]
        for source, rows in sorted(merged.items())
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    if args.samples_per_source <= 0 or args.workers <= 0:
        raise ValueError("sample and worker counts must be positive")
    run_root = args.run_root.resolve(strict=True)
    release_root = run_root / "release-pre-dedup"
    manifest = read_object(release_root / "manifests" / "combined_manifest.json")
    candidate = [row for row in manifest["files"] if row.get("origin") == "candidate"]
    jobs = [
        {
            "path": str((release_root / str(row["path"])).resolve(strict=True)),
            "limit": args.samples_per_source,
        }
        for row in candidate
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(scan_file, jobs))
    counts: Counter[str] = Counter()
    for result in results:
        counts.update({key: int(value) for key, value in result["counts"].items()})
    by_source: dict[str, dict[str, int]] = defaultdict(dict)
    for key, value in sorted(counts.items()):
        source, filter_value = key.split("\0", 1)
        by_source[source][filter_value] = value
    receipt = {
        "schema_version": SCHEMA,
        "status": "passed",
        "run_id": manifest["run_id"],
        "candidate_files": len(candidate),
        "filter_failure_rows": sum(counts.values()),
        "filter_failure_counts": dict(sorted(by_source.items())),
        "deterministic_samples": merge_bounded(
            results, "random", args.samples_per_source
        ),
        "worst_score_samples": merge_bounded(
            results, "worst", args.samples_per_source, worst=True
        ),
        "text_scope": "private truncated previews for human quality review; do not publish",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    temporary.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(canonical_json({"ok": True, "rows": receipt["filter_failure_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

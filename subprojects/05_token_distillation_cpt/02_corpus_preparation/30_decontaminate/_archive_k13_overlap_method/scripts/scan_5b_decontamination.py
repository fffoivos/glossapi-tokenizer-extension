#!/usr/bin/env python3
"""Stream the Task-1 5B JSONL and audit n-gram benchmark contamination."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
DEFAULT_THRESHOLDS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class SurfaceKey:
    benchmark: str
    example_id: str
    subject: str | None
    surface: str
    k: int
    query_shingles: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", default="8,13", help="Comma-separated shingle sizes.")
    parser.add_argument("--thresholds", default=",".join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--progress-every-rows", type=int, default=100_000)
    parser.add_argument("--max-hit-records", type=int, default=500_000)
    parser.add_argument("--max-span-chars", type=int, default=300)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return text.casefold()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def shingles(tokens: list[str], k: int) -> Iterable[tuple[str, ...]]:
    if k <= 0 or len(tokens) < k:
        return
    for idx in range(0, len(tokens) - k + 1):
        yield tuple(tokens[idx : idx + k])


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_queries(path: Path, ks: list[int]) -> tuple[dict[tuple[str, ...], list[int]], list[SurfaceKey], dict[str, Any]]:
    gram_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    surface_keys: list[SurfaceKey] = []
    item_counts: dict[str, set[str]] = defaultdict(set)
    surface_counts: dict[str, int] = defaultdict(int)
    zero_shingle_surfaces = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            benchmark = row["benchmark"]
            example_id = str(row["example_id"])
            subject = row.get("subject")
            item_counts[benchmark].add(example_id)
            surfaces = row.get("surfaces") or {}
            for surface_name, surface_text in surfaces.items():
                surface_counts[benchmark] += 1
                tokens = tokenize(str(surface_text or ""))
                for k in ks:
                    query_grams = set(shingles(tokens, k))
                    key_id = len(surface_keys)
                    surface_keys.append(
                        SurfaceKey(
                            benchmark=benchmark,
                            example_id=example_id,
                            subject=subject,
                            surface=surface_name,
                            k=k,
                            query_shingles=len(query_grams),
                        )
                    )
                    if not query_grams:
                        zero_shingle_surfaces += 1
                        continue
                    for gram in query_grams:
                        gram_index[gram].append(key_id)

    summary = {
        "query_file": str(path),
        "k": ks,
        "benchmarks": {
            benchmark: {
                "items": len(items),
                "surfaces": surface_counts.get(benchmark, 0),
            }
            for benchmark, items in sorted(item_counts.items())
        },
        "surface_keys": len(surface_keys),
        "indexed_unique_grams": len(gram_index),
        "zero_shingle_surface_keys": zero_shingle_surfaces,
    }
    return gram_index, surface_keys, summary


def first_span(tokens: list[str], gram: tuple[str, ...], max_chars: int) -> dict[str, Any]:
    k = len(gram)
    for idx in range(0, max(0, len(tokens) - k + 1)):
        if tuple(tokens[idx : idx + k]) == gram:
            return {
                "token_start": idx,
                "token_end": idx + k,
                "normalized_text": " ".join(tokens[idx : idx + k])[:max_chars],
            }
    return {"token_start": None, "token_end": None, "normalized_text": " ".join(gram)[:max_chars]}


def main() -> None:
    args = parse_args()
    ks = parse_csv_ints(args.k)
    thresholds = parse_csv_floats(args.thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hits_jsonl = args.output_dir / "hits.jsonl"
    audit_json = args.output_dir / "decontamination_audit_5b.json"

    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gram_index, surface_keys, query_summary = load_queries(args.queries_jsonl, ks)
    print(json.dumps({"loaded_queries": query_summary}, ensure_ascii=False, indent=2), flush=True)

    item_best: dict[tuple[str, str, int], dict[str, Any]] = {}
    item_any_docs: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    surface_hit_counts: dict[tuple[str, str, str, int], int] = defaultdict(int)
    benchmark_docs: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, int] = defaultdict(int)
    hit_records_written = 0
    hit_records_seen = 0
    rows = 0
    bytes_read = 0
    t0 = time.time()

    with args.corpus_jsonl.open(encoding="utf-8") as corpus, hits_jsonl.open("w", encoding="utf-8") as hit_out:
        for line in corpus:
            if not line.strip():
                continue
            rows += 1
            bytes_read += len(line.encode("utf-8"))
            row = json.loads(line)
            text = str(row.get(args.text_key) or "")
            if not text:
                continue
            doc_id = str(row.get("doc_id") or f"row_{rows - 1}")
            source = str(row.get("source") or "")
            source_counts[source] += 1
            tokens = tokenize(text)
            if not tokens:
                continue

            # Per-document unique query shingle matches keyed by surface id.
            doc_matches: dict[int, set[tuple[str, ...]]] = defaultdict(set)
            first_match: dict[int, tuple[str, ...]] = {}
            for k in ks:
                if len(tokens) < k:
                    continue
                for idx in range(0, len(tokens) - k + 1):
                    gram = tuple(tokens[idx : idx + k])
                    key_ids = gram_index.get(gram)
                    if not key_ids:
                        continue
                    for key_id in key_ids:
                        doc_matches[key_id].add(gram)
                        first_match.setdefault(key_id, gram)

            if doc_matches:
                text_sha = sha256_text(text)
            for key_id, matched_grams in doc_matches.items():
                key = surface_keys[key_id]
                if key.query_shingles <= 0:
                    continue
                matched = len(matched_grams)
                overlap_fraction = matched / key.query_shingles
                item_key = (key.benchmark, key.example_id, key.k)
                surface_key = (key.benchmark, key.example_id, key.surface, key.k)
                doc_key = f"{source}:{doc_id}"
                item_any_docs[item_key].add(doc_key)
                benchmark_docs[key.benchmark].add(doc_key)
                surface_hit_counts[surface_key] += 1
                prev = item_best.get(item_key)
                if prev is None or overlap_fraction > prev["overlap_fraction"]:
                    item_best[item_key] = {
                        "benchmark": key.benchmark,
                        "example_id": key.example_id,
                        "subject": key.subject,
                        "k": key.k,
                        "surface": key.surface,
                        "overlap_fraction": overlap_fraction,
                        "matched_query_shingles": matched,
                        "query_shingles": key.query_shingles,
                        "doc_id": doc_id,
                        "source": source,
                        "doc_row_index": rows - 1,
                    }

                hit_records_seen += 1
                if hit_records_written < args.max_hit_records:
                    span = first_span(tokens, first_match[key_id], args.max_span_chars)
                    record = {
                        "schema": "greek-mcq-decontam-hit-v1",
                        "benchmark": key.benchmark,
                        "example_id": key.example_id,
                        "subject": key.subject,
                        "surface": key.surface,
                        "k": key.k,
                        "doc_row_index": rows - 1,
                        "doc_id": doc_id,
                        "source": source,
                        "text_sha256": text_sha,
                        "matched_query_shingles": matched,
                        "query_shingles": key.query_shingles,
                        "overlap_fraction": overlap_fraction,
                        "first_match": span,
                    }
                    hit_out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    hit_records_written += 1

            if args.progress_every_rows > 0 and rows % args.progress_every_rows == 0:
                elapsed = max(time.time() - t0, 1e-9)
                print(
                    f"rows={rows:,} hit_surfaces={hit_records_seen:,} "
                    f"written={hit_records_written:,} rate={rows / elapsed:,.1f} rows/s",
                    flush=True,
                )

    elapsed = time.time() - t0

    item_counts: dict[str, int] = {}
    for benchmark, data in query_summary["benchmarks"].items():
        item_counts[benchmark] = int(data["items"])

    threshold_summary: dict[str, Any] = {}
    for benchmark, total_items in sorted(item_counts.items()):
        threshold_summary[benchmark] = {}
        for k in ks:
            best_for_benchmark = [
                best
                for (b, _example_id, kk), best in item_best.items()
                if b == benchmark and kk == k
            ]
            for threshold in thresholds:
                if threshold <= 0.0:
                    contaminated = len(best_for_benchmark)
                else:
                    contaminated = sum(1 for best in best_for_benchmark if best["overlap_fraction"] >= threshold)
                threshold_summary[benchmark][f"k{k}_threshold_{threshold:g}"] = {
                    "contaminated_items": contaminated,
                    "total_items": total_items,
                    "contaminated_fraction": contaminated / total_items if total_items else None,
                }

    best_hits = sorted(
        item_best.values(),
        key=lambda row: (row["benchmark"], row["k"], -row["overlap_fraction"], row["example_id"]),
    )
    top_hits_path = args.output_dir / "top_item_hits.jsonl"
    with top_hits_path.open("w", encoding="utf-8") as fh:
        for row in best_hits:
            item_key = (row["benchmark"], row["example_id"], row["k"])
            out = dict(row)
            out["matching_docs"] = len(item_any_docs[item_key])
            fh.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")

    audit = {
        "schema": "greek-mcq-5b-decontamination-audit-v1",
        "generated_utc": generated_utc,
        "method": {
            "name": "exact normalized word n-gram overlap",
            "normalization": "NFKC, strip combining marks, casefold, Unicode word tokens excluding underscore",
            "k": ks,
            "thresholds": thresholds,
            "overlap_fraction": "unique matched query shingles divided by unique query shingles for the same benchmark item/surface/k",
            "gpu_required": False,
        },
        "inputs": {
            "queries_jsonl": str(args.queries_jsonl),
            "corpus_jsonl": str(args.corpus_jsonl),
        },
        "query_summary": query_summary,
        "scan_summary": {
            "rows": rows,
            "bytes_read": bytes_read,
            "wall_seconds": elapsed,
            "source_row_counts": dict(sorted(source_counts.items())),
            "hit_records_seen": hit_records_seen,
            "hit_records_written": hit_records_written,
            "max_hit_records": args.max_hit_records,
            "hits_truncated": hit_records_seen > hit_records_written,
        },
        "threshold_summary": threshold_summary,
        "benchmark_matching_docs": {key: len(value) for key, value in sorted(benchmark_docs.items())},
        "outputs": {
            "hits_jsonl": str(hits_jsonl),
            "top_item_hits_jsonl": str(top_hits_path),
            "audit_json": str(audit_json),
        },
        "clean_subset_note": "This scan emits contaminated benchmark item ids. Clean-subset score recomputation is a separate deterministic join against per-checkpoint prediction JSONL.",
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

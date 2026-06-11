#!/usr/bin/env python3
"""Build a compact parquet of GreekMMLU benchmark/source match pairs."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
DEFAULT_SUBJECTS = ("Driving Rules", "World Religions", "Law")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits-jsonl", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--selected-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--benchmark", default="greekmmlu")
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--k", type=int, default=13)
    parser.add_argument("--min-overlap", type=float, default=0.2)
    parser.add_argument(
        "--max-examples-per-subject",
        type=int,
        default=0,
        help="If >0, keep only this many representative examples per subject; default 0 keeps all matching hit rows.",
    )
    return parser.parse_args()


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


def load_queries(path: Path, benchmark: str, subjects: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark") != benchmark:
                continue
            if row.get("subject") not in subjects:
                continue
            out[str(row["example_id"])] = row
    return out


def load_hits(
    path: Path,
    benchmark: str,
    subjects: set[str],
    k: int,
    min_overlap: float,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark") != benchmark:
                continue
            if row.get("subject") not in subjects:
                continue
            if int(row.get("k", -1)) != k:
                continue
            if float(row.get("overlap_fraction") or 0.0) < min_overlap:
                continue
            key = (
                row.get("subject"),
                row.get("example_id"),
                row.get("surface"),
                row.get("doc_id"),
                row.get("doc_row_index"),
            )
            if key in seen:
                continue
            seen.add(key)
            hits.append(row)
    hits.sort(key=lambda r: (str(r.get("subject")), str(r.get("example_id")), int(r.get("doc_row_index", -1)), str(r.get("surface"))))
    return hits


def select_representative_hits(hits: list[dict[str, Any]], max_per_subject: int) -> list[dict[str, Any]]:
    """Keep a tiny, reviewable sample: high-overlap rows, distinct examples first."""
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        by_subject.setdefault(str(hit.get("subject")), []).append(hit)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    for subject in sorted(by_subject):
        ranked = sorted(
            by_subject[subject],
            key=lambda row: (
                -float(row.get("overlap_fraction") or 0.0),
                -int(row.get("matched_query_shingles") or 0),
                str(row.get("example_id")),
                str(row.get("doc_id")),
                str(row.get("surface")),
            ),
        )
        used_examples: set[str] = set()
        subject_rows: list[dict[str, Any]] = []
        for row in ranked:
            example_id = str(row.get("example_id"))
            key = (subject, example_id, row.get("doc_id"), row.get("surface"))
            if example_id in used_examples or key in selected_keys:
                continue
            subject_rows.append(row)
            used_examples.add(example_id)
            selected_keys.add(key)
            if len(subject_rows) >= max_per_subject:
                break
        if len(subject_rows) < max_per_subject:
            for row in ranked:
                key = (subject, row.get("example_id"), row.get("doc_id"), row.get("surface"))
                if key in selected_keys:
                    continue
                subject_rows.append(row)
                selected_keys.add(key)
                if len(subject_rows) >= max_per_subject:
                    break
        selected.extend(subject_rows)
    selected.sort(key=lambda r: (str(r.get("subject")), str(r.get("example_id")), int(r.get("doc_row_index", -1)), str(r.get("surface"))))
    return selected


def load_corpus_rows(path: Path, row_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not row_indices:
        return {}
    out: dict[int, dict[str, Any]] = {}
    max_row = max(row_indices)
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx in row_indices:
                out[idx] = json.loads(line)
                if len(out) == len(row_indices):
                    break
            if idx > max_row:
                break
    missing = sorted(row_indices - set(out))
    if missing:
        raise RuntimeError(f"missing corpus rows: {missing[:10]} total={len(missing)}")
    return out


def load_metadata(selected_parquet: Path, doc_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not doc_ids:
        return {}
    dataset = ds.dataset(str(selected_parquet), format="parquet")
    table = dataset.to_table(
        columns=["doc_key", "source_doc_id", "title", "source_metadata_json"],
        filter=pc.field("doc_key").isin(list(doc_ids)),
    )
    out: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        raw_metadata = row.get("source_metadata_json")
        metadata: dict[str, Any] = {}
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                metadata = {"_raw_source_metadata_json": str(raw_metadata)[:1000]}
        out[str(row["doc_key"])] = {
            "source_doc_id": row.get("source_doc_id"),
            "title": row.get("title"),
            "url": metadata.get("url"),
            "host": metadata.get("host"),
            "crawl_id": metadata.get("crawl_id"),
            "timestamp": metadata.get("timestamp"),
            "content_type": metadata.get("content_type"),
            "register_level_1": metadata.get("register_level_1"),
            "register_level_2": metadata.get("register_level_2"),
        }
    return out


def first_doc_positions(doc_tokens: list[str], query_grams: set[tuple[str, ...]], k: int) -> dict[tuple[str, ...], int]:
    positions: dict[tuple[str, ...], int] = {}
    if not query_grams or len(doc_tokens) < k:
        return positions
    for idx in range(0, len(doc_tokens) - k + 1):
        gram = tuple(doc_tokens[idx : idx + k])
        if gram in query_grams and gram not in positions:
            positions[gram] = idx
    return positions


def find_phrase(tokens: list[str], phrase_tokens: list[str]) -> int | None:
    if not phrase_tokens or len(tokens) < len(phrase_tokens):
        return None
    n = len(phrase_tokens)
    for idx in range(0, len(tokens) - n + 1):
        if tokens[idx : idx + n] == phrase_tokens:
            return idx
    return None


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    subjects = {item.strip() for item in args.subjects.split(",") if item.strip()}
    queries = load_queries(args.queries_jsonl, args.benchmark, subjects)
    hits = load_hits(args.hits_jsonl, args.benchmark, subjects, args.k, args.min_overlap)
    if args.max_examples_per_subject > 0:
        hits = select_representative_hits(hits, args.max_examples_per_subject)
    corpus_rows = load_corpus_rows(args.corpus_jsonl, {int(hit["doc_row_index"]) for hit in hits})
    metadata = load_metadata(args.selected_parquet, {str(hit["doc_id"]) for hit in hits})

    rows: list[dict[str, Any]] = []
    for hit in hits:
        example_id = str(hit["example_id"])
        query = queries.get(example_id)
        if query is None:
            continue
        surface = str(hit["surface"])
        surface_text = str((query.get("surfaces") or {}).get(surface) or "")
        if not surface_text:
            continue
        choices = query.get("choices") or []
        answer_text = str(query.get("answer_text") or "")

        row_index = int(hit["doc_row_index"])
        corpus_row = corpus_rows[row_index]
        doc_text = str(corpus_row.get("text") or "")
        doc_tokens = tokenize(doc_text)
        doc_id = str(hit["doc_id"])
        meta = metadata.get(doc_id, {})
        mmlu_match = str((hit.get("first_match") or {}).get("normalized_text") or "")
        phrase_tokens = tokenize(mmlu_match)
        start = find_phrase(doc_tokens, phrase_tokens)
        if start is None:
            start = int((hit.get("first_match") or {}).get("token_start") or 0)
        source_match = " ".join(doc_tokens[start : start + len(phrase_tokens)]) if phrase_tokens else ""
        subject = str(hit.get("subject"))
        rows.append(
            {
                "subject": subject,
                "example_id": example_id,
                "url": meta.get("url"),
                "question": query.get("question"),
                "choices_json": json.dumps(choices, ensure_ascii=False),
                "answer_text": answer_text,
                "matched_surface": surface,
                "mmlu_used_to_match": mmlu_match,
                "training_matched_to": source_match,
                "training_document_text": doc_text,
            }
        )
    rows.sort(key=lambda row: (row["subject"], row["example_id"], row["matched_surface"], row["url"] or ""))
    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output_parquet, compression="zstd")
    summary = {
        "output_parquet": str(args.output_parquet),
        "rows": len(rows),
        "subjects": sorted({row["subject"] for row in rows}),
        "unique_examples": len({row["example_id"] for row in rows}),
        "unique_urls": len({row["url"] for row in rows if row.get("url")}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

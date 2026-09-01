#!/usr/bin/env python3
"""Audit frozen benchmark queries against a sharded Parquet corpus.

This is an evidence-only scan: it never rewrites or removes corpus rows.  It
emits one row per benchmark-unit/document match, including stable document and
1-based source-line locators.  A match is recommended for post-hoc exclusion
only when a question/source anchor and its correct-answer/paired-source anchor
both occur in the configured proximity window.

The implementation is resumable by Parquet shard.  Every shard receipt is
bound to the query file, dataset release manifest and expected shard metadata.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--minimum-short-question-tokens", type=int, default=3)
    parser.add_argument("--max-gap-tokens", type=int, default=50)
    parser.add_argument("--max-gap-tokens-short", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(192, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(value.casefold().split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def tokenize_document(text: str) -> tuple[list[str], list[int], list[str]]:
    """Return normalized tokens, their 1-based source lines, and raw lines."""
    raw_lines = text.splitlines() or [text]
    tokens: list[str] = []
    line_numbers: list[int] = []
    for line_number, line in enumerate(raw_lines, 1):
        line_tokens = tokenize(line)
        tokens.extend(line_tokens)
        line_numbers.extend([line_number] * len(line_tokens))
    return tokens, line_numbers, raw_lines


def kgrams(tokens: list[str], k: int) -> Iterable[tuple[str, ...]]:
    for index in range(max(0, len(tokens) - k + 1)):
        yield tuple(tokens[index : index + k])


def digit_fraction(tokens: tuple[str, ...]) -> float:
    return sum(token.isdigit() for token in tokens) / max(len(tokens), 1)


def load_queries(
    path: Path,
    *,
    k: int,
    minimum_short_question_tokens: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[tuple[str, ...], list[int]]], dict[str, int]]:
    queries: list[dict[str, Any]] = []
    issues: collections.Counter[str] = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "greek-benchmark-decontam-query-v2":
                raise ValueError(f"{path}:{line_number}: unsupported query schema")
            question_tokens = tokenize(str(row.get("question") or ""))
            if len(question_tokens) >= k:
                patterns = {
                    gram for gram in kgrams(question_tokens, k) if digit_fraction(gram) <= 0.5
                }
                pattern_kind = "kgram"
            elif len(question_tokens) >= minimum_short_question_tokens:
                patterns = {tuple(question_tokens)}
                pattern_kind = "short_exact"
                issues["short_question_exact_fallback"] += 1
            else:
                patterns = set()
                pattern_kind = "unmeasurable"
                issues["question_below_minimum_tokens"] += 1
            choices = [tokenize(str(value)) for value in row.get("choices") or []]
            answer_index = int(row["answer_index"])
            if not 0 <= answer_index < len(choices):
                raise ValueError(f"invalid answer index for {row.get('benchmark')}:{row.get('example_id')}")
            queries.append(
                {
                    **row,
                    "question_tokens": question_tokens,
                    "question_patterns": patterns,
                    "question_pattern_kind": pattern_kind,
                    "choice_tokens": choices,
                    "choice_kgrams": [set(kgrams(value, k)) for value in choices],
                    "direction": "any" if str(row.get("query_kind", "")).startswith(("nli_", "lexical_")) else "after",
                }
            )

    pattern_index: dict[int, dict[tuple[str, ...], list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for query_index, query in enumerate(queries):
        for pattern in query["question_patterns"]:
            pattern_index[len(pattern)][pattern].append(query_index)
    return queries, {length: dict(index) for length, index in pattern_index.items()}, dict(issues)


def find_subsequence(tokens: list[str], needle: list[str], low: int, high: int) -> int:
    if not needle:
        return -1
    high = min(high, len(tokens) - len(needle) + 1)
    for position in range(max(0, low), max(0, high)):
        if tokens[position : position + len(needle)] == needle:
            return position
    return -1


def option_evidence(
    doc_tokens: list[str],
    query: dict[str, Any],
    q_hits: list[tuple[int, int, tuple[str, ...]]],
    *,
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
) -> tuple[dict[int, tuple[int, int]], tuple[int, int, tuple[str, ...]]]:
    """Return option positions and the question hit used for the evidence."""
    best_q = q_hits[0]
    fired: dict[int, tuple[int, int]] = {}
    for q_hit in q_hits:
        q_position, q_length, _ = q_hit
        q_end = q_position + q_length
        for option_index, option_tokens in enumerate(query["choice_tokens"]):
            if not option_tokens or option_index in fired:
                continue
            gap = max_gap_tokens_short if len(option_tokens) < k else max_gap_tokens
            if query["direction"] == "after":
                low, high = q_end, q_end + gap + 1
            else:
                low = q_position - gap - len(option_tokens)
                high = q_end + gap + 1
            if len(option_tokens) < k:
                position = find_subsequence(doc_tokens, option_tokens, low, high)
                matched_length = len(option_tokens)
            else:
                position = -1
                matched_length = k
                option_patterns = query["choice_kgrams"][option_index]
                scan_low = max(0, low)
                scan_high = min(high, len(doc_tokens) - k + 1)
                for candidate in range(scan_low, max(scan_low, scan_high)):
                    if tuple(doc_tokens[candidate : candidate + k]) in option_patterns:
                        position = candidate
                        break
            if position >= 0:
                fired[option_index] = (position, matched_length)
                if option_index == query["answer_index"]:
                    best_q = q_hit
        if query["answer_index"] in fired:
            break
    return fired, best_q


def source_url(metadata_json: Any) -> str | None:
    if not metadata_json:
        return None
    try:
        metadata = json.loads(str(metadata_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    for key in ("url", "source_url", "download_url", "document_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def line_snippet(raw_lines: list[str], start: int, end: int, limit: int = 500) -> str:
    value = "\n".join(raw_lines[max(0, start - 1) : min(len(raw_lines), end)])
    value = " ".join(value.split())
    return value[:limit]


def scan_document(
    row: dict[str, Any],
    *,
    shard_path: str,
    shard_row_index: int,
    queries: list[dict[str, Any]],
    pattern_index: dict[int, dict[tuple[str, ...], list[int]]],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
) -> list[dict[str, Any]]:
    text = str(row.get("text") or "")
    doc_tokens, token_lines, raw_lines = tokenize_document(text)
    if not doc_tokens:
        return []
    hits: dict[int, list[tuple[int, int, tuple[str, ...]]]] = collections.defaultdict(list)
    for length, index in pattern_index.items():
        if len(doc_tokens) < length:
            continue
        for position in range(len(doc_tokens) - length + 1):
            pattern = tuple(doc_tokens[position : position + length])
            for query_index in index.get(pattern, ()):
                hits[query_index].append((position, length, pattern))
    if not hits:
        return []

    output: list[dict[str, Any]] = []
    for query_index, q_hits in hits.items():
        query = queries[query_index]
        fired, selected_q = option_evidence(
            doc_tokens,
            query,
            q_hits,
            k=k,
            max_gap_tokens=max_gap_tokens,
            max_gap_tokens_short=max_gap_tokens_short,
        )
        correct = query["answer_index"] in fired
        wrong = bool(set(fired) - {query["answer_index"]})
        if correct and wrong:
            category = "q_plus_correct_and_wrong"
        elif correct:
            category = "q_plus_correct_only"
        elif wrong:
            category = "q_plus_wrong_only"
        else:
            category = "q_only"
        q_position, q_length, q_pattern = selected_q
        q_line_start = token_lines[q_position]
        q_line_end = token_lines[min(q_position + q_length - 1, len(token_lines) - 1)]
        correct_position = fired.get(query["answer_index"])
        if correct_position:
            a_position, a_length = correct_position
            a_line_start = token_lines[a_position]
            a_line_end = token_lines[min(a_position + a_length - 1, len(token_lines) - 1)]
            evidence_line_start = min(q_line_start, a_line_start)
            evidence_line_end = max(q_line_end, a_line_end)
            answer_pattern = tuple(doc_tokens[a_position : a_position + a_length])
        else:
            a_line_start = a_line_end = None
            evidence_line_start, evidence_line_end = q_line_start, q_line_end
            answer_pattern = ()
        source_dataset = str(row.get("source_dataset") or "")
        source_doc_id = str(row.get("source_doc_id") or f"{Path(shard_path).name}:{shard_row_index}")
        output.append(
            {
                "benchmark": query["benchmark"],
                "evaluation_unit_id": query["evaluation_unit_id"],
                "example_id": query["example_id"],
                "discount_example_ids": query["discount_example_ids"],
                "source_group_id": query.get("source_group_id"),
                "subject": query.get("subject"),
                "query_kind": query["query_kind"],
                "match_category": category,
                "match_strength": "strong" if correct else ("contextual" if wrong else "candidate"),
                "recommended_exclusion": bool(correct),
                "source_dataset": source_dataset,
                "source_doc_id": source_doc_id,
                "document_key_sha256": sha256_text(f"{source_dataset}\0{source_doc_id}"),
                "document_text_sha256": sha256_text(text),
                "document_url": source_url(row.get("source_metadata_json")),
                "dataset_shard": shard_path,
                "dataset_row_index": shard_row_index,
                "question_pattern_kind": query["question_pattern_kind"],
                "question_match_tokens": q_length,
                "question_match_sha256": sha256_text(" ".join(q_pattern)),
                "question_line_start": q_line_start,
                "question_line_end": q_line_end,
                "correct_answer_match_tokens": len(answer_pattern),
                "correct_answer_match_sha256": sha256_text(" ".join(answer_pattern)) if answer_pattern else None,
                "correct_answer_line_start": a_line_start,
                "correct_answer_line_end": a_line_end,
                "matched_choice_indices": sorted(fired),
                "evidence_line_start": evidence_line_start,
                "evidence_line_end": evidence_line_end,
                "evidence_snippet": line_snippet(raw_lines, evidence_line_start, evidence_line_end),
            }
        )
    return output


def init_worker(
    queries: list[dict[str, Any]],
    pattern_index: dict[int, dict[tuple[str, ...], list[int]]],
    k: int,
    max_gap_tokens: int,
    max_gap_tokens_short: int,
    batch_size: int,
) -> None:
    _STATE.update(
        queries=queries,
        pattern_index=pattern_index,
        k=k,
        max_gap_tokens=max_gap_tokens,
        max_gap_tokens_short=max_gap_tokens_short,
        batch_size=batch_size,
    )


def scan_parquet_task(task: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = Path(task["absolute_path"])
    parquet = pq.ParquetFile(path)
    expected_rows = int(task["rows"])
    if parquet.metadata.num_rows != expected_rows or path.stat().st_size != int(task["bytes"]):
        raise ValueError(f"corpus shard metadata drift: {path}")
    matches: list[dict[str, Any]] = []
    row_index = 0
    columns = ["source_dataset", "source_doc_id", "text", "source_metadata_json"]
    for batch in parquet.iter_batches(batch_size=_STATE["batch_size"], columns=columns):
        for row in batch.to_pylist():
            matches.extend(
                scan_document(
                    row,
                    shard_path=task["path"],
                    shard_row_index=row_index,
                    queries=_STATE["queries"],
                    pattern_index=_STATE["pattern_index"],
                    k=_STATE["k"],
                    max_gap_tokens=_STATE["max_gap_tokens"],
                    max_gap_tokens_short=_STATE["max_gap_tokens_short"],
                )
            )
            row_index += 1
    if row_index != expected_rows:
        raise ValueError(f"row-count drift while scanning {path}: {row_index} != {expected_rows}")
    return {"task": task, "matches": matches, "rows_scanned": row_index}


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def valid_existing_receipt(path: Path, *, query_sha256: str, manifest_sha256: str, task: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("status") == "passed"
        and receipt.get("queries_sha256") == query_sha256
        and receipt.get("corpus_manifest_sha256") == manifest_sha256
        and receipt.get("input", {}).get("path") == task["path"]
        and receipt.get("input", {}).get("sha256") == task["sha256"]
    )


def main() -> int:
    args = parse_args()
    if not 0 <= args.lane < args.lanes:
        raise ValueError("lane must be in [0, lanes)")
    if args.k < 1 or args.minimum_short_question_tokens < 1:
        raise ValueError("token thresholds must be positive")
    manifest_sha256 = sha256_file(args.corpus_manifest)
    queries_sha256 = sha256_file(args.queries_jsonl)
    publication_sha256 = sha256_file(args.publication_receipt)
    manifest = json.loads(args.corpus_manifest.read_text())
    publication = json.loads(args.publication_receipt.read_text())
    if manifest.get("status") != "passed" or publication.get("status") != "passed":
        raise ValueError("dataset manifest/publication receipt is not passed")
    if publication.get("manifest", {}).get("sha256") != manifest_sha256:
        raise ValueError("publication receipt is not bound to the corpus manifest")
    dataset_revision = str(publication["commit_sha"])
    queries, pattern_index, query_issues = load_queries(
        args.queries_jsonl,
        k=args.k,
        minimum_short_question_tokens=args.minimum_short_question_tokens,
    )
    tasks = []
    for task_index, item in enumerate(manifest["files"]):
        if task_index % args.lanes != args.lane:
            continue
        task = dict(item)
        task["absolute_path"] = str(args.corpus_root / item["path"])
        stem = Path(item["path"]).stem
        receipt_path = args.output_root / "shards" / f"{stem}.receipt.json"
        if valid_existing_receipt(
            receipt_path,
            query_sha256=queries_sha256,
            manifest_sha256=manifest_sha256,
            task=task,
        ):
            continue
        tasks.append(task)

    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed = 0
    context = mp.get_context("fork")
    with context.Pool(
        processes=max(1, args.workers),
        initializer=init_worker,
        initargs=(
            queries,
            pattern_index,
            args.k,
            args.max_gap_tokens,
            args.max_gap_tokens_short,
            args.batch_size,
        ),
    ) as pool:
        for result in pool.imap_unordered(scan_parquet_task, tasks, chunksize=1):
            task = result["task"]
            stem = Path(task["path"]).stem
            match_path = args.output_root / "shards" / f"{stem}.matches.jsonl"
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in sorted(
                    result["matches"],
                    key=lambda row: (
                        row["benchmark"],
                        row["evaluation_unit_id"],
                        row["source_dataset"],
                        row["source_doc_id"],
                    ),
                )
            )
            atomic_write(match_path, payload)
            receipt = {
                "schema_version": "greek_benchmark_contamination_shard_v1",
                "status": "passed",
                "dataset_revision": dataset_revision,
                "queries_sha256": queries_sha256,
                "corpus_manifest_sha256": manifest_sha256,
                "publication_receipt_sha256": publication_sha256,
                "input": {
                    "path": task["path"],
                    "sha256": task["sha256"],
                    "bytes": task["bytes"],
                    "rows": task["rows"],
                },
                "output": {
                    "path": str(match_path),
                    "sha256": sha256_file(match_path),
                    "matches": len(result["matches"]),
                },
                "rows_scanned": result["rows_scanned"],
            }
            atomic_write(
                args.output_root / "shards" / f"{stem}.receipt.json",
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "completed_this_run": completed,
                        "pending_at_start": len(tasks),
                        "shard": task["path"],
                        "matches": len(result["matches"]),
                        "elapsed_seconds": round(time.monotonic() - start, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.stop_file and args.stop_file.exists():
                pool.terminate()
                break

    run_receipt = {
        "schema_version": "greek_benchmark_contamination_scan_lane_v1",
        "status": "stopped" if args.stop_file and args.stop_file.exists() else "passed",
        "lane": args.lane,
        "lanes": args.lanes,
        "dataset_revision": dataset_revision,
        "queries_sha256": queries_sha256,
        "corpus_manifest_sha256": manifest_sha256,
        "publication_receipt_sha256": publication_sha256,
        "query_issues": query_issues,
        "queries": len(queries),
        "pattern_lengths": {str(length): len(index) for length, index in pattern_index.items()},
        "shards_completed_this_run": completed,
        "shards_pending_at_start": len(tasks),
        "wall_seconds": round(time.monotonic() - start, 2),
    }
    atomic_write(
        args.output_root / f"lane_{args.lane}.receipt.json",
        json.dumps(run_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(run_receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if run_receipt["status"] == "passed" else 75


if __name__ == "__main__":
    raise SystemExit(main())

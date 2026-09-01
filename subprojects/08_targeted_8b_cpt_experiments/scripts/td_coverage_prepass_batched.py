#!/usr/bin/env python3
"""Exact-order batched encoder for the pinned Token-Distillation prepass.

The scientific loop, reservoir updates and output rows are inherited from the
pinned historical prepass.  Only independent document encoding is batched so
the Rust tokenizer can use the CPUs already allocated on a Clariden debug node.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import unicodedata as ud
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer


_ENCODE_WORKER_TOKENIZER: Tokenizer | None = None


def initialize_encode_worker(tokenizer_json: str) -> None:
    global _ENCODE_WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _ENCODE_WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_json)


def encode_batch_worker(
    batch: list[tuple[Path, int, dict[str, object], str]],
) -> tuple[list[tuple[Path, int, dict[str, object], str]], list[tuple[list[int], list[tuple[int, int]]]]]:
    if _ENCODE_WORKER_TOKENIZER is None:
        raise RuntimeError("encode worker tokenizer is not initialized")
    encodings = _ENCODE_WORKER_TOKENIZER.encode_batch(
        [item[3] for item in batch], add_special_tokens=False
    )
    return batch, [(encoding.ids, encoding.offsets) for encoding in encodings]


def encode_batch_local(
    tokenizer: Tokenizer,
    batch: list[tuple[Path, int, dict[str, object], str]],
) -> tuple[list[tuple[Path, int, dict[str, object], str]], list[tuple[list[int], list[tuple[int, int]]]]]:
    encodings = tokenizer.encode_batch([item[3] for item in batch], add_special_tokens=False)
    return batch, [(encoding.ids, encoding.offsets) for encoding in encodings]


def ordered_parallel_batches(
    executor: ProcessPoolExecutor,
    batches: Iterable[list[tuple[Path, int, dict[str, object], str]]],
    max_in_flight: int,
) -> Iterator[tuple[list[tuple[Path, int, dict[str, object], str]], list[tuple[list[int], list[tuple[int, int]]]]]]:
    pending: deque[Future] = deque()
    for batch in batches:
        pending.append(executor.submit(encode_batch_worker, batch))
        if len(pending) >= max_in_flight:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def load_reference(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pinned_td_coverage_prepass", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load pinned TD prepass: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def text_batches(
    reference: ModuleType,
    paths: Iterable[Path],
    text_key: str,
    max_documents: int,
    max_characters: int,
) -> Iterator[list[tuple[Path, int, dict[str, object], str]]]:
    batch: list[tuple[Path, int, dict[str, object], str]] = []
    characters = 0
    for path, line_no, row in reference.iter_jsonl(paths):
        text = row.get(text_key)
        if not isinstance(text, str) or not text:
            continue
        if batch and (len(batch) >= max_documents or characters + len(text) > max_characters):
            yield batch
            batch = []
            characters = 0
        batch.append((path, line_no, row, text))
        characters += len(text)
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-id-start", type=int, default=131_072)
    parser.add_argument("--new-id-end", type=int, default=148_480)
    parser.add_argument("--target-extended-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--source-key", default="source")
    parser.add_argument("--doc-id-key", default="doc_id")
    parser.add_argument("--lang-key", default="lang")
    parser.add_argument("--snippet-token-radius", type=int, default=50)
    parser.add_argument("--snippets-per-token", type=int, default=100)
    parser.add_argument("--example-refs-per-token", type=int, default=5)
    parser.add_argument("--progress-token-interval", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--require-nfc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encode-batch-documents", type=int, default=128)
    parser.add_argument("--encode-batch-characters", type=int, default=4_000_000)
    parser.add_argument("--encode-workers", type=int, default=1)
    parser.add_argument("--max-encode-batches-in-flight", type=int, default=2)
    parser.add_argument("--parity-documents", type=int, default=256)
    return parser.parse_args()


def reservoir_slot(
    rng: random.Random,
    coverage: object,
    max_snippets: int,
) -> int | None:
    """Advance the pinned reservoir exactly, without eagerly building a discarded snippet."""
    coverage.snippets_seen += 1
    if len(coverage.snippets) < max_snippets:
        return len(coverage.snippets)
    index = rng.randrange(coverage.snippets_seen)
    return index if index < max_snippets else None


def main() -> int:
    args = parse_args()
    if args.new_id_start >= args.new_id_end:
        raise SystemExit("--new-id-start must be smaller than --new-id-end")
    if args.target_extended_tokens <= 0:
        raise SystemExit("--target-extended-tokens must be positive")
    if (
        args.encode_batch_documents <= 0
        or args.encode_batch_characters <= 0
        or args.encode_workers <= 0
        or args.max_encode_batches_in_flight < args.encode_workers
    ):
        raise SystemExit("batch bounds must be positive")
    if args.parity_documents < 0:
        raise SystemExit("--parity-documents must be non-negative")

    reference = load_reference(args.reference_script)
    base_tokenizer = Tokenizer.from_file(str(reference.tokenizer_json_path(args.base_tokenizer)))
    student_tokenizer = Tokenizer.from_file(str(reference.tokenizer_json_path(args.student_tokenizer)))
    vocab_size = student_tokenizer.get_vocab_size(with_added_tokens=True)
    if args.new_id_end > vocab_size:
        raise SystemExit(f"new-id-end {args.new_id_end} exceeds student vocab {vocab_size}")

    out_dir = args.output_dir
    snippet_dir = out_dir / "td_snippet_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    snippet_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ids_of_interest = set(range(args.new_id_start, args.new_id_end))
    coverage = {token_id: reference.TokenCoverage() for token_id in ids_of_interest}
    static_info = {
        token_id: reference.build_static_token_info(base_tokenizer, student_tokenizer, token_id)
        for token_id in range(args.new_id_start, args.new_id_end)
    }

    docs_seen = 0
    docs_used = 0
    tokens_seen = 0
    chars_seen = 0
    non_nfc_docs = 0
    stopped_on_budget = False
    parity_checked = 0
    next_progress_tokens = args.progress_token_interval if args.progress_token_interval > 0 else None

    batches = text_batches(
        reference,
        args.input_jsonl,
        args.text_key,
        args.encode_batch_documents,
        args.encode_batch_characters,
    )
    if args.encode_workers == 1:
        encoded_batches = (encode_batch_local(student_tokenizer, batch) for batch in batches)
        executor = None
    else:
        worker_tokenizer = str(reference.tokenizer_json_path(args.student_tokenizer))
        executor = ProcessPoolExecutor(
            max_workers=args.encode_workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_encode_worker,
            initargs=(worker_tokenizer,),
        )
        encoded_batches = ordered_parallel_batches(
            executor,
            batches,
            args.max_encode_batches_in_flight,
        )
    try:
        for batch, encoded in encoded_batches:
            if len(encoded) != len(batch):
                raise SystemExit("batched tokenizer changed document cardinality")
            for (path, line_no, row, text), (encoded_ids, encoded_offsets) in zip(batch, encoded):
                ids = encoded_ids
                offsets = encoded_offsets
                if parity_checked < args.parity_documents:
                    sequential = student_tokenizer.encode(text, add_special_tokens=False)
                    if ids != sequential.ids or offsets != sequential.offsets:
                        raise SystemExit(f"batched/sequential tokenizer parity failed at {path}:{line_no}")
                    parity_checked += 1

                docs_seen += 1
                chars_seen += len(text)
                if text != ud.normalize("NFC", text):
                    non_nfc_docs += 1
                if not ids:
                    continue
                remaining = args.target_extended_tokens - tokens_seen
                if remaining <= 0:
                    stopped_on_budget = True
                    break
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    offsets = offsets[:remaining]
                    stopped_on_budget = True

                docs_used += 1
                tokens_seen += len(ids)
                source = row.get(args.source_key)
                lang = row.get(args.lang_key)
                doc_id = row.get(args.doc_id_key) or f"{path.name}:{line_no}"
                seen_in_doc: set[int] = set()
                token_array = np.asarray(ids, dtype=np.int32)
                new_positions = np.flatnonzero(
                    (token_array >= args.new_id_start) & (token_array < args.new_id_end)
                )
                for position in new_positions:
                    token_index = int(position)
                    token_id = int(token_array[token_index])
                    cov = coverage[token_id]
                    cov.extended_firings += 1
                    seen_in_doc.add(token_id)
                    start, end = offsets[token_index]
                    if end <= start:
                        continue
                    slot = reservoir_slot(rng, cov, args.snippets_per_token)
                    if slot is None:
                        continue
                    left_i = max(0, token_index - args.snippet_token_radius)
                    right_i = min(len(ids), token_index + args.snippet_token_radius + 1)
                    char_start = offsets[left_i][0]
                    char_end = offsets[right_i - 1][1]
                    surface = text[start:end]
                    snippet = {
                        "new_token_id": token_id,
                        "doc_ref": f"{path}:{line_no}",
                        "doc_id": doc_id,
                        "source": source,
                        "lang": lang,
                        "token_index": token_index,
                        "char_start": start,
                        "char_end": end,
                        "snippet_char_start": char_start,
                        "snippet_char_end": char_end,
                        "surface": surface,
                        "span_base_subtoken_ids": reference.encode_ids(base_tokenizer, surface),
                        "snippet_text": text[char_start:char_end],
                    }
                    if slot == len(cov.snippets):
                        cov.snippets.append(snippet)
                    else:
                        cov.snippets[slot] = snippet
                for token_id in seen_in_doc:
                    coverage[token_id].docs_with_firing += 1

                if stopped_on_budget:
                    break
                if next_progress_tokens is not None and tokens_seen >= next_progress_tokens:
                    print(json.dumps({
                        "event": "td_coverage_progress",
                        "tokens_scanned": tokens_seen,
                        "target_extended_tokens": args.target_extended_tokens,
                        "docs_seen": docs_seen,
                        "docs_used": docs_used,
                        "chars_seen": chars_seen,
                    }, ensure_ascii=False), flush=True)
                    while next_progress_tokens is not None and tokens_seen >= next_progress_tokens:
                        next_progress_tokens += args.progress_token_interval
            if stopped_on_budget:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if parity_checked < min(args.parity_documents, docs_seen):
        raise SystemExit("requested batched/sequential parity sample was not completed")
    if args.require_nfc and non_nfc_docs:
        raise SystemExit(
            f"Refusing to emit TD coverage: {non_nfc_docs} docs were not NFC. "
            "Run on the post-normalize corpus or pass --no-require-nfc for a diagnostic-only scan."
        )

    snippet_jsonl = snippet_dir / "snippets.jsonl"
    prepass_jsonl = out_dir / "td_coverage_prepass.jsonl"
    summary_path = out_dir / "td_coverage_summary.json"
    snippet_refs_by_token: dict[int, list[str]] = {token_id: [] for token_id in ids_of_interest}
    with snippet_jsonl.open("w", encoding="utf-8") as handle:
        for token_id in range(args.new_id_start, args.new_id_end):
            for index, snippet in enumerate(coverage[token_id].snippets):
                snippet_id = f"{token_id}:{index:04d}"
                snippet_refs_by_token[token_id].append(snippet_id)
                handle.write(json.dumps({"snippet_id": snippet_id, **snippet}, ensure_ascii=False) + "\n")

    status_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    with prepass_jsonl.open("w", encoding="utf-8") as handle:
        for token_id in range(args.new_id_start, args.new_id_end):
            cov = coverage[token_id]
            usable = len(cov.snippets)
            status, action = reference.status_for(usable)
            status_counts[status] = status_counts.get(status, 0) + 1
            action_counts[action] = action_counts.get(action, 0) + 1
            row = {
                **static_info[token_id],
                "extended_firings": cov.extended_firings,
                "raw_surface_occurrences": None,
                "usable_snippets_25": min(usable, 25),
                "usable_snippets_100": min(usable, 100),
                "docs_with_firing": cov.docs_with_firing,
                "example_snippet_refs": snippet_refs_by_token[token_id][: args.example_refs_per_token],
                "status": status,
                "recommended_action": action,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_count = args.new_id_end - args.new_id_start
    enough_100 = status_counts.get("enough_100", 0)
    enough_25 = enough_100 + status_counts.get("enough_25", 0)
    if enough_100 / token_count >= 0.90:
        recommended = "run_full_td_100"
    elif enough_25 / token_count >= 0.90:
        recommended = "run_td_25_with_flagged_tail"
    else:
        recommended = "do_not_launch_full_td_inspect_coverage"
    summary = {
        "inputs": [str(path) for path in args.input_jsonl],
        "base_tokenizer": str(args.base_tokenizer),
        "student_tokenizer": str(args.student_tokenizer),
        "new_id_start": args.new_id_start,
        "new_id_end": args.new_id_end,
        "target_extended_tokens": args.target_extended_tokens,
        "tokens_scanned": tokens_seen,
        "docs_seen": docs_seen,
        "docs_used": docs_used,
        "chars_seen": chars_seen,
        "stopped_on_budget": stopped_on_budget,
        "require_nfc": args.require_nfc,
        "non_nfc_docs": non_nfc_docs,
        "snippet_token_radius": args.snippet_token_radius,
        "snippets_per_token": args.snippets_per_token,
        "status_counts": status_counts,
        "action_counts": action_counts,
        "enough_100_fraction": enough_100 / token_count,
        "enough_25_fraction": enough_25 / token_count,
        "low_lt25_count": token_count - enough_25,
        "recommended_next_step": recommended,
        "artifacts": {
            "coverage_jsonl": str(prepass_jsonl),
            "summary_json": str(summary_path),
            "snippet_index_jsonl": str(snippet_jsonl),
        },
        "counting_rule": "A firing means the extended tokenizer emitted the new token ID in the scanned training stream.",
        "encoding_execution": {
            "mode": "ordered_multiprocess_encode_batches_with_sequential_parity_guard",
            "workers": args.encode_workers,
            "max_batches_in_flight": args.max_encode_batches_in_flight,
            "batch_documents": args.encode_batch_documents,
            "batch_characters": args.encode_batch_characters,
            "parity_documents": parity_checked,
            "new_token_position_filter": "numpy_flatnonzero_contiguous_id_range_preserving_ascending_order",
            "snippet_materialization": "deferred_until_exact_pinned_reservoir_slot_is_selected",
            "scientific_state_update_order": "identical_to_pinned_sequential_reference",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

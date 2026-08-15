#!/usr/bin/env python3
"""Apply historical E001 and GreekMMLU filtering to one selected Greek mix.

The historical implementation first wrote an E001-cleaned JSONL, scanned it,
then reread it to filter.  This implementation composes the byte-identical
``clean_text`` and per-document matching functions in one ordered pass.  It
changes neither rule and performs no deduplication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


_STATE: dict[str, Any] = {}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_doc_id(source_dataset: str, source_doc_id: str, release_shard: str, release_row: int) -> str:
    payload = f"{source_dataset}\0{source_doc_id}\0{release_shard}\0{release_row}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def init_worker(e001_path: str, scanner_path: str, items: list[dict[str, Any]], index: dict, k: int) -> None:
    e001 = load_module("h2g_frozen_e001", Path(e001_path))
    scanner = load_module("h2g_frozen_greekmmlu_scanner", Path(scanner_path))
    _STATE.update(e001=e001, scanner=scanner, items=items, index=index, k=k)


def process_row(task: tuple[str, int, dict[str, Any]]) -> tuple[str, bool, list[tuple[int, str]], int, bool]:
    release_shard, release_row, row = task
    text = row.get("text")
    source_dataset = str(row.get("source_dataset") or "")
    source_doc_id = str(row.get("source_doc_id") or "")
    if not isinstance(text, str) or not text or not source_dataset or not source_doc_id:
        raise ValueError(f"invalid source row: {release_shard}:{release_row}")
    cleaned, removed = _STATE["e001"].clean_text(text)
    scanner = _STATE["scanner"]
    tokens = scanner.tokenize(cleaned)
    categories: list[tuple[int, str]] = []
    excluded = False
    if len(tokens) >= _STATE["k"]:
        matches = scanner.find_q_match_positions(tokens, _STATE["index"], _STATE["k"])
        for item_index, positions in matches.items():
            item = _STATE["items"][item_index]
            fired = scanner.option_hits_in_windows(tokens, item, positions, _STATE["k"], 50, 5, "after")
            category = scanner.categorize_pair(item["answer_index"], fired)
            categories.append((item_index, category))
            if category in {"q_plus_correct_only", "q_plus_correct_and_wrong"}:
                excluded = True
    doc_id = canonical_doc_id(source_dataset, source_doc_id, release_shard, release_row)
    normalized = {
        "text": cleaned,
        "source": source_dataset,
        "doc_id": doc_id,
        "source_dataset": source_dataset,
        "source_doc_id": source_doc_id,
        "source_metadata_json": row.get("source_metadata_json"),
        "release_shard": release_shard,
        "release_row_index": release_row,
        "document_text_sha256_before_e001": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "document_text_sha256_after_e001": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True), excluded, categories, removed, cleaned != text


def selected_tasks(selection: dict[str, Any], pool: str) -> Iterable[tuple[str, int, dict[str, Any]]]:
    binding = selection["output"]
    path = Path(binding["path"])
    require(path.is_file(), f"selected mix missing: {path}")
    require(path.stat().st_size == int(binding["bytes"]), f"selected mix byte drift: {path}")
    require(sha256_file(path) == binding["sha256"], f"selected mix SHA drift: {path}")
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            for name in ("text", "source_dataset", "source_doc_id", "_source_release_shard", "_source_release_row_index"):
                require(row.get(name) not in (None, ""), f"selected {pool} row {line_number} lacks {name}")
            release_shard = str(row.pop("_source_release_shard"))
            release_row = int(row.pop("_source_release_row_index"))
            yield release_shard, release_row, row
            rows += 1
    require(rows == int(selection["actual_rows"]), f"selected {pool} row-count drift")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hplt", "openarchives"), required=True)
    parser.add_argument("--selected-mix-receipt", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--query-receipt", type=Path, required=True)
    parser.add_argument("--e001-script", type=Path, required=True)
    parser.add_argument("--scanner-script", type=Path, required=True)
    parser.add_argument("--clean-jsonl", type=Path, required=True)
    parser.add_argument("--dropped-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=256)
    args = parser.parse_args()
    for output in (args.clean_jsonl, args.dropped_jsonl, args.output_receipt):
        require(not output.exists(), f"immutable prepared source output exists: {output}")
    require(1 <= args.workers <= 64 and args.chunksize >= 1, "invalid worker geometry")
    selection = read_json(args.selected_mix_receipt)
    require(selection.get("schema_version") in {
        "apertus_hard_h_to_g_modern_mix_v1",
        "apertus_hard_h_to_g_openarchives_candidate_source_v1",
    }, "selected-mix/candidate schema drift")
    require(selection.get("status") == "passed" and selection.get("pool") == args.pool, "selected-mix pool/status drift")
    expected_rows = int(selection["actual_rows"])
    query_receipt = read_json(args.query_receipt)
    require(query_receipt.get("status") == "passed", "GreekMMLU query receipt did not pass")
    require(query_receipt["queries"]["sha256"] == sha256_file(args.queries_jsonl), "GreekMMLU query binding drift")
    scanner = load_module("h2g_scanner_parent", args.scanner_script)
    queries = scanner.load_items(args.queries_jsonl, "greekmmlu")
    items, issues = scanner.build_item_grams(queries, 8, 0.5)
    index = scanner.build_global_q_index(items)
    require(len(queries) == int(query_receipt["queries"]["rows"]), "GreekMMLU query row drift")

    args.clean_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp_paths = []
    descriptors = []
    for output in (args.clean_jsonl, args.dropped_jsonl):
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        descriptors.append(descriptor); temp_paths.append(Path(name))
    counts: Counter[str] = Counter()
    category_pairs: Counter[str] = Counter()
    category_items: dict[str, set[int]] = defaultdict(set)
    context = mp.get_context("fork")
    try:
        with os.fdopen(descriptors[0], "w", encoding="utf-8") as clean, os.fdopen(descriptors[1], "w", encoding="utf-8") as dropped, context.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(str(args.e001_script), str(args.scanner_script), items, index, 8),
        ) as pool:
            results = pool.imap(process_row, selected_tasks(selection, args.pool), chunksize=args.chunksize)
            for line, excluded, categories, removed, changed in results:
                counts["input_rows"] += 1
                counts["e001_characters_removed"] += removed
                counts["e001_changed_documents"] += int(changed)
                for item_index, category in categories:
                    category_pairs[category] += 1
                    category_items[category].add(item_index)
                if excluded:
                    dropped.write(line + "\n")
                    counts["greekmmlu_dropped_rows"] += 1
                else:
                    clean.write(line + "\n")
                    counts["clean_rows"] += 1
                if counts["input_rows"] % 1_000_000 == 0:
                    print({"pool": args.pool, "rows": counts["input_rows"], "clean": counts["clean_rows"], "dropped": counts["greekmmlu_dropped_rows"]}, flush=True)
            clean.flush(); os.fsync(clean.fileno()); dropped.flush(); os.fsync(dropped.fileno())
        require(counts["input_rows"] == expected_rows, f"{args.pool} source row drift")
        require(counts["input_rows"] == counts["clean_rows"] + counts["greekmmlu_dropped_rows"], "prepared stream row accounting drift")
        for temporary, output in zip(temp_paths, (args.clean_jsonl, args.dropped_jsonl), strict=True):
            os.link(temporary, output); temporary.unlink()
    except BaseException:
        for descriptor in descriptors:
            try: os.close(descriptor)
            except OSError: pass
        for temporary in temp_paths: temporary.unlink(missing_ok=True)
        raise

    payload = {
        "schema_version": "apertus_hard_h_to_g_prepared_greek_stream_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pool": args.pool,
        "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodes": int(os.environ.get("SLURM_NNODES", "0"))},
        "executing_code_bundle": executing_code_bundle(),
        "selected_mix_receipt": file_binding(args.selected_mix_receipt),
        "queries": file_binding(args.queries_jsonl), "query_receipt": file_binding(args.query_receipt),
        "e001_implementation": file_binding(args.e001_script), "greekmmlu_scanner_implementation": file_binding(args.scanner_script),
        "command_contract": {"k": 8, "max_gap_tokens": 50, "max_gap_tokens_short": 5, "direction": "after", "primary_rule": "correct_only", "max_q_kgram_digit_fraction": 0.5},
        "query_input_issues": issues,
        "counts": dict(counts),
        "match_pairs_by_category": dict(sorted(category_pairs.items())),
        "matched_items_by_category": {name: len(indices) for name, indices in sorted(category_items.items())},
        "clean": {**file_binding(args.clean_jsonl), "rows": counts["clean_rows"]},
        "dropped": {**file_binding(args.dropped_jsonl), "rows": counts["greekmmlu_dropped_rows"]},
        "invariants": {"selected_mix_order_preserved": True, "nonmatching_row_multiplicity_preserved": True, "additional_deduplication": False, "e001_and_greekmmlu_composed_per_document": True},
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

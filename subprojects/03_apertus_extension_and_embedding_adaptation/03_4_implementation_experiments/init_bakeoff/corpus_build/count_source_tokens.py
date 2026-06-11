#!/usr/bin/env python3
"""Count tokenizer tokens by source_dataset in staged parquet files.

The run is resumable at file granularity. Each input parquet writes one JSON
partial under `<output_dir>/partials/`; re-runs skip completed partials.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOKENIZER: Any = None


def expand_inputs(patterns: list[str], exclude_prefixes: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(p) for p in matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)
    unique = sorted({p.resolve() for p in paths})
    if exclude_prefixes:
        unique = [
            path for path in unique
            if not any(path.name.startswith(prefix) for prefix in exclude_prefixes)
        ]
    if not unique:
        raise SystemExit(f"No parquet files matched after exclusions: {patterns}")
    return unique


def partial_name(path: Path) -> str:
    return f"{path.name.replace('/', '__')}.json"


def infer_source_from_filename(path: Path) -> str:
    name = path.name
    name = re.sub(r"\.part-\d+\.parquet$", "", name)
    name = re.sub(r"\.parquet$", "", name)
    return name.replace("__", "/")


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    from tokenizers import Tokenizer  # type: ignore

    path = Path(tokenizer_path)
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_json.exists():
        raise RuntimeError(f"tokenizer.json not found at {tokenizer_json}")
    _TOKENIZER = Tokenizer.from_file(str(tokenizer_json))


def empty_counter() -> dict[str, int]:
    return {
        "files": 0,
        "rows": 0,
        "nonempty_rows": 0,
        "chars": 0,
        "utf8_bytes": 0,
        "tokens_no_eod": 0,
        "tokens_with_eod": 0,
    }


def count_file(args: tuple[str, str, int, str, str]) -> dict[str, Any]:
    path_raw, partial_dir_raw, batch_size, text_column, source_column = args
    path = Path(path_raw)
    partial_dir = Path(partial_dir_raw)
    partial_path = partial_dir / partial_name(path)
    if partial_path.exists():
        data = json.loads(partial_path.read_text(encoding="utf-8"))
        data["from_cache"] = True
        return data

    import pyarrow.parquet as pq

    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer was not initialized")

    started = time.time()
    pf = pq.ParquetFile(path)
    columns = set(pf.schema_arrow.names)
    if text_column not in columns:
        raise RuntimeError(f"{path} has no {text_column!r} column; columns={pf.schema_arrow.names}")

    read_columns = [text_column]
    has_source_column = source_column in columns
    if has_source_column:
        read_columns.append(source_column)
    fallback_source = infer_source_from_filename(path)

    by_source: dict[str, dict[str, int]] = defaultdict(empty_counter)
    file_totals = empty_counter()

    for batch in pf.iter_batches(batch_size=batch_size, columns=read_columns):
        texts_raw = batch.column(0).to_pylist()
        texts = [text if isinstance(text, str) else "" for text in texts_raw]
        if has_source_column:
            sources_raw = batch.column(1).to_pylist()
            sources = [
                source if isinstance(source, str) and source else fallback_source
                for source in sources_raw
            ]
        else:
            sources = [fallback_source] * len(texts)

        encodings = _TOKENIZER.encode_batch(texts, add_special_tokens=False) if texts else []
        for source, text, encoding in zip(sources, texts, encodings):
            token_count = len(encoding.ids)
            char_count = len(text)
            byte_count = len(text.encode("utf-8"))
            row = by_source[source]
            row["rows"] += 1
            row["nonempty_rows"] += 1 if text else 0
            row["chars"] += char_count
            row["utf8_bytes"] += byte_count
            row["tokens_no_eod"] += token_count
            row["tokens_with_eod"] += token_count + 1

            file_totals["rows"] += 1
            file_totals["nonempty_rows"] += 1 if text else 0
            file_totals["chars"] += char_count
            file_totals["utf8_bytes"] += byte_count
            file_totals["tokens_no_eod"] += token_count
            file_totals["tokens_with_eod"] += token_count + 1

    for row in by_source.values():
        row["files"] += 1
    file_totals["files"] = 1

    out = {
        "path": str(path),
        "file": path.name,
        "by_source": by_source,
        "totals": file_totals,
        "wall_seconds": time.time() - started,
        "from_cache": False,
    }
    tmp_path = partial_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(partial_path)
    return out


def merge_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_source: dict[str, dict[str, int]] = defaultdict(empty_counter)
    for item in rows:
        for source, counts in item["by_source"].items():
            out = by_source[source]
            for key in empty_counter():
                out[key] += int(counts.get(key, 0))

    source_rows: list[dict[str, Any]] = []
    for source, counts in by_source.items():
        row: dict[str, Any] = {"source_dataset": source, **counts}
        tokens = int(row["tokens_no_eod"])
        chars = int(row["chars"])
        row["chars_per_token_no_eod"] = chars / tokens if tokens else None
        row["tokens_per_row_no_eod"] = tokens / int(row["rows"]) if int(row["rows"]) else None
        source_rows.append(row)
    source_rows.sort(key=lambda row: int(row["tokens_no_eod"]), reverse=True)

    totals = empty_counter()
    for row in source_rows:
        for key in empty_counter():
            totals[key] += int(row[key])
    totals["source_datasets"] = len(source_rows)
    totals["worker_wall_seconds_sum"] = sum(float(row["wall_seconds"]) for row in rows)
    totals["chars_per_token_no_eod"] = (
        totals["chars"] / totals["tokens_no_eod"] if totals["tokens_no_eod"] else None
    )
    totals["tokens_per_row_no_eod"] = (
        totals["tokens_no_eod"] / totals["rows"] if totals["rows"] else None
    )
    return totals, source_rows


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    tokenizer_path: str,
    started: float,
) -> None:
    totals, source_rows = merge_rows(rows)
    totals["job_wall_seconds"] = time.time() - started

    summary = {
        "tokenizer": tokenizer_path,
        "totals": totals,
        "notes": "Token count from staged parquet files; no special tokens in tokens_no_eod.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "by_source.json").write_text(
        json.dumps(source_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "source_dataset",
        "files",
        "rows",
        "nonempty_rows",
        "chars",
        "utf8_bytes",
        "tokens_no_eod",
        "tokens_with_eod",
        "chars_per_token_no_eod",
        "tokens_per_row_no_eod",
    ]
    with (output_dir / "by_source.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glob", action="append", required=True, help="Input parquet glob. Repeatable.")
    parser.add_argument("--exclude-basename-prefix", action="append", default=[], help="Skip input basenames with this prefix.")
    parser.add_argument("--tokenizer", required=True, help="Local HF tokenizer directory or tokenizer.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--source-column", default="source_dataset")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    started = time.time()
    paths = expand_inputs(args.source_glob, args.exclude_basename_prefix)
    output_dir = args.output_dir
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    print("=== count_source_tokens ===", flush=True)
    print(f"inputs: {len(paths)} files", flush=True)
    print(f"tokenizer: {args.tokenizer}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    if args.exclude_basename_prefix:
        print(f"exclude_basename_prefix: {args.exclude_basename_prefix}", flush=True)

    tasks = [
        (str(path), str(partial_dir), args.batch_size, args.text_column, args.source_column)
        for path in paths
    ]
    rows: list[dict[str, Any]] = []
    done = 0
    if args.workers <= 1:
        init_worker(args.tokenizer)
        iterator = map(count_file, tasks)
    else:
        pool = cf.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(args.tokenizer,),
        )
        iterator = pool.map(count_file, tasks)
    try:
        for row in iterator:
            rows.append(row)
            done += 1
            if done == 1 or done % 5 == 0 or done == len(tasks):
                tokens = sum(int(item["totals"]["tokens_no_eod"]) for item in rows)
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"progress files={done}/{len(tasks)} "
                    f"rows={sum(int(item['totals']['rows']) for item in rows):,} "
                    f"tokens={tokens:,} rate={tokens / elapsed / 1e6:.2f}M tok/s",
                    flush=True,
                )
    finally:
        if args.workers > 1:
            pool.shutdown()

    write_outputs(output_dir, rows, args.tokenizer, started)
    print(json.dumps(json.loads((output_dir / "summary.json").read_text())["totals"], indent=2), flush=True)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)
    print(f"wrote {output_dir / 'by_source.csv'}", flush=True)
    print(f"wrote {output_dir / 'by_source.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

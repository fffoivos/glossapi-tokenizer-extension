#!/usr/bin/env python3
"""Count tokenizer tokens in staged HPLT parquet files.

The script is resumable at file granularity: each input parquet writes one
JSON partial under `<output_dir>/partials/`. Re-running the job skips completed
partials and only processes missing files.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

_TOKENIZER: Any = None


def expand_inputs(patterns: list[str]) -> list[Path]:
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
    if not unique:
        raise SystemExit(f"No parquet files matched: {patterns}")
    return unique


def partial_name(path: Path) -> str:
    safe = path.name.replace("/", "__")
    return f"{safe}.json"


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    from tokenizers import Tokenizer  # type: ignore

    path = Path(tokenizer_path)
    tokenizer_json = path / "tokenizer.json" if path.is_dir() else path
    if not tokenizer_json.exists():
        raise RuntimeError(f"tokenizer.json not found at {tokenizer_json}")
    _TOKENIZER = Tokenizer.from_file(str(tokenizer_json))


def count_file(args: tuple[str, str, int, str]) -> dict[str, Any]:
    path_raw, partial_dir_raw, batch_size, text_column = args
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
    if text_column not in set(pf.schema_arrow.names):
        raise RuntimeError(f"{path} has no {text_column!r} column; columns={pf.schema_arrow.names}")

    rows = 0
    nonempty_rows = 0
    chars = 0
    utf8_bytes = 0
    tokens = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column]):
        texts_raw = batch.column(0).to_pylist()
        texts = [text if isinstance(text, str) else "" for text in texts_raw]
        rows += len(texts)
        nonempty_rows += sum(1 for text in texts if text)
        chars += sum(len(text) for text in texts)
        utf8_bytes += sum(len(text.encode("utf-8")) for text in texts)
        if texts:
            encodings = _TOKENIZER.encode_batch(texts, add_special_tokens=False)
            tokens += sum(len(encoding.ids) for encoding in encodings)

    elapsed = time.time() - started
    out = {
        "path": str(path),
        "file": path.name,
        "rows": rows,
        "nonempty_rows": nonempty_rows,
        "chars": chars,
        "utf8_bytes": utf8_bytes,
        "tokens_no_eod": tokens,
        "tokens_with_eod": tokens + rows,
        "wall_seconds": elapsed,
        "from_cache": False,
    }
    tmp_path = partial_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(partial_path)
    return out


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], tokenizer_path: str, started: float) -> None:
    totals = {
        "files": len(rows),
        "rows": sum(int(row["rows"]) for row in rows),
        "nonempty_rows": sum(int(row["nonempty_rows"]) for row in rows),
        "chars": sum(int(row["chars"]) for row in rows),
        "utf8_bytes": sum(int(row["utf8_bytes"]) for row in rows),
        "tokens_no_eod": sum(int(row["tokens_no_eod"]) for row in rows),
        "tokens_with_eod": sum(int(row["tokens_with_eod"]) for row in rows),
        "worker_wall_seconds_sum": sum(float(row["wall_seconds"]) for row in rows),
        "job_wall_seconds": time.time() - started,
    }
    totals["chars_per_token_no_eod"] = (
        totals["chars"] / totals["tokens_no_eod"] if totals["tokens_no_eod"] else None
    )
    totals["tokens_per_row_no_eod"] = (
        totals["tokens_no_eod"] / totals["rows"] if totals["rows"] else None
    )

    summary = {
        "tokenizer": tokenizer_path,
        "totals": totals,
        "notes": "HPLT token count from staged parquet files; no special tokens in tokens_no_eod.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "per_file.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "file",
            "rows",
            "nonempty_rows",
            "chars",
            "utf8_bytes",
            "tokens_no_eod",
            "tokens_with_eod",
            "wall_seconds",
            "from_cache",
            "path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["file"]):
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-glob", action="append", required=True, help="Input parquet glob. Repeatable.")
    parser.add_argument("--tokenizer", required=True, help="Local HF tokenizer directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    started = time.time()
    paths = expand_inputs(args.source_glob)
    output_dir = args.output_dir
    partial_dir = output_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    print("=== count_hplt_tokens ===", flush=True)
    print(f"inputs: {len(paths)} files", flush=True)
    print(f"tokenizer: {args.tokenizer}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)

    tasks = [(str(path), str(partial_dir), args.batch_size, args.text_column) for path in paths]
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
            if done == 1 or done % 10 == 0 or done == len(tasks):
                tokens = sum(int(item["tokens_no_eod"]) for item in rows)
                elapsed = max(time.time() - started, 1e-6)
                rate = tokens / elapsed
                print(
                    f"progress files={done}/{len(tasks)} rows={sum(int(item['rows']) for item in rows):,} "
                    f"tokens={tokens:,} rate={rate/1e6:.2f}M tok/s",
                    flush=True,
                )
    finally:
        if args.workers > 1:
            pool.shutdown()

    write_outputs(output_dir, rows, args.tokenizer, started)
    print(json.dumps(json.loads((output_dir / "summary.json").read_text())["totals"], indent=2), flush=True)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)
    print(f"wrote {output_dir / 'per_file.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

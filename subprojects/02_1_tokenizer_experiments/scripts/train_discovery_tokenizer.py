#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from time import time

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discovery tokenizer from parquet text exports using the base Apertus fast tokenizer.")
    parser.add_argument("--base-tokenizer", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--input-glob", action="append", required=True, help="Parquet glob(s) with a text column.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--name", default="discovery_bpe")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matches)
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError(f"No input parquet files matched: {patterns}")
    return unique


def collect_input_stats(paths: list[Path], text_column: str) -> dict[str, int]:
    file_count = len(paths)
    byte_count = 0
    row_count = 0
    for path in paths:
        byte_count += path.stat().st_size
        meta = pq.ParquetFile(path).metadata
        row_count += meta.num_rows
    return {"file_count": file_count, "byte_count": byte_count, "row_count": row_count, "text_column": text_column}


def iter_text_batches(paths: list[Path], text_column: str, batch_size: int):
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=[text_column], batch_size=batch_size):
            col = batch.column(0)
            yield [value.as_py() for value in col if value is not None]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = expand_inputs(args.input_glob)
    stats = collect_input_stats(input_paths, args.text_column)

    base = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if not getattr(base, "is_fast", False):
        raise ValueError("Base tokenizer must be a fast tokenizer")

    started = time()
    trained = base.train_new_from_iterator(
        iter_text_batches(input_paths, args.text_column, args.batch_size),
        vocab_size=args.vocab_size,
        length=stats["row_count"],
    )
    elapsed = time() - started

    trained.save_pretrained(output_dir)

    summary = {
        "name": args.name,
        "base_tokenizer": args.base_tokenizer,
        "output_dir": str(output_dir),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": int(trained.vocab_size),
        "input": {
            "patterns": args.input_glob,
            "files": [str(p) for p in input_paths],
            **stats,
        },
        "runtime_seconds": elapsed,
        "special_tokens_map": trained.special_tokens_map,
        "tokenizer_config": {
            "is_fast": bool(trained.is_fast),
            "bos_token_id": trained.bos_token_id,
            "eos_token_id": trained.eos_token_id,
            "pad_token_id": trained.pad_token_id,
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

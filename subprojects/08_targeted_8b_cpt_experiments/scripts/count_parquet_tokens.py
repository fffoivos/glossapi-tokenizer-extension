#!/usr/bin/env python3
"""Count production-tokenizer text plus one EOD per immutable Parquet row."""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path

import pyarrow.parquet as pq

from contract_utils import TOKENIZER_SHA256, file_binding, read_json, require, sha256_file, write_json_atomic


TOKENIZER = None


def init_worker(tokenizer_json: str) -> None:
    global TOKENIZER
    from tokenizers import Tokenizer

    TOKENIZER = Tokenizer.from_file(tokenizer_json)


def count_text(text: object) -> int:
    if TOKENIZER is None:
        raise RuntimeError("tokenizer worker is not initialized")
    return len(TOKENIZER.encode("" if text is None else str(text), add_special_tokens=False).ids)


def parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {path}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--schema-version", default="targeted_8b_parquet_token_receipt_v1")
    parser.add_argument("--release-source-selection-audit", type=Path)
    parser.add_argument("--source-dataset", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite token receipt: {args.output}")
    require(sha256_file(args.tokenizer_json) == TOKENIZER_SHA256, "production tokenizer hash drift")
    release_selection: dict[str, object] | None = None
    sources = list(dict.fromkeys(args.source_dataset or []))
    require(len(sources) == len(args.source_dataset or []), "duplicate --source-dataset value")
    if args.release_source_selection_audit is not None:
        audit = read_json(args.release_source_selection_audit)
        require(
            audit.get("schema_version") == "targeted_8b_release_polytonic_source_audit_v1"
            and audit.get("status") == "passed"
            and audit.get("selection_authority") == "pinned_hf_release_only",
            "release polytonic source audit drift",
        )
        require(sources and audit.get("source_datasets") == sources, "release polytonic source-set drift")
        release_selection = {
            "selection_authority": "pinned_hf_release_only",
            "source_datasets": sources,
            "release_source_selection_audit": file_binding(args.release_source_selection_audit),
        }
    else:
        require(not sources, "--source-dataset requires --release-source-selection-audit")
    files = parquet_files(args.input)
    rows = 0
    text_tokens = 0
    context = mp.get_context("fork")
    with context.Pool(args.workers, initializer=init_worker, initargs=(str(args.tokenizer_json),)) as pool:
        for path in files:
            source = pq.ParquetFile(path)
            for batch in source.iter_batches(columns=["text"], batch_size=256):
                values = batch.column(0).to_pylist()
                rows += len(values)
                text_tokens += sum(pool.map(count_text, values, chunksize=4))
    if args.expected_rows is not None:
        require(rows == args.expected_rows, f"token-count row drift: {rows} != {args.expected_rows}")
    payload = {
        "schema_version": args.schema_version,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_files": [file_binding(path) for path in files],
        "rows": rows,
        "text_tokens": text_tokens,
        "eod_tokens": rows,
        "training_tokens": text_tokens + rows,
        "tokenizer_json": file_binding(args.tokenizer_json),
        "tokenizer_json_sha256": TOKENIZER_SHA256,
        "data_modified": False,
    }
    if release_selection is not None:
        payload.update(release_selection)
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

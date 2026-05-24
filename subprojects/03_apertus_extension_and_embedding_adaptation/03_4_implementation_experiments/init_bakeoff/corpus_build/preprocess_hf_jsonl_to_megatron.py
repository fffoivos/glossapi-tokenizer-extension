#!/usr/bin/env python3
"""Tokenize JSONL text into Megatron indexed-dataset files without torch.

This is a CPU/xfer fallback for environments where Megatron's
tools/preprocess_data.py cannot run because the xfer nodes do not expose uenv or
torch. It intentionally writes the same uncompressed indexed-dataset format used
by Megatron-LM for a single JSON key at document level.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer


INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODE_INT32 = 4
DTYPE_SIZE_INT32 = 4

_TOKENIZER = None
_JSON_KEY = "text"
_APPEND_EOD = True
_EOS_ID = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Final Megatron data prefix; writes <prefix>.bin and <prefix>.idx.",
    )
    parser.add_argument("--tokenizer-model", required=True, help="HF tokenizer dir.")
    parser.add_argument("--json-key", default="text")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    parser.add_argument("--no-append-eod", action="store_true")
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def init_worker(tokenizer_model: str, json_key: str, append_eod: bool) -> None:
    global _TOKENIZER, _JSON_KEY, _APPEND_EOD, _EOS_ID
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_model, local_files_only=True)
    _JSON_KEY = json_key
    _APPEND_EOD = append_eod
    _EOS_ID = _TOKENIZER.eos_token_id
    if _APPEND_EOD and _EOS_ID is None:
        raise ValueError("Tokenizer has no eos_token_id for --append-eod behavior")


def encode_line(line: str) -> tuple[list[int], int]:
    data = json.loads(line)
    text = data[_JSON_KEY]
    if not isinstance(text, str):
        raise TypeError(f"JSON key {_JSON_KEY!r} must contain a string")
    ids = _TOKENIZER(text, add_special_tokens=True).input_ids
    if ids and _APPEND_EOD:
        ids.append(_EOS_ID)
    return ids, len(line)


def iter_lines(path: Path, limit: int) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if limit and idx > limit:
                break
            yield line


def write_index(idx_path: Path, sequence_lengths: list[int], document_indices: list[int]) -> None:
    lengths = np.asarray(sequence_lengths, dtype=np.int32)
    pointers = np.empty(len(sequence_lengths), dtype=np.int64)
    offset = 0
    for i, length in enumerate(sequence_lengths):
        pointers[i] = offset
        offset += length * DTYPE_SIZE_INT32
    docs = np.asarray(document_indices, dtype=np.int64)

    with idx_path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", DTYPE_CODE_INT32))
        handle.write(struct.pack("<Q", len(sequence_lengths)))
        handle.write(struct.pack("<Q", len(document_indices)))
        handle.write(lengths.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(docs.tobytes(order="C"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    bin_path = output_prefix.with_suffix(output_prefix.suffix + ".bin")
    idx_path = output_prefix.with_suffix(output_prefix.suffix + ".idx")
    manifest_path = Path(args.manifest) if args.manifest else output_prefix.with_suffix(
        output_prefix.suffix + ".manifest.json"
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sequence_lengths: list[int] = []
    document_indices = [0]
    total_bytes = 0
    total_tokens = 0
    rows = 0

    pool = mp.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.tokenizer_model, args.json_key, not args.no_append_eod),
    )
    try:
        encoded = pool.imap(encode_line, iter_lines(input_path, args.limit), chunksize=args.chunk_size)
        with bin_path.open("wb") as bin_handle:
            for rows, (ids, n_bytes) in enumerate(encoded, start=1):
                total_bytes += n_bytes
                if ids:
                    arr = np.asarray(ids, dtype=np.int32)
                    bin_handle.write(arr.tobytes(order="C"))
                    sequence_lengths.append(int(arr.size))
                    total_tokens += int(arr.size)
                document_indices.append(len(sequence_lengths))
                if args.log_interval and rows % args.log_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    mb_s = total_bytes / elapsed / 1024 / 1024
                    docs_s = rows / elapsed
                    print(
                        f"Processed {rows} documents ({docs_s:.2f} docs/s, {mb_s:.2f} MB/s)",
                        flush=True,
                    )
    finally:
        pool.close()
        pool.join()

    write_index(idx_path, sequence_lengths, document_indices)

    manifest = {
        "input": str(input_path),
        "output_prefix": str(output_prefix),
        "bin": str(bin_path),
        "idx": str(idx_path),
        "tokenizer_model": args.tokenizer_model,
        "json_key": args.json_key,
        "append_eod": not args.no_append_eod,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "limit": args.limit,
        "rows": rows,
        "sequences": len(sequence_lengths),
        "documents": len(document_indices),
        "tokens": total_tokens,
        "bytes_read": total_bytes,
        "wall_seconds": time.time() - started,
        "format": "Megatron indexed dataset v1, int32 document-level",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

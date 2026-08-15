#!/usr/bin/env python3
"""Freeze one historical-148480 Megatron indexed dataset with row/token checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic
from finalize_training_megatron import validate_runtime


TOKENIZER_SHA256 = "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394"
INDEX_MAGIC = b"MMIDIDX\x00\x00"
DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


BASE_STREAMS = {"hplt", "openarchives", "foreign", "old_greek"}
PHASE3_STREAM_TO_POOL = {
    "phase3_openarchives": "openarchives",
    "phase3_foreign": "foreign_replay",
    "phase3_old_greek": "old_greek_replay",
}
STREAM_TO_POOL = {
    "hplt": "hplt",
    "openarchives": "openarchives",
    "foreign": "foreign_replay",
    "old_greek": "old_greek_replay",
    **PHASE3_STREAM_TO_POOL,
}


def upstream_output(receipt: dict, stream: str) -> dict:
    if receipt.get("schema_version") == "apertus_hard_h_to_g_stage_b_stream_v1":
        require(receipt.get("stream") == stream and receipt.get("status") == "passed", "Stage-B stream receipt drift")
        return receipt["output"]
    if receipt.get("schema_version") == "apertus_hard_h_to_g_replay_split_v1":
        require(receipt.get("status") == "passed" and stream in {"foreign", "old_greek"}, "replay split receipt drift")
        return receipt["outputs"][stream]
    require(receipt.get("schema_version") == "apertus_phase3_unseen_catalog_receipt_v1", "tokenization upstream schema drift")
    pool = PHASE3_STREAM_TO_POOL.get(stream)
    require(receipt.get("status") == "passed" and pool is not None, "Phase-3 stream receipt drift")
    return receipt["pools"][pool]["output"]


def inspect_index(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        require(handle.read(len(INDEX_MAGIC)) == INDEX_MAGIC, "Megatron index magic drift")
        version = struct.unpack("<Q", handle.read(8))[0]
        dtype_code = struct.unpack("<B", handle.read(1))[0]
        sequences = struct.unpack("<Q", handle.read(8))[0]
        document_indices = struct.unpack("<Q", handle.read(8))[0]
        offset = handle.tell()
    require(version == 1 and dtype_code in DTYPE_SIZES, "Megatron index header drift")
    expected_bytes = offset + sequences * 4 + sequences * 8 + document_indices * 8
    require(path.stat().st_size == expected_bytes, "Megatron index byte geometry drift")
    lengths = np.memmap(path, mode="r", dtype="<i4", offset=offset, shape=(sequences,))
    require(bool(np.all(lengths > 0)), "Megatron index has non-positive sequence lengths")
    tokens = int(np.sum(lengths, dtype=np.int64))
    del lengths
    require(document_indices >= 1, "Megatron index has no document boundary array")
    return {
        "version": version,
        "dtype_code": dtype_code,
        "dtype_bytes": DTYPE_SIZES[dtype_code],
        "sequences": sequences,
        "document_indices": document_indices,
        "documents": document_indices - 1,
        "tokens_including_eod": tokens,
    }


def document_identity(row: dict, line_number: int) -> tuple[str, str, str, str]:
    text = row.get("text")
    source_dataset = row.get("source_dataset") or row.get("source")
    source_doc_id = row.get("source_doc_id") or row.get("doc_id") or row.get("id")
    require(isinstance(text, str) and text, f"tokenized input row {line_number} has no text")
    require(source_dataset not in (None, "") and source_doc_id not in (None, ""), f"tokenized input row {line_number} has no stable identity")
    source_dataset = str(source_dataset)
    source_doc_id = str(source_doc_id)
    key = hashlib.sha256((source_dataset + "\0" + source_doc_id).encode("utf-8")).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return source_dataset, source_doc_id, key, text_hash


def write_document_catalog(input_jsonl: Path, idx_path: Path, stream: str, output: Path) -> dict[str, object]:
    require(not output.exists(), f"immutable document catalog exists: {output}")
    with idx_path.open("rb") as handle:
        handle.seek(len(INDEX_MAGIC) + 8 + 1)
        sequences = struct.unpack("<Q", handle.read(8))[0]
        handle.read(8)  # document-index length
        lengths_offset = handle.tell()
    lengths = np.memmap(idx_path, mode="r", dtype="<i4", offset=lengths_offset, shape=(sequences,))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    temporary = Path(name)
    rows = 0
    tokens = 0
    try:
        with input_jsonl.open(encoding="utf-8") as source, os.fdopen(descriptor, "w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                require(rows < sequences, "document catalog has more rows than indexed sequences")
                row = json.loads(line)
                source_dataset, source_doc_id, key, text_hash = document_identity(row, line_number)
                token_count = int(lengths[rows])
                require(token_count > 0, f"document catalog row {line_number} has no tokens")
                target.write(json.dumps({
                    "stream": stream,
                    "pool": STREAM_TO_POOL[stream],
                    "document_index": rows,
                    "input_row_index": rows,
                    "source_dataset": source_dataset,
                    "source_doc_id": source_doc_id,
                    "document_key_sha256": key,
                    "document_text_sha256": text_hash,
                    "token_count": token_count,
                }, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
                tokens += token_count
            target.flush(); os.fsync(target.fileno())
        require(rows == sequences, "document catalog row/index sequence count drift")
        os.link(temporary, output); temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        del lengths
    return {**file_binding(output), "rows": rows, "tokens_including_eod": tokens}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=tuple(sorted(BASE_STREAMS | set(PHASE3_STREAM_TO_POOL))), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--megatron-receipt", type=Path, required=True)
    parser.add_argument("--dataset-prefix", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--document-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable tokenization receipt exists: {args.output}")
    upstream = read_json(args.input_receipt)
    block = upstream_output(upstream, args.stream)
    require(block["sha256"] == sha256_file(args.input_jsonl), "tokenization input SHA drift")
    require(block["bytes"] == args.input_jsonl.stat().st_size, "tokenization input byte drift")
    expected_rows = int(block["rows"])
    tokenizer = read_json(args.tokenizer_receipt)
    require(tokenizer.get("schema_version") == "apertus_historical_tokenizer_148480_v1" and tokenizer.get("status") == "passed", "tokenizer receipt drift")
    tokenizer_json = args.tokenizer_root / "tokenizer.json"
    require(tokenizer_json.is_file() and sha256_file(tokenizer_json) == TOKENIZER_SHA256, "tokenizer bytes drift")
    require(tokenizer["files"]["tokenizer.json"]["sha256"] == TOKENIZER_SHA256, "tokenizer receipt hash drift")
    megatron = read_json(args.megatron_receipt)
    validate_runtime(megatron, args.megatron_root, Path(megatron["patch"]["path"]))
    bin_path = Path(f"{args.dataset_prefix}.bin")
    idx_path = Path(f"{args.dataset_prefix}.idx")
    require(bin_path.is_file() and idx_path.is_file(), "tokenized Megatron payload missing")
    index = inspect_index(idx_path)
    require(index["documents"] == expected_rows, "tokenized document count differs from input rows")
    require(index["sequences"] == expected_rows, "preprocess_data emitted unexpected multi-sequence documents")
    require(bin_path.stat().st_size == index["tokens_including_eod"] * index["dtype_bytes"], "Megatron binary/token geometry drift")
    if args.stream in PHASE3_STREAM_TO_POOL:
        require(
            int(block.get("tokens", -1)) == index["tokens_including_eod"],
            "Phase-3 tokenizer-counted capacity differs from Megatron preprocessing",
        )
    require(args.workers == 64, "historical preprocessing worker geometry drift")
    document_catalog = write_document_catalog(args.input_jsonl, idx_path, args.stream, args.document_catalog)
    require(document_catalog["rows"] == expected_rows, "document catalog input-row drift")
    require(document_catalog["tokens_including_eod"] == index["tokens_including_eod"], "document catalog token drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_tokenized_stream_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "stream": args.stream,
        "input": file_binding(args.input_jsonl),
        "input_receipt": file_binding(args.input_receipt),
        "tokenizer_receipt": file_binding(args.tokenizer_receipt),
        "tokenizer_json": file_binding(tokenizer_json),
        "megatron_receipt": file_binding(args.megatron_receipt),
        "megatron_root": str(args.megatron_root.resolve()),
        "command_contract": {
            "tool": "tools/preprocess_data.py",
            "tokenizer_type": "HuggingFaceTokenizer",
            "append_eod": True,
            "json_keys": ["text"],
            "workers": args.workers,
        },
        "dataset_prefix": str(args.dataset_prefix.resolve()),
        "files": {"bin": file_binding(bin_path), "idx": file_binding(idx_path)},
        "document_catalog": document_catalog,
        "index": index,
        "invariants": {
            "one_indexed_document_per_input_row": True,
            "tokenizer_is_historical_148480": True,
            "append_eod_enabled": True,
            "additional_deduplication": False,
        },
    }
    write_json_atomic(args.output, payload)
    print(json.dumps({"stream": args.stream, "documents": expected_rows, "tokens": index["tokens_including_eod"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an exact tokenizer-counted document catalog from immutable JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

from transformers import AutoTokenizer

from contract_utils import executing_code_bundle, file_binding, read_json, require, sha256_file, write_json_atomic


TOKENIZER_SHA256 = "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394"


def upstream_output(receipt: dict) -> dict:
    for name in ("output", "clean"):
        value = receipt.get(name)
        if isinstance(value, dict) and value.get("sha256") and value.get("rows") is not None:
            return value
    raise ValueError("document-catalog upstream receipt has no row-bound output")


def identity(row: dict, line_number: int) -> tuple[str, str, str, str]:
    text = row.get("text")
    source_dataset = row.get("source_dataset") or row.get("source")
    source_doc_id = row.get("source_doc_id") or row.get("doc_id") or row.get("id")
    require(isinstance(text, str) and text, f"row {line_number}: missing text")
    require(source_dataset not in (None, "") and source_doc_id not in (None, ""), f"row {line_number}: missing identity")
    source_dataset = str(source_dataset)
    source_doc_id = str(source_doc_id)
    key = hashlib.sha256((source_dataset + "\0" + source_doc_id).encode("utf-8")).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return source_dataset, source_doc_id, key, text_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=("phase3_openarchives_candidates",), required=True)
    parser.add_argument("--pool", choices=("openarchives",), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(args.batch_size == 512, "document-catalog batch geometry drift")
    require(not args.output_jsonl.exists() and not args.output_receipt.exists(), "immutable document catalog exists")
    upstream = read_json(args.input_receipt)
    require(
        upstream.get("schema_version") in {
            "apertus_hard_h_to_g_stage_b_stream_v1",
            "apertus_hard_h_to_g_published_anonymized_stream_v1",
        }
        and upstream.get("status") == "passed"
        and upstream.get("stream") == args.stream,
        "document-catalog upstream identity drift",
    )
    block = upstream_output(upstream)
    require(
        args.input_jsonl.is_file()
        and args.input_jsonl.stat().st_size == int(block.get("bytes", -1))
        and isinstance(block.get("sha256"), str)
        and int(block.get("rows", -1)) > 0,
        "document-catalog source binding drift",
    )
    # The input's SHA-256 was established by its immutable upstream receipt.
    # Rehashing 38GB here would add a redundant pre-tokenization scan.
    source_binding = {"path": str(args.input_jsonl.resolve()), "bytes": args.input_jsonl.stat().st_size, "sha256": block["sha256"]}
    tokenizer_receipt = read_json(args.tokenizer_receipt)
    tokenizer_json = args.tokenizer_root / "tokenizer.json"
    require(tokenizer_receipt.get("schema_version") == "apertus_historical_tokenizer_148480_v1" and tokenizer_receipt.get("status") == "passed", "document-catalog tokenizer receipt drift")
    require(sha256_file(tokenizer_json) == TOKENIZER_SHA256 and tokenizer_receipt.get("files", {}).get("tokenizer.json", {}).get("sha256") == TOKENIZER_SHA256, "document-catalog tokenizer bytes drift")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, local_files_only=True, use_fast=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{args.output_jsonl.name}.", suffix=".partial", dir=args.output_jsonl.parent)
    temporary = Path(name)
    rows = 0
    tokens = 0
    batch_rows: list[tuple[int, str, str, str, str]] = []

    def flush(output) -> None:
        nonlocal rows, tokens, batch_rows
        if not batch_rows:
            return
        encoded = tokenizer([row[1] for row in batch_rows], add_special_tokens=False, padding=False, truncation=False, return_length=True)
        lengths = encoded.get("length")
        require(isinstance(lengths, list) and len(lengths) == len(batch_rows), "tokenizer length batch drift")
        for (input_row_index, _text, source_dataset, source_doc_id, identity_json), length in zip(batch_rows, lengths, strict=True):
            key, text_hash = json.loads(identity_json)
            token_count = int(length) + 1  # historical preprocess_data --append-eod
            require(token_count > 1, f"candidate row {input_row_index} tokenized empty")
            output.write(json.dumps({
                "stream": args.stream,
                "pool": args.pool,
                "document_index": input_row_index,
                "input_row_index": input_row_index,
                "source_dataset": source_dataset,
                "source_doc_id": source_doc_id,
                "document_key_sha256": key,
                "document_text_sha256": text_hash,
                "token_count": token_count,
            }, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
            tokens += token_count
        batch_rows = []

    try:
        with args.input_jsonl.open(encoding="utf-8") as source, os.fdopen(descriptor, "w", encoding="utf-8") as output:
            input_row_index = 0
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_dataset, source_doc_id, key, text_hash = identity(row, line_number)
                batch_rows.append((input_row_index, row["text"], source_dataset, source_doc_id, json.dumps((key, text_hash))))
                input_row_index += 1
                if len(batch_rows) == args.batch_size:
                    flush(output)
            flush(output)
            output.flush(); os.fsync(output.fileno())
        require(rows == int(block["rows"]), "document-catalog row-count drift")
        os.link(temporary, args.output_jsonl); temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema_version": "apertus_hard_h_to_g_document_catalog_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "stream": args.stream,
        "pool": args.pool,
        "source": source_binding,
        "source_receipt": file_binding(args.input_receipt),
        "tokenizer_receipt": file_binding(args.tokenizer_receipt),
        "tokenizer_json": file_binding(tokenizer_json),
        "document_catalog": {**file_binding(args.output_jsonl), "rows": rows, "tokens_including_eod": tokens},
        "command_contract": {"add_special_tokens": False, "append_eod": True, "batch_size": args.batch_size},
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

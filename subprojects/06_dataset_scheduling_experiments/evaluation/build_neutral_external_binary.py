#!/usr/bin/env python3
"""Tokenize the frozen neutral external Greek corpus into a Megatron heldout."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np


INDEX_HEADER = b"MMIDIDX\x00\x00"
MIN_TOKENS = 10_000_000
MAX_TOKENS = 20_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_index(path: Path, lengths: list[int]) -> tuple[int, int]:
    values = np.asarray(lengths, dtype=np.int32)
    pointers = np.empty(len(values), dtype=np.int64)
    offset = 0
    for index, length in enumerate(values):
        pointers[index] = offset
        offset += int(length) * 4
    documents = np.arange(len(values) + 1, dtype=np.int64)
    with path.open("wb") as handle:
        handle.write(INDEX_HEADER)
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<B", 4))
        handle.write(struct.pack("<Q", len(values)))
        handle.write(struct.pack("<Q", len(documents)))
        handle.write(values.tobytes(order="C"))
        handle.write(pointers.tobytes(order="C"))
        handle.write(documents.tobytes(order="C"))
        handle.flush()
        os.fsync(handle.fileno())
    return len(values), int(values.astype(np.int64).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(str(args.output_prefix) + ".manifest.json")
    outputs = [
        Path(str(args.output_prefix) + suffix)
        for suffix in (".bin", ".idx", ".retained.jsonl", ".manifest.json")
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to replace neutral external heldout outputs")
    corpus = read(args.corpus_receipt)
    if corpus.get("schema_version") != "apertus_mini_neutral_external_corpus_v1" or corpus.get("status") != "frozen":
        raise ValueError("neutral external corpus is not frozen")
    source = Path(corpus["candidate_jsonl"]["path"])
    if source.stat().st_size != int(corpus["candidate_jsonl"]["bytes"]) or sha256_file(source) != corpus["candidate_jsonl"]["sha256"]:
        raise ValueError("neutral external source drift")
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    if tokenizer.get_vocab_size(with_added_tokens=True) != 148_992:
        raise ValueError("neutral heldout tokenizer vocabulary drift")
    config = read(args.tokenizer_dir / "tokenizer_config.json")
    eos_value = config.get("eos_token")
    eos_token = eos_value.get("content") if isinstance(eos_value, dict) else eos_value
    eos_id = tokenizer.token_to_id(eos_token) if isinstance(eos_token, str) else None
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_tmp = Path(str(args.output_prefix) + ".bin.partial")
    idx_tmp = Path(str(args.output_prefix) + ".idx.partial")
    ledger_tmp = Path(str(args.output_prefix) + ".retained.jsonl.partial")
    lengths = []
    with source.open(encoding="utf-8") as input_handle, bin_tmp.open("wb") as binary, ledger_tmp.open("w", encoding="utf-8") as ledger:
        for line in input_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values = tokenizer.encode(row["text"], add_special_tokens=False).ids
            values.append(eos_id)
            encoded = np.asarray(values, dtype=np.int32)
            binary.write(encoded.tobytes(order="C"))
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            ledger.write(json.dumps({"doc_id": row["cluster_id"], "source_id": row["source_id"], "text_sha256": digest, "tokens": len(values)}, sort_keys=True) + "\n")
            lengths.append(len(values))
        binary.flush(); os.fsync(binary.fileno())
        ledger.flush(); os.fsync(ledger.fileno())
    documents, tokens = write_index(idx_tmp, lengths)
    if documents != int(corpus["documents"]) or not MIN_TOKENS <= tokens <= MAX_TOKENS:
        raise ValueError(f"neutral external heldout must contain 10M-20M tokens, got documents={documents} tokens={tokens}")
    bin_path = Path(str(args.output_prefix) + ".bin")
    idx_path = Path(str(args.output_prefix) + ".idx")
    ledger_path = Path(str(args.output_prefix) + ".retained.jsonl")
    os.replace(bin_tmp, bin_path); os.replace(idx_tmp, idx_path); os.replace(ledger_tmp, ledger_path)
    payload = {
        "schema_version": "full_cpt_megatron_shard_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "heldout",
        "heldout_name": "neutral_external_modern_greek",
        "output_prefix": str(args.output_prefix.resolve()),
        "counts": {"documents": documents, "document_index_entries": documents + 1, "tokens": tokens},
        "tokenizer": {"root": str(args.tokenizer_dir.resolve()), "tokenizer_json_sha256": sha256_file(tokenizer_json), "vocab_size": 148_992},
        "corpus_receipt": {"path": str(args.corpus_receipt.resolve()), "bytes": args.corpus_receipt.stat().st_size, "sha256": sha256_file(args.corpus_receipt)},
        "external_validation": {
            "document_cluster_split": True,
            "global_exact_dedup_against_training": True,
            "global_minhash_dedup_against_training": True,
            "minhash_threshold": 0.85,
            "publishers_or_domains_absent_from_training": corpus["publishers_or_domains_absent_from_training"],
            "source_time_window_absent_from_training": corpus["source_time_window_absent_from_training"],
            "source_separation_rule": "publisher_or_domain_or_time_window",
            "candidate_documents_never_used_for_training": True,
            "source_snapshot_receipts": corpus["source_snapshot_receipts"],
            "dedup_receipt": corpus["dedup_receipt"],
        },
        "outputs": {
            "bin": {"path": str(bin_path.resolve()), "bytes": bin_path.stat().st_size, "sha256": sha256_file(bin_path)},
            "idx": {"path": str(idx_path.resolve()), "bytes": idx_path.stat().st_size, "sha256": sha256_file(idx_path)},
            "retained_ledger": {"path": str(ledger_path.resolve()), "bytes": ledger_path.stat().st_size, "sha256": sha256_file(ledger_path), "rows": documents},
        },
    }
    temporary = Path(str(manifest_path) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps({"ok": True, "documents": documents, "tokens": tokens, "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

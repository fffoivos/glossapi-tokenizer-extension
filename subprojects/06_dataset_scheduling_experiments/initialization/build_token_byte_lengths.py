#!/usr/bin/env python3
"""Freeze UTF-8 byte lengths for every token in the selected tokenizer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output_npy.exists() or args.output_receipt.exists():
        raise FileExistsError("refusing to overwrite token byte-length artifacts")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if len(tokenizer) != 148_992:
        raise ValueError("tokenizer vocabulary size drift")
    values = np.zeros(len(tokenizer), dtype=np.uint16)
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        values[token_id] = len(text.encode("utf-8"))
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_npy) + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output_npy)
    payload = {
        "schema_version": "apertus_token_utf8_byte_lengths_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        "tokenizer_json_sha256": sha256(args.tokenizer_dir / "tokenizer.json"),
        "vocab_size": len(tokenizer),
        "dtype": str(values.dtype),
        "byte_lengths": {
            "path": str(args.output_npy.resolve()),
            "bytes": args.output_npy.stat().st_size,
            "sha256": sha256(args.output_npy),
        },
        "definition": "len(tokenizer.decode([token_id], skip_special_tokens=false, clean_up_tokenization_spaces=false).encode(utf-8))",
    }
    args.output_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "vocab_size": len(tokenizer)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


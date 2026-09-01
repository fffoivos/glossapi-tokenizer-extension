#!/usr/bin/env python3
"""Materialize the byte-exact historical 148,480 tokenizer with a receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil

from contract_utils import executing_code_bundle, require, sha256_file, write_json_atomic


EXPECTED = {
    "tokenizer.json": "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394",
    "tokenizer_config.json": "ea64a17b41e1deaa7469212f413676129f33977ca3a48767f0ca68dc346df502",
    "special_tokens_map.json": "816ec96e37c6d15e3cbc535dc146c898a7218f209fc154384f31fc1e6ad31ba5",
    "manifest.json": "09c84b4086e9f800174942dcba688d8eefa8e40f3b60c20341699a6a7a11e2bf",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"immutable tokenizer output exists: {args.output_root}")
    require(not args.output_receipt.exists(), f"immutable tokenizer receipt exists: {args.output_receipt}")
    for name, digest in EXPECTED.items():
        path = args.source_root / name
        require(path.is_file(), f"tokenizer source missing: {path}")
        require(sha256_file(path) == digest, f"tokenizer source drift: {name}")
    value = json.loads((args.source_root / "tokenizer.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.source_root / "manifest.json").read_text(encoding="utf-8"))
    vocab_size = len(value["model"]["vocab"])
    require(vocab_size == 148_480 and vocab_size % 256 == 0, "tokenizer vocabulary geometry drift")
    require(value.get("normalizer") is None, "tokenizer normalizer drift")
    require(manifest["base_apertus_vocab"] == 131_072, "base vocabulary drift")
    require(manifest["added_modern_greek_c3"] == 17_408, "added vocabulary drift")
    require(manifest["added_polytonic_greek"] == 0, "historical tokenizer unexpectedly contains polytonic stage")
    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        for name in EXPECTED:
            # The immutable contract is byte identity, not preservation of
            # source filesystem metadata.  copy2() also propagates xattrs;
            # Capstor rejects some bundle-source xattrs and can therefore
            # fail after the bytes have already been copied.  Copy only the
            # file contents and verify every published SHA-256 below.
            shutil.copyfile(args.source_root / name, temporary / name)
        os.replace(temporary, args.output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    bindings = {
        name: {
            "path": str((args.output_root / name).resolve()),
            "bytes": (args.output_root / name).stat().st_size,
            "sha256": sha256_file(args.output_root / name),
        }
        for name in EXPECTED
    }
    payload = {
        "schema_version": "apertus_historical_tokenizer_148480_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executing_code_bundle": executing_code_bundle(),
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "files": bindings,
        "vocab_size": vocab_size,
        "make_vocab_size_divisible_by": 256,
        "padding_tokens": 0,
        "base_vocab_size": 131_072,
        "added_modern_greek_tokens": 17_408,
        "added_polytonic_tokens": 0,
        "normalizer": None,
    }
    write_json_atomic(args.output_receipt, payload)
    print(args.output_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

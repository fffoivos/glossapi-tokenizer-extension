#!/usr/bin/env python3
"""Prove Goldfish hash-table eligibility is identical for every added token ID."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, math.isqrt(value) + 1):
        if value % divisor == 0:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-gpt-dataset-source", type=Path, required=True)
    parser.add_argument("--base-vocab-size", type=int, default=131_072)
    parser.add_argument("--target-vocab-size", type=int, default=148_992)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--context-width", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.pinned_gpt_dataset_source.resolve()
    text = source.read_text(encoding="utf-8")
    size_match = re.search(r"^_HASH_TABLE_SIZE\s*=\s*([\d_]+)\s*$", text, re.MULTILINE)
    seed_match = re.search(r"manual_seed\(([\d_]+)\)", text)
    if size_match is None or seed_match is None:
        raise ValueError("could not recover pinned Goldfish table geometry")
    table_size = int(size_match.group(1).replace("_", ""))
    seed = int(seed_match.group(1).replace("_", ""))
    if (table_size, seed, args.k, args.context_width) != (1_000_003, 2_971_215_073, 50, 50):
        raise ValueError("Goldfish production contract drift")
    if not is_prime(table_size):
        raise ValueError("Goldfish hash-table size is not prime")
    token_ids = range(args.base_vocab_size, args.target_vocab_size)
    noninvertible = [token_id for token_id in token_ids if math.gcd(token_id, table_size) != 1]
    if noninvertible:
        raise ValueError(f"added token IDs not invertible modulo hash table: {noninvertible[:10]}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    table = torch.rand(table_size, device="cpu", generator=generator)
    table_bytes = table.numpy().astype(np.float32, copy=False).tobytes(order="C")
    dropped = int(torch.count_nonzero(table < 1 / args.k).item())
    rate = dropped / table_size
    # For a complete residue system and prime modulus, multiplication by every
    # nonzero added token ID is a permutation. Every added token therefore sees
    # exactly the same hash-table values and exact drop count.
    payload = {
        "schema_version": "apertus_mini_goldfish_added_token_uniformity_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pinned_source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "goldfish": {
            "hash_table_size": table_size,
            "hash_seed": seed,
            "hash_table_float32_sha256": hashlib.sha256(table_bytes).hexdigest(),
            "k": args.k,
            "context_width": args.context_width,
            "complete_residue_dropped": dropped,
            "complete_residue_drop_rate": rate,
        },
        "added_tokens": {
            "first_id": args.base_vocab_size,
            "last_id": args.target_vocab_size - 1,
            "count": args.target_vocab_size - args.base_vocab_size,
            "noninvertible_modulo_hash_table": noninvertible,
            "all_share_exact_complete_residue_drop_count": True,
        },
        "proof": (
            "1000003 is prime; each added ID is nonzero and invertible modulo it; "
            "multiplication by that ID permutes the complete residue system, so every "
            "added token indexes the identical multiset of the pinned random hash table"
        ),
        "scope_note": (
            "This proves token-ID neutrality of the hash rule. Real-corpus mask-rate "
            "diagnostics remain observational because context residues need not be uniform."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"ok": True, "added_tokens": payload["added_tokens"]["count"], "rate": rate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

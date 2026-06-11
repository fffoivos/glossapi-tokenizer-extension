#!/usr/bin/env python3
"""Count Ancient/Polytonic added-token firings on the kept corpus.

This computes final tokenizer usage counts, not BPE merge-selection
frequencies. A token can be frequent when selected as a merge and later fire
less often if it is swallowed by longer merges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_hash(text: str) -> str:
    return " ".join(ud.normalize("NFC", text or "").split())


def split_for_text(text: str, seed: int, val_pct: int, test_pct: int) -> str:
    text_hash = stable_hash(normalize_for_hash(text))
    split_hash = stable_hash(text_hash + f":{seed}")
    bucket = int(split_hash[:16], 16) % 10_000
    if bucket < test_pct * 100:
        return "test"
    if bucket < (test_pct + val_pct) * 100:
        return "val"
    return "train"


def is_hygiene_kept(text: str, row: dict[str, object], min_chars: int, max_control_ratio: float) -> bool:
    if text is None:
        return False
    chars = len(text)
    if chars < min_chars:
        return False
    if bool(row.get("is_empty") or False):
        return False
    if text.lstrip().startswith("{\\rtf") or "\\fonttbl" in text[:10_000]:
        return False
    control_chars = len(CONTROL_RE.findall(text))
    control_ratio = (control_chars / chars) if chars else 0.0
    return control_ratio <= max_control_ratio


def split_merge(merge):
    if isinstance(merge, str):
        parts = merge.split(" ", 1)
        return parts if len(parts) == 2 else None
    if isinstance(merge, list) and len(merge) == 2:
        return merge
    return None


def tokenizer_for_cutoff(base_tokenizer_json: Path, full_tokenizer_json: Path, added: int) -> Tokenizer:
    base = json.loads(base_tokenizer_json.read_text(encoding="utf-8"))
    full = json.loads(full_tokenizer_json.read_text(encoding="utf-8"))
    base_vocab = base["model"]["vocab"]
    full_vocab = full["model"]["vocab"]
    base_merges = base["model"]["merges"]
    full_merges = full["model"]["merges"]
    base_size = len(base_vocab)
    if added == len(full_vocab) - base_size:
        return Tokenizer.from_str(json.dumps(full, ensure_ascii=False))
    new_vocab = dict(base_vocab)
    added_merges = full_merges[len(base_merges) : len(base_merges) + added]
    next_id = base_size
    for merge in added_merges:
        parts = split_merge(merge)
        if parts is None:
            raise SystemExit(f"bad merge format: {merge!r}")
        new_vocab[parts[0] + parts[1]] = next_id
        next_id += 1
    payload = json.loads(json.dumps(full))
    payload["model"]["vocab"] = new_vocab
    payload["model"]["merges"] = base_merges + added_merges
    return Tokenizer.from_str(json.dumps(payload, ensure_ascii=False))


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def readable_token(token: str) -> str:
    raw = bytes(BYTE_DECODER[ch] for ch in token)
    return raw.decode("utf-8", errors="backslashreplace")


def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize(counts: Counter[int], ids: list[int]) -> dict[str, object]:
    values = [int(counts.get(i, 0)) for i in ids]
    sorted_values = sorted(values)
    return {
        "tokens": len(ids),
        "used_tokens": sum(1 for v in values if v > 0),
        "zero_count_tokens": sum(1 for v in values if v == 0),
        "total_firings": int(sum(values)),
        "min": int(sorted_values[0]) if sorted_values else 0,
        "p01": percentile(sorted_values, 0.01),
        "p05": percentile(sorted_values, 0.05),
        "p10": percentile(sorted_values, 0.10),
        "p25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.50),
        "p75": percentile(sorted_values, 0.75),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": int(sorted_values[-1]) if sorted_values else 0,
        "tokens_lt_100": sum(1 for v in values if v < 100),
        "tokens_lt_500": sum(1 for v in values if v < 500),
        "tokens_lt_1000": sum(1 for v in values if v < 1000),
        "tokens_ge_1000": sum(1 for v in values if v >= 1000),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--base-tokenizer-json", type=Path, required=True)
    parser.add_argument("--full-tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--added", type=int, action="append", required=True)
    parser.add_argument("--base-vocab-size", type=int, default=148_480)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-control-ratio", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = json.loads(args.full_tokenizer_json.read_text(encoding="utf-8"))
    id_to_raw = {idx: token for token, idx in full["model"]["vocab"].items()}

    tokenizers = {added: tokenizer_for_cutoff(args.base_tokenizer_json, args.full_tokenizer_json, added) for added in args.added}
    counts = {
        added: {
            "hygiene_kept_all": Counter(),
            "poly_train": Counter(),
        }
        for added in args.added
    }
    row_counts = {
        "rows_seen": 0,
        "hygiene_kept_all": 0,
        "poly_train": 0,
        "poly_val": 0,
        "poly_test": 0,
    }
    char_counts = dict.fromkeys(row_counts.keys(), 0)

    pf = pq.ParquetFile(args.input_parquet)
    columns = ["text"]
    schema_names = set(pf.schema.names)
    if "is_empty" in schema_names:
        columns.append("is_empty")

    for batch in pf.iter_batches(columns=columns, batch_size=args.batch_size):
        rows = batch.to_pylist()
        usable_texts: list[str] = []
        train_texts: list[str] = []
        for row in rows:
            text = row.get("text") or ""
            row_counts["rows_seen"] += 1
            char_counts["rows_seen"] += len(text)
            if not is_hygiene_kept(text, row, args.min_chars, args.max_control_ratio):
                continue
            split = split_for_text(text, args.seed, args.val_pct, args.test_pct)
            row_counts["hygiene_kept_all"] += 1
            char_counts["hygiene_kept_all"] += len(text)
            row_counts[f"poly_{split}"] += 1
            char_counts[f"poly_{split}"] += len(text)
            usable_texts.append(text)
            if split == "train":
                train_texts.append(text)

        for added, tok in tokenizers.items():
            final_size = args.base_vocab_size + added
            ids_of_interest = set(range(args.base_vocab_size, final_size))
            if usable_texts:
                for enc in tok.encode_batch(usable_texts):
                    counts[added]["hygiene_kept_all"].update(i for i in enc.ids if i in ids_of_interest)
            if train_texts:
                for enc in tok.encode_batch(train_texts):
                    counts[added]["poly_train"].update(i for i in enc.ids if i in ids_of_interest)

    summary = {
        "input_parquet": str(args.input_parquet),
        "base_tokenizer_json": str(args.base_tokenizer_json),
        "full_tokenizer_json": str(args.full_tokenizer_json),
        "base_vocab_size": args.base_vocab_size,
        "split_policy": {
            "seed": args.seed,
            "val_pct": args.val_pct,
            "test_pct": args.test_pct,
            "hygiene_min_chars": args.min_chars,
            "hygiene_max_control_ratio": args.max_control_ratio,
        },
        "row_counts": row_counts,
        "char_counts": char_counts,
        "cutoffs": {},
    }

    for added in args.added:
        final_size = args.base_vocab_size + added
        ids = list(range(args.base_vocab_size, final_size))
        cutoff_summary = {"final_vocab_size": final_size, "counts": {}}
        for split_name, split_counts in counts[added].items():
            out_tsv = args.output_dir / f"polytonic_added_{added:04d}_{split_name}_token_firings.tsv"
            with out_tsv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["id", "token", "count"], delimiter="\t")
                writer.writeheader()
                for idx in ids:
                    writer.writerow(
                        {
                            "id": idx,
                            "token": readable_token(id_to_raw[idx]),
                            "count": int(split_counts.get(idx, 0)),
                        }
                    )
            cutoff_summary["counts"][split_name] = {
                "path": str(out_tsv),
                **summarize(split_counts, ids),
            }
        summary["cutoffs"][str(added)] = cutoff_summary

    summary_path = args.output_dir / "polytonic_token_firing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

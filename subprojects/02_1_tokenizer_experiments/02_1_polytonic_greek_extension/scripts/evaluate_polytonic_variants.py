#!/usr/bin/env python3
"""Evaluate Ancient/Polytonic Greek cutoff variants on held-out slices."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

GREEK_WORD_RE = re.compile(r"[\u0300-\u036f\u0370-\u03ff\u1f00-\u1fff]+")
DISTINCTIVE_MARKS = {"\u0300", "\u0313", "\u0314", "\u0342", "\u0345"}


def has_distinctive_polytonic(text: str) -> bool:
    return any(ch in DISTINCTIVE_MARKS for ch in ud.normalize("NFD", text or ""))


def greek_words(text: str) -> list[str]:
    return [m.group(0) for m in GREEK_WORD_RE.finditer(text or "")]


def parse_name_path(item: str) -> tuple[str, Path]:
    name, sep, path = item.partition("=")
    if not sep:
        raise SystemExit(f"expected name=path, got {item!r}")
    return name, Path(path)


def load_slice_texts(path: Path, max_docs: int, max_chars: int, max_doc_chars: int, batch_size: int) -> tuple[list[str], dict[str, int]]:
    texts: list[str] = []
    chars = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
        col = batch.column(0)
        for value in col:
            text = value.as_py()
            if not text:
                continue
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars]
            texts.append(text)
            chars += len(text)
            if len(texts) >= max_docs or chars >= max_chars:
                return texts, {"sampled_docs": len(texts), "sampled_chars": chars}
    return texts, {"sampled_docs": len(texts), "sampled_chars": chars}


def renyi_entropy_bits(counts: Counter[int], alpha: float = 2.5) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    s = sum((c / total) ** alpha for c in counts.values())
    if s <= 0:
        return 0.0
    return math.log2(s) / (1.0 - alpha)


def gini(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v) and v >= 0)
    if not vals:
        return 0.0
    n = len(vals)
    total = sum(vals)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * total) - (n + 1) / n


def token_has_distinctive_poly(token: str) -> bool:
    return has_distinctive_polytonic(token)


def evaluate(tokenizer_spec: dict[str, object], slice_name: str, texts: list[str], sample_info: dict[str, int], batch_size: int, word_batch_size: int) -> dict[str, object]:
    tok = AutoTokenizer.from_pretrained(str(tokenizer_spec["path"]), use_fast=True)
    vocab_size = len(tok)
    base_vocab = int(tokenizer_spec["base_vocab_size"])
    c3_base_vocab = int(tokenizer_spec["c3_base_vocab_size"])
    poly_added = int(tokenizer_spec["polytonic_added_count"])

    total_tokens = 0
    total_chars = 0
    total_bytes = 0
    greek_word_count = 0
    poly_word_count = 0
    greek_word_tokens = 0
    poly_word_tokens = 0
    single_token_greek_words = 0
    single_token_poly_words = 0
    poly_added_token_count = 0
    c3_added_token_count = 0
    byteish_token_count = 0
    polytonic_token_count = 0
    replacement_char_texts = 0
    exact_roundtrip = 0
    token_counts: Counter[int] = Counter()
    used_poly_added: set[int] = set()
    used_c3_added: set[int] = set()

    id_to_token = tok.convert_ids_to_tokens(list(range(vocab_size)))
    byteish_ids = {i for i, t in enumerate(id_to_token) if t and "�" in t}
    poly_token_ids = {i for i, t in enumerate(id_to_token) if t and token_has_distinctive_poly(t)}

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(batch, add_special_tokens=False).input_ids
        for text, ids in zip(batch, enc):
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            total_tokens += len(ids)
            token_counts.update(ids)
            if "�" in text:
                replacement_char_texts += 1
            decoded = tok.decode(ids, skip_special_tokens=False)
            if decoded == text:
                exact_roundtrip += 1
            for tid in ids:
                if tid >= c3_base_vocab:
                    poly_added_token_count += 1
                    used_poly_added.add(tid)
                elif tid >= base_vocab:
                    c3_added_token_count += 1
                    used_c3_added.add(tid)
                if tid in byteish_ids:
                    byteish_token_count += 1
                if tid in poly_token_ids:
                    polytonic_token_count += 1

    all_words: list[str] = []
    all_poly_words: list[str] = []
    for text in texts:
        words = greek_words(text)
        greek_word_count += len(words)
        all_words.extend(words)
        poly_words = [w for w in words if has_distinctive_polytonic(w)]
        poly_word_count += len(poly_words)
        all_poly_words.extend(poly_words)

    for start in range(0, len(all_words), word_batch_size):
        enc = tok(all_words[start : start + word_batch_size], add_special_tokens=False).input_ids
        for ids in enc:
            greek_word_tokens += len(ids)
            if len(ids) == 1:
                single_token_greek_words += 1
    for start in range(0, len(all_poly_words), word_batch_size):
        enc = tok(all_poly_words[start : start + word_batch_size], add_special_tokens=False).input_ids
        for ids in enc:
            poly_word_tokens += len(ids)
            if len(ids) == 1:
                single_token_poly_words += 1

    return {
        "variant_id": tokenizer_spec["name"],
        "tokenizer_path": str(tokenizer_spec["path"]),
        "tokenizer_sha256": tokenizer_spec.get("tokenizer_sha256"),
        "slice": slice_name,
        "base_vocab_size": base_vocab,
        "c3_base_vocab_size": c3_base_vocab,
        "vocab_size": vocab_size,
        "polytonic_added_count": poly_added,
        **sample_info,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_utf8_bytes": total_bytes,
        "greek_word_count": greek_word_count,
        "distinctive_polytonic_word_count": poly_word_count,
        "chars_per_token": (total_chars / total_tokens) if total_tokens else None,
        "bytes_per_token": (total_bytes / total_tokens) if total_tokens else None,
        "tokens_per_byte": (total_tokens / total_bytes) if total_bytes else None,
        "greek_word_fertility": (greek_word_tokens / greek_word_count) if greek_word_count else None,
        "distinctive_polytonic_word_fertility": (poly_word_tokens / poly_word_count) if poly_word_count else None,
        "single_token_greek_word_share": (single_token_greek_words / greek_word_count) if greek_word_count else None,
        "single_token_polytonic_word_share": (single_token_poly_words / poly_word_count) if poly_word_count else None,
        "poly_added_token_rate": (poly_added_token_count / total_tokens) if total_tokens else None,
        "c3_added_token_rate": (c3_added_token_count / total_tokens) if total_tokens else None,
        "poly_added_vocab_utilization_rate": (len(used_poly_added) / poly_added) if poly_added else 0.0,
        "poly_added_vocab_used": len(used_poly_added),
        "poly_added_vocab_unused": max(0, poly_added - len(used_poly_added)),
        "c3_added_vocab_used": len(used_c3_added),
        "byteish_token_rate": (byteish_token_count / total_tokens) if total_tokens else None,
        "polytonic_token_rate": (polytonic_token_count / total_tokens) if total_tokens else None,
        "replacement_char_doc_rate": (replacement_char_texts / len(texts)) if texts else None,
        "roundtrip_exact_doc_rate": (exact_roundtrip / len(texts)) if texts else None,
        "renyi_2_5_entropy_bits": renyi_entropy_bits(token_counts, alpha=2.5),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants-manifest", type=Path, required=True)
    parser.add_argument("--base-vocab-size", type=int, default=131072)
    parser.add_argument("--c3-base-vocab-size", type=int, default=148480)
    parser.add_argument("--slice", action="append", required=True, help="name=/path/to/parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-docs-per-slice", type=int, default=1000)
    parser.add_argument("--max-chars-per-slice", type=int, default=5_000_000)
    parser.add_argument("--max-doc-chars", type=int, default=120_000)
    parser.add_argument("--parquet-batch-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--word-batch-size", type=int, default=1024)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants_payload = json.loads(args.variants_manifest.read_text(encoding="utf-8"))
    tokenizer_specs = []
    for item in variants_payload["variants"]:
        tokenizer_specs.append(
            {
                "name": item["variant_id"],
                "path": item["variant_dir"],
                "tokenizer_sha256": item.get("tokenizer_sha256"),
                "base_vocab_size": args.base_vocab_size,
                "c3_base_vocab_size": args.c3_base_vocab_size,
                "polytonic_added_count": item["polytonic_added_count"],
            }
        )
    slices = [parse_name_path(s) for s in args.slice]

    samples = {}
    sample_manifest = {}
    for name, path in slices:
        texts, info = load_slice_texts(path, args.max_docs_per_slice, args.max_chars_per_slice, args.max_doc_chars, args.parquet_batch_size)
        samples[name] = texts
        sample_manifest[name] = {"path": str(path), **info}
    write_json(args.output_dir / "sample_manifest.json", sample_manifest)
    write_json(args.output_dir / "tokenizers.json", tokenizer_specs)

    rows = []
    for spec in tokenizer_specs:
        for slice_name, _path in slices:
            row = evaluate(spec, slice_name, samples[slice_name], sample_manifest[slice_name], args.batch_size, args.word_batch_size)
            rows.append(row)
            write_json(args.output_dir / "progress_latest.json", {"completed_rows": len(rows), "last": row})

    # Fairness-style gini over fertility for any explicitly supplied apertus55 slices.
    for spec in tokenizer_specs:
        vals = [
            r["greek_word_fertility"]
            for r in rows
            if r["variant_id"] == spec["name"] and str(r["slice"]).startswith("apertus55_") and r["greek_word_fertility"] is not None
        ]
        if vals:
            rows.append(
                {
                    "variant_id": spec["name"],
                    "tokenizer_path": str(spec["path"]),
                    "tokenizer_sha256": spec.get("tokenizer_sha256"),
                    "slice": "apertus55_fertility_gini",
                    "base_vocab_size": args.base_vocab_size,
                    "c3_base_vocab_size": args.c3_base_vocab_size,
                    "vocab_size": args.c3_base_vocab_size + int(spec["polytonic_added_count"]),
                    "polytonic_added_count": spec["polytonic_added_count"],
                    "sampled_docs": len(vals),
                    "sampled_chars": None,
                    "greek_word_fertility": gini(vals),
                }
            )

    write_json(args.output_dir / "metrics_by_slice.json", rows)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (args.output_dir / "metrics_by_slice.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

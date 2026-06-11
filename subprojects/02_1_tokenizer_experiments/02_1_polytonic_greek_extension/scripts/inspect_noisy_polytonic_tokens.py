#!/usr/bin/env python3
"""Inspect source distribution and contexts for selected added-token ids."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


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


def bytelevel_bytes(token: str) -> bytes:
    return bytes(BYTE_DECODER[ch] for ch in token)


def readable(token: str) -> str:
    return bytelevel_bytes(token).decode("utf-8", errors="backslashreplace")


def classify_bytes(raw: bytes) -> str:
    if not raw:
        return "empty"
    first = raw[0]
    if 0x80 <= first <= 0xBF:
        return "starts_with_utf8_continuation_byte"
    try:
        raw.decode("utf-8")
        return "valid_utf8_standalone"
    except UnicodeDecodeError:
        return "invalid_utf8_standalone"


def context(text: str, token_text: str, width: int) -> str | None:
    if token_text.startswith("\\x"):
        return None
    needle = token_text.strip()
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    start = max(0, idx - width)
    end = min(len(text), idx + len(needle) + width)
    return re.sub(r"\s+", " ", text[start:end])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--token-id", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-examples-per-token", type=int, default=12)
    parser.add_argument("--context-width", type=int, default=80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(args.tokenizer_json))
    payload = json.loads(args.tokenizer_json.read_text(encoding="utf-8"))
    id_to_token = {idx: token for token, idx in payload["model"]["vocab"].items()}
    targets = set(args.token_id)
    source_counts: dict[int, Counter[str]] = {tid: Counter() for tid in targets}
    total_counts: Counter[int] = Counter()
    examples: dict[int, list[dict[str, object]]] = defaultdict(list)
    token_info = {}
    for tid in sorted(targets):
        raw_token = id_to_token[tid]
        raw_bytes = bytelevel_bytes(raw_token)
        token_info[tid] = {
            "id": tid,
            "bytelevel_token": raw_token,
            "readable_or_escaped": readable(raw_token),
            "raw_bytes_hex": raw_bytes.hex(" "),
            "byte_class": classify_bytes(raw_bytes),
        }

    pf = pq.ParquetFile(args.input_parquet)
    cols = ["text"]
    for maybe in ("source_dataset", "source", "doc_id", "id", "title", "path", "url"):
        if maybe in pf.schema.names:
            cols.append(maybe)
    for batch in pf.iter_batches(columns=cols, batch_size=args.batch_size):
        rows = batch.to_pylist()
        texts = [r.get("text") or "" for r in rows]
        encs = tok.encode_batch(texts)
        for row, enc in zip(rows, encs):
            seen = Counter(i for i in enc.ids if i in targets)
            if not seen:
                continue
            src = str(row.get("source_dataset") or row.get("source") or "unknown")
            for tid, count in seen.items():
                total_counts[tid] += count
                source_counts[tid][src] += count
                if len(examples[tid]) < args.max_examples_per_token:
                    text = row.get("text") or ""
                    examples[tid].append(
                        {
                            "source_dataset": src,
                            "count_in_row": count,
                            "row_keys": {k: row.get(k) for k in ("doc_id", "id", "title", "path", "url") if k in row and row.get(k) is not None},
                            "context_guess": context(text, token_info[tid]["readable_or_escaped"], args.context_width),
                            "text_prefix": re.sub(r"\s+", " ", text[:300]),
                        }
                    )

    summary = []
    for tid in sorted(targets):
        row = dict(token_info[tid])
        row["total_count"] = int(total_counts[tid])
        row["source_counts"] = dict(source_counts[tid].most_common())
        row["examples"] = examples[tid]
        summary.append(row)

    (args.output_dir / "noisy_token_source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "noisy_token_source_counts.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "readable_or_escaped", "raw_bytes_hex", "byte_class", "source_dataset", "count"], delimiter="\t")
        writer.writeheader()
        for row in summary:
            for src, count in row["source_counts"].items():
                writer.writerow(
                    {
                        "id": row["id"],
                        "readable_or_escaped": row["readable_or_escaped"],
                        "raw_bytes_hex": row["raw_bytes_hex"],
                        "byte_class": row["byte_class"],
                        "source_dataset": src,
                        "count": count,
                    }
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

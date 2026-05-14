#!/usr/bin/env python3
"""Apply char_language_bitmask.parquet to the Apertus vocab.

For each token in the Apertus tokenizer vocab, decode its ByteLevel
bytes back to a UTF-8 string and compute:
  - bitmask_and: languages whose (script, encoding) admits EVERY decoded char
  - bitmask_or:  languages whose (script, encoding) admits AT LEAST ONE

Codepoint lookup follows the strict rule:
  - If in bitmask table → use stored bits.
  - Else if Unicode category is substrate (N*/P*/S*/Z*/Cc/Cf) → ALL_BITS
    (substrate contributes zero exclusion power).
  - Else (letter/mark in a script outside our scope) → 0 bits, which
    forces AND-rejection of every in-scope language for that token.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


GPT2_BYTELEVEL_MAP_OFFSET = 256

# Mirrors EXTRA_SUBSTRATE_CODEPOINTS in build_char_language_bitmask.py.
# Ll/Lo codepoints that function as language-neutral typography.
EXTRA_SUBSTRATE_CODEPOINTS: frozenset[int] = frozenset({
    0x00AA,  # FEMININE ORDINAL INDICATOR
    0x00B5,  # MICRO SIGN
    0x00BA,  # MASCULINE ORDINAL INDICATOR
})


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(GPT2_BYTELEVEL_MAP_OFFSET + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def load_char_bitmask(parquet_path: Path) -> dict[int, int]:
    table = pq.read_table(parquet_path, columns=["codepoint", "bitmask"])
    cps = table.column("codepoint").to_pylist()
    masks = table.column("bitmask").to_pylist()
    return dict(zip(cps, masks))


def load_apertus_vocab(snapshot_dir: Path) -> list[tuple[int, str]]:
    tjson = snapshot_dir / "tokenizer.json"
    if tjson.exists():
        spec = json.loads(tjson.read_text())
        vocab = spec["model"]["vocab"]
        return [(idx, tok) for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1])]
    vjson = snapshot_dir / "vocab.json"
    if vjson.exists():
        vocab = json.loads(vjson.read_text())
        return [(idx, tok) for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1])]
    sys.exit(f"no tokenizer.json or vocab.json under {snapshot_dir}")


def special_token_set(snapshot_dir: Path) -> set[str]:
    tjson = snapshot_dir / "tokenizer.json"
    if not tjson.exists():
        return set()
    spec = json.loads(tjson.read_text())
    return {t["content"] for t in spec.get("added_tokens", [])}


def token_string_to_bytes(token: str, u2b: dict[str, int]) -> bytes | None:
    out = bytearray()
    for ch in token:
        b = u2b.get(ch)
        if b is None:
            return None
        out.append(b)
    return bytes(out)


def codepoint_bits(cp: int, table: dict[int, int], all_bits: int) -> tuple[int, bool]:
    """Returns (bits, in_scope).
    in_scope = True iff the codepoint contributed exclusion power, i.e.
    came from either CLDR-derived membership or the substrate override.
    A letter outside our scope returns (0, False).
    """
    m = table.get(cp)
    if m is not None:
        return m, True
    if cp in EXTRA_SUBSTRATE_CODEPOINTS:
        return all_bits, True
    try:
        cat = unicodedata.category(chr(cp))
    except ValueError:
        return 0, False
    if cat == "Lm" or cat[0] in ("N", "P", "S", "Z") or cat in ("Cc", "Cf"):
        return all_bits, True
    return 0, False


def compute_token_row(
    token_id: int,
    token: str,
    specials: set[str],
    u2b: dict[str, int],
    cp_mask: dict[int, int],
    all_bits: int,
) -> dict:
    if token in specials:
        return _row(token_id, None, None, 0, 0, 0, "special")

    raw = token_string_to_bytes(token, u2b)
    if raw is None:
        return _row(token_id, None, None, 0, 0, 0, "byte_unmapped")

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _row(token_id, raw, None, 0, 0, 0, "partial_utf8")

    and_mask = all_bits
    or_mask = 0
    n = 0
    any_known = False
    any_letter_unmodeled = False
    for ch in decoded:
        n += 1
        m, in_scope = codepoint_bits(ord(ch), cp_mask, all_bits)
        and_mask &= m
        or_mask |= m
        if in_scope:
            any_known = True
        else:
            any_letter_unmodeled = True

    if not any_known:
        status = "no_in_scope_chars"
    elif any_letter_unmodeled:
        status = "text_with_unmodeled_letters"
    else:
        status = "text"

    return _row(token_id, raw, decoded, and_mask, or_mask, n, status)


def _row(
    token_id: int,
    token_bytes: bytes | None,
    decoded_text: str | None,
    bitmask_and: int,
    bitmask_or: int,
    num_chars: int,
    status: str,
) -> dict:
    return {
        "token_id": token_id,
        "token_bytes": token_bytes,
        "decoded_text": decoded_text,
        "bitmask_and": bitmask_and,
        "bitmask_or": bitmask_or,
        "num_chars": num_chars,
        "status": status,
    }


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--char-bitmask",
        type=Path,
        default=here / "artifacts" / "char_language_bitmask.parquet",
    )
    ap.add_argument(
        "--apertus-snapshot",
        type=Path,
        required=True,
        help="Path to the Apertus tokenizer snapshot directory.",
    )
    ap.add_argument(
        "--languages",
        type=Path,
        default=here / "languages.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "artifacts" / "token_language_bitmask.parquet",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=here / "artifacts" / "token_manifest.json",
    )
    ap.add_argument(
        "--char-manifest",
        type=Path,
        default=here / "artifacts" / "manifest.json",
    )
    args = ap.parse_args()

    import yaml

    spec = yaml.safe_load(args.languages.read_text())
    num_langs = len(spec["languages"])
    all_bits = (1 << num_langs) - 1

    print(f"loading char bitmask from {args.char_bitmask}")
    cp_mask = load_char_bitmask(args.char_bitmask)
    print(f"  {len(cp_mask)} codepoints loaded ({num_langs} bits)")

    print(f"loading Apertus vocab from {args.apertus_snapshot}")
    vocab = load_apertus_vocab(args.apertus_snapshot)
    specials = special_token_set(args.apertus_snapshot)
    print(f"  vocab size {len(vocab)}, {len(specials)} special tokens")

    u2b = unicode_to_bytes()

    rows = []
    for token_id, token in vocab:
        rows.append(
            compute_token_row(
                token_id, token, specials, u2b, cp_mask, all_bits
            )
        )

    table = pa.table(
        {
            "token_id": pa.array([r["token_id"] for r in rows], type=pa.uint32()),
            "token_bytes": pa.array(
                [r["token_bytes"] for r in rows], type=pa.binary()
            ),
            "decoded_text": pa.array(
                [r["decoded_text"] for r in rows], type=pa.string()
            ),
            "bitmask_and": pa.array(
                [r["bitmask_and"] for r in rows], type=pa.uint64()
            ),
            "bitmask_or": pa.array(
                [r["bitmask_or"] for r in rows], type=pa.uint64()
            ),
            "num_chars": pa.array([r["num_chars"] for r in rows], type=pa.uint16()),
            "status": pa.array([r["status"] for r in rows], type=pa.string()),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"wrote {args.out}")

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("by status:")
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v}")

    and_popcount = collections.Counter(
        bin(r["bitmask_and"]).count("1")
        for r in rows
        if r["status"] == "text"
    )

    snapshot = args.apertus_snapshot.resolve()
    snapshot_sha = snapshot.name if "snapshots" in str(snapshot) else None

    char_manifest_data = {}
    char_build_ts = None
    char_schema_version = None
    if args.char_manifest.exists():
        char_manifest_data = json.loads(args.char_manifest.read_text())
        char_build_ts = char_manifest_data.get("build_timestamp_utc")
        char_schema_version = char_manifest_data.get("schema_version")

    manifest = {
        "apply_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "apertus_snapshot_path": str(snapshot),
        "apertus_snapshot_revision_sha": snapshot_sha,
        "vocab_size": len(vocab),
        "num_special_tokens": len(specials),
        "char_bitmask_path": str(args.char_bitmask.resolve()),
        "char_bitmask_build_timestamp_utc": char_build_ts,
        "char_bitmask_schema_version": char_schema_version,
        "num_language_bits": num_langs,
        "status_counts": by_status,
        "text_token_and_popcount": {
            str(k): v for k, v in sorted(and_popcount.items())
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()

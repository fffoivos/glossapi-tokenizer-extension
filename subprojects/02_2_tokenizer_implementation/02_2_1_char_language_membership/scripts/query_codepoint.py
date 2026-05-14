#!/usr/bin/env python3
"""Substrate-aware codepoint → bitmask lookup helper.

`char_language_bitmask.parquet` is a **sparse table**: it stores
codepoints that received explicit language evidence (CLDR + closures
+ script-range fallback) plus the substrate codepoints we
explicitly seeded (ASCII printable, the small supplementary list,
EXTRA_SUBSTRATE_CODEPOINTS). Most Unicode substrate (emoji,
exotic punctuation, supplementary-plane symbols, …) is NOT in the
table. A direct consumer that does `table[cp]` and gets 0 will
false-reject those codepoints.

This helper reproduces the same fallback rule that
`apply_to_apertus_vocab.py` uses at apply time, so any consumer
gets identical semantics:

    >>> from query_codepoint import load, codepoint_bits
    >>> table, all_bits = load(
    ...     "artifacts/char_language_bitmask.parquet",
    ...     "languages.yaml",
    ... )
    >>> bin(codepoint_bits(0x4E2D, table, all_bits)).count("1")
    3                                          # `中` — zh-Hans/zh-Hant/ja
    >>> codepoint_bits(0x1F389, table, all_bits) == all_bits
    True                                       # `🎉` substrate via So
    >>> codepoint_bits(0x13A0, table, all_bits)
    0                                          # Cherokee Ꭰ — not modeled

Bit layout: see `languages.yaml`. Bit positions are stable wire
identifiers; never reused.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from _common import (
    BITMASK_BYTES,
    EXTRA_SUBSTRATE_CODEPOINTS,
    codepoint_bits as _codepoint_bits,
    compute_all_bits,
    decode_mask,
)


def load(
    parquet_path: str | Path,
    languages_yaml_path: str | Path,
) -> tuple[dict[int, int], int]:
    """Returns (table, all_bits) for the language layer (back-compat
    with the v2.2 signature). `table` maps codepoint → language bitmask
    int. Missing codepoints are handled by `codepoint_bits` below."""
    t = pq.read_table(str(parquet_path), columns=["codepoint", "bitmask"])
    cps = t["codepoint"].to_pylist()
    masks = t["bitmask"].to_pylist()
    table = {cp: decode_mask(m) for cp, m in zip(cps, masks)}

    spec = yaml.safe_load(Path(languages_yaml_path).read_text())
    all_bits = compute_all_bits(spec["languages"])
    return table, all_bits


def load_all(
    parquet_path: str | Path,
    languages_yaml_path: str | Path,
    families_yaml_path: str | Path,
    scripts_yaml_path: str | Path,
) -> dict:
    """Load all three per-codepoint tables and their ALL_BITS values.

    Returns a dict with keys (the convention used by
    `token_bits_and_three` and `token_bits_or_three` below):
      `table_script` / `table_family` / `table_lang` (dict cp → int)
      `all_script` / `all_family` / `all_lang` (int)
      `scripts` / `families` / `languages` (yaml-loaded lists)
    """
    t = pq.read_table(
        str(parquet_path),
        columns=["codepoint", "script_bits", "family_bits", "bitmask"],
    )
    cps = t["codepoint"].to_pylist()
    s = t["script_bits"].to_pylist()
    f = t["family_bits"].to_pylist()
    l = t["bitmask"].to_pylist()
    table_script = {cp: decode_mask(b) for cp, b in zip(cps, s)}
    table_family = {cp: decode_mask(b) for cp, b in zip(cps, f)}
    table_lang = {cp: decode_mask(b) for cp, b in zip(cps, l)}

    languages = yaml.safe_load(Path(languages_yaml_path).read_text())["languages"]
    families = yaml.safe_load(Path(families_yaml_path).read_text())["families"]
    scripts = yaml.safe_load(Path(scripts_yaml_path).read_text())["scripts"]

    return {
        "table_script": table_script,
        "table_family": table_family,
        "table_lang": table_lang,
        "all_script": compute_all_bits(scripts),
        "all_family": compute_all_bits(families),
        "all_lang": compute_all_bits(languages),
        "scripts": scripts,
        "families": families,
        "languages": languages,
    }


def codepoint_bits(cp: int, table: dict[int, int], all_bits: int) -> int:
    """Substrate-aware lookup. Thin wrapper around `_common.codepoint_bits`
    that drops the `in_scope` flag (callers of this module only want
    the mask). See _common.codepoint_bits for the contract."""
    m, _ = _codepoint_bits(cp, table, all_bits)
    return m


def token_bits_and(
    text: str, table: dict[int, int], all_bits: int
) -> int:
    """AND across the codepoint bitmasks of `text`. Empty string
    returns `all_bits` (no constraint)."""
    out = all_bits
    for ch in text:
        out &= codepoint_bits(ord(ch), table, all_bits)
        if out == 0:
            break
    return out


def token_bits_or(
    text: str, table: dict[int, int], all_bits: int
) -> int:
    """OR across the codepoint bitmasks of `text`."""
    out = 0
    for ch in text:
        out |= codepoint_bits(ord(ch), table, all_bits)
    return out


def token_bits_and_three(
    text: str, loaded: dict
) -> tuple[int, int, int]:
    """Three-layer AND aggregation across `text`.

    Takes the dict returned by `load_all()`. Returns
    `(script_and, family_and, language_and)` — each is the AND of
    its layer's codepoint bits across every char in `text`,
    applying the substrate-aware fallback for codepoints not in
    the stored tables. An empty `text` returns all-bits at every
    layer (no constraint).
    """
    s_and = loaded["all_script"]
    f_and = loaded["all_family"]
    l_and = loaded["all_lang"]
    for ch in text:
        cp = ord(ch)
        s_and &= codepoint_bits(cp, loaded["table_script"], loaded["all_script"])
        f_and &= codepoint_bits(cp, loaded["table_family"], loaded["all_family"])
        l_and &= codepoint_bits(cp, loaded["table_lang"], loaded["all_lang"])
        if s_and == 0 and f_and == 0 and l_and == 0:
            break
    return s_and, f_and, l_and


def token_bits_or_three(
    text: str, loaded: dict
) -> tuple[int, int, int]:
    """Three-layer OR aggregation across `text`. Same loaded-dict
    convention as `token_bits_and_three`."""
    s_or = f_or = l_or = 0
    for ch in text:
        cp = ord(ch)
        s_or |= codepoint_bits(cp, loaded["table_script"], loaded["all_script"])
        f_or |= codepoint_bits(cp, loaded["table_family"], loaded["all_family"])
        l_or |= codepoint_bits(cp, loaded["table_lang"], loaded["all_lang"])
    return s_or, f_or, l_or


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--char-bitmask",
        type=Path,
        default=here / "artifacts" / "char_language_bitmask.parquet",
    )
    ap.add_argument(
        "--languages",
        type=Path,
        default=here / "languages.yaml",
    )
    ap.add_argument(
        "codepoints",
        nargs="+",
        help="Codepoints to query. Accept 'U+XXXX', '0xXXXX', or a single char.",
    )
    args = ap.parse_args()

    table, all_bits = load(args.char_bitmask, args.languages)
    spec = yaml.safe_load(args.languages.read_text())
    bit_to_code = {L["bit"]: L["code"] for L in spec["languages"]}

    for token in args.codepoints:
        if token.startswith(("U+", "u+")):
            cp = int(token[2:], 16)
        elif token.startswith(("0x", "0X")):
            cp = int(token, 16)
        elif len(token) == 1:
            cp = ord(token)
        else:
            cp = int(token)
        m = codepoint_bits(cp, table, all_bits)
        bits = [bit_to_code[b] for b in sorted(bit_to_code) if m & (1 << b)]
        ch = chr(cp) if cp < 0x110000 else "?"
        print(
            f"U+{cp:04X} {ch!r:5s}  popcount={bin(m).count('1'):3d}  "
            f"bits={bits if len(bits) < 30 else f'[{len(bits)} langs]'}"
        )

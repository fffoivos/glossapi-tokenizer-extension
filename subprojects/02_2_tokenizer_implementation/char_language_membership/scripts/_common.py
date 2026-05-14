"""Shared constants and helpers for the char_language_membership scripts.

Single source of truth for:
  - BITMASK_BYTES (the fixed-width binary mask size)
  - EXTRA_SUBSTRATE_CODEPOINTS (Ll/Lo codepoints treated as substrate)
  - encode_mask / decode_mask (binary ↔ int)
  - compute_all_bits (OR over the configured language bits)
  - codepoint_bits (substrate-aware lookup mirroring the build rule)

Imported by build_char_language_bitmask.py, apply_to_apertus_vocab.py,
query_codepoint.py, and validate.py.
"""

from __future__ import annotations

import unicodedata


# Bitmask storage: fixed-width little-endian binary in Parquet.
# 16 bytes = 128-bit budget; today's scope uses 55 bits, leaving plenty
# of headroom for audit-driven additions (Kazakh, Pashto, Khmer, etc.).
BITMASK_BYTES: int = 16
BITMASK_MAX_BIT: int = BITMASK_BYTES * 8


# Ll/Lo codepoints categorised as letters by Unicode but functioning as
# language-neutral typography. Treated as substrate (ALL_BITS) so they
# contribute zero exclusion power.
EXTRA_SUBSTRATE_CODEPOINTS: frozenset[int] = frozenset(
    {
        0x00AA,  # FEMININE ORDINAL INDICATOR
        0x00B5,  # MICRO SIGN — distinct from Greek mu U+03BC
        0x00BA,  # MASCULINE ORDINAL INDICATOR
    }
    | set(range(0xFF10, 0xFF1A))  # Fullwidth digits 0–9
    | set(range(0xFF21, 0xFF3B))  # Fullwidth Latin A–Z
    | set(range(0xFF41, 0xFF5B))  # Fullwidth Latin a–z
)


def encode_mask(m: int) -> bytes:
    return m.to_bytes(BITMASK_BYTES, "little")


def decode_mask(b: bytes) -> int:
    return int.from_bytes(b, "little")


def compute_all_bits(languages: list[dict]) -> int:
    """OR one bit per language entry. Robust to non-dense bit ids
    (which the docs already promise — bits are stable wire identifiers,
    never reused).

    Raises if bit ids are duplicated or out of [0, BITMASK_MAX_BIT).
    """
    bits = [L["bit"] for L in languages]
    if len(set(bits)) != len(bits):
        raise RuntimeError(f"duplicate bit ids in languages: {bits}")
    if any(b < 0 or b >= BITMASK_MAX_BIT for b in bits):
        bad = [b for b in bits if b < 0 or b >= BITMASK_MAX_BIT]
        raise RuntimeError(
            f"bit id(s) out of [0, {BITMASK_MAX_BIT}): {bad}"
        )
    out = 0
    for b in bits:
        out |= 1 << b
    return out


def codepoint_bits(
    cp: int, table: dict[int, int], all_bits: int
) -> tuple[int, bool]:
    """Substrate-aware codepoint lookup. Returns (bits, in_scope).

      - If `cp` is in the table, returns the stored mask (in_scope=True).
      - Else if substrate (Unicode category N*/P*/S*/Z*/Cc/Cf/Lm OR in
        EXTRA_SUBSTRATE_CODEPOINTS), returns (all_bits, True).
      - Else (letter/mark in a script we don't model): returns (0, False),
        which AND-rejects every in-scope language.
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

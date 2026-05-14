"""Shared constants and helpers for the 02_2_1_char_language_membership scripts.

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


def compute_all_bits(entries: list[dict]) -> int:
    """OR one bit per entry's `bit` field. Robust to non-dense bit
    ids (bits are stable wire identifiers, never reused).

    Works for languages, families, and scripts — anything with a
    `bit` field. Raises on duplicate or out-of-range bit ids.
    """
    bits = [e["bit"] for e in entries]
    if len(set(bits)) != len(bits):
        raise RuntimeError(f"duplicate bit ids: {bits}")
    if any(b < 0 or b >= BITMASK_MAX_BIT for b in bits):
        bad = [b for b in bits if b < 0 or b >= BITMASK_MAX_BIT]
        raise RuntimeError(
            f"bit id(s) out of [0, {BITMASK_MAX_BIT}): {bad}"
        )
    out = 0
    for b in bits:
        out |= 1 << b
    return out


def derive_family_bits(
    lang_bits: int,
    families: list[dict],
    lang_code_to_bit: dict[str, int],
) -> int:
    """Project a language bitmask onto the family layer.

    A family bit is set iff the input has the language bit of any
    locale in that family's `locales` list. Pure derivation — no
    new evidence introduced.
    """
    out = 0
    for f in families:
        fam_lang_mask = 0
        for loc in f["locales"]:
            fam_lang_mask |= 1 << lang_code_to_bit[loc]
        if lang_bits & fam_lang_mask:
            out |= 1 << f["bit"]
    return out


def build_script_to_lang_mask(
    scripts: list[dict],
    families: list[dict],
    lang_code_to_bit: dict[str, int],
) -> dict[str, int]:
    """Pre-compute the script-`code` → language-bitmask mapping.

    For each script entry in scripts.yaml, the value is the OR of
    every language bit belonging to any family assigned to that
    script. Used at derive-script-bits time as a fast lookup so the
    script-bits derivation is just one bitwise AND per script.
    """
    script_to_lang_mask: dict[str, int] = {s["code"]: 0 for s in scripts}
    for f in families:
        sc = f["script"]
        mask = 0
        for loc in f["locales"]:
            mask |= 1 << lang_code_to_bit[loc]
        script_to_lang_mask[sc] = script_to_lang_mask.get(sc, 0) | mask
    return script_to_lang_mask


def derive_script_bits(
    lang_bits: int,
    scripts: list[dict],
    script_to_lang_mask: dict[str, int],
) -> int:
    """Project a language bitmask onto the script layer.

    A script bit is set iff any locale belonging to a family
    declared under this script has its language bit set. Pure
    projection — no new evidence introduced.

    NOTE: we deliberately do *not* also set a script bit when the
    codepoint's Unicode script matches the script's iso15924 code.
    Two of our scripts (Grek-modern, Grek-polyton) share
    iso15924=Grek; under that rule a polytonic-only codepoint
    would get both Greek bits, collapsing the encoding distinction.
    Language-bit projection alone preserves the modern / polytonic
    split correctly: ἀ (U+1F00) has only el-polyton language bit →
    only Grek-polyton script bit; α (U+03B1) has both el and
    el-polyton language bits → both Greek script bits.
    """
    out = 0
    for s in scripts:
        if lang_bits & script_to_lang_mask.get(s["code"], 0):
            out |= 1 << s["bit"]
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

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
# 16 bytes = 128-bit budget; current scope (as of v3.3.3) uses 88
# language bits / 47 family bits / 29 script bits, leaving plenty of
# headroom for future audit-driven additions.
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


def build_iso_lookup(
    languages: list[dict],
    scripts: list[dict],
) -> dict[tuple[str, str], str]:
    """Build a `(iso_639_3, iso_15924) → char_tool_code` lookup table.

    Resolves consumer canonical keys like `eng_Latn`, `srp_Cyrl`,
    `cmn_Hani`, `est_Latn`, `lvs_Latn`, `ekk_Latn` to the char-tool's
    BCP 47 code at bit-level. Built from `languages.yaml` + `scripts.yaml`
    primary fields and the documented aliases:

      - `iso_639_3` primary + `iso_639_3_aliases` (e.g., est+ekk,
        lav+lvs, zho+chi+cmn, mon+khk).
      - `iso15924` primary + `iso15924_aliases` (e.g., Hans+Hani as
        the FineWeb-2 default; Jpan+Hira+Kana+Hrkt for Japanese).

    The lookup is many-to-one: every alias resolves to a single
    char-tool code. When multiple char-tool codes could resolve (e.g.
    `zho_Hani` could match both `zh-Hans` and `zh-Hant`), the first
    matching script in `iso15924` (primary) wins — for `Hani`, that's
    `zh-Hans` (the documented Simplified-as-default rule).
    """
    # script-code → (iso15924 primary, set of aliases)
    script_iso_to_code: dict[str, str] = {}
    for s in scripts:
        primary = s["iso15924"]
        # primary always points to this script
        script_iso_to_code.setdefault(primary, s["code"])
        for alias in s.get("iso15924_aliases", []) or []:
            script_iso_to_code.setdefault(alias, s["code"])

    out: dict[tuple[str, str], str] = {}
    for L in languages:
        char_tool_code = L["code"]
        primary_iso = L["iso_639_3"]
        aliases = L.get("iso_639_3_aliases", []) or []
        # All ISO 639-3 codes that resolve to this language
        all_lang_iso = [primary_iso] + list(aliases)
        # The script for this language. Match scripts by EITHER `code`
        # OR `iso15924` — most language entries use the iso15924
        # value directly (e.g. ru's `script: Cyrl` matches scripts.yaml's
        # `code: Cyrl, iso15924: Cyrl`), but Greek is an exception
        # because there are two scripts (Grek-modern, Grek-polyton) that
        # both have `iso15924: Grek` while their `code` differs.
        # Tolerant matching here ensures `el` / `el-polyton` (both with
        # `script: Grek`) pick up entries for `(ell, Grek)` and
        # `(gre, Grek)`. Tie-break is iteration order in languages.yaml
        # (el wins for `ell_Grek` because it's declared first).
        lang_script = L["script"]
        compatible_scripts: list[str] = []
        seen_iso: set[str] = set()
        for s in scripts:
            if s["code"] == lang_script or s["iso15924"] == lang_script:
                primary_iso15924 = s["iso15924"]
                if primary_iso15924 not in seen_iso:
                    compatible_scripts.append(primary_iso15924)
                    seen_iso.add(primary_iso15924)
                for a in s.get("iso15924_aliases", []) or []:
                    if a not in seen_iso:
                        compatible_scripts.append(a)
                        seen_iso.add(a)
        for liso in all_lang_iso:
            for sc in compatible_scripts:
                key = (liso, sc)
                # First write wins (handles Hans-default-for-Hani semantics
                # because zh-Hans is declared before zh-Hant in scripts.yaml,
                # and `Hani` is in zh-Hans's aliases not zh-Hant's; same
                # mechanism makes el win over el-polyton for `ell_Grek`).
                out.setdefault(key, char_tool_code)
    return out


def code_from_canonical_key(
    canonical_key: str,
    iso_lookup: dict[tuple[str, str], str],
) -> str | None:
    """Resolve a `<iso_639_3>_<iso_15924>` canonical key (FineWeb-2 /
    ISO style) to the char-tool's BCP 47 code, or None if the key
    references a language not in scope.

    Examples:
      `eng_Latn` → `en`
      `srp_Cyrl` → `sr-Cyrl`
      `cmn_Hani` → `zh-Hans` (default per Hans/Hani alias rule)
      `ekk_Latn` → `et` (macrolanguage-individual alias)
      `lvs_Latn` → `lv`
      `swa_Latn` → `sw` (Swahili, added v3.2)
      `gmh_Latn` → None  (Middle High German — no CLDR data; consumer
                          should fall back to script-only resolution)
      `khk_Cyrl` → `mn`
    """
    if "_" not in canonical_key:
        return None
    iso_lang, _, iso_script = canonical_key.partition("_")
    return iso_lookup.get((iso_lang, iso_script))


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

#!/usr/bin/env python3
"""Post-build sanity checks (v2).

Exits non-zero on first failure. Two phases:

  Phase 1 — char-level spot checks against char_language_bitmask.parquet
    - v1 spot checks (universal Latin letters, language-specific
      diacritics, Greek, Cyrillic, CJK, Arabic).
    - Case closure: Ά/Έ/Ώ/Ç/Ñ uppercase variants inherit lowercase bits.
    - NFD closure: combining acute U+0301 ends up in every locale
      whose alphabet contains an acute-accented letter; combining
      ypogegrammeni U+0345 inherits el-polyton via post-fallback NFD.
    - Script-range fallback: rare Han / Greek codepoints not in CLDR's
      curated exemplar still get the right bits. Cyrillic and Arabic
      do NOT fall back (deliberate, per strict rule); rare Cyrillic
      letters used by out-of-scope languages must have 0 bits.
    - Substrate: digits, punctuation, whitespace, code/math symbols
      all have ALL_BITS set; µ/º/ª are substrate too.
    - New v2 locales: ko/hi/he/th/hy/ka/Indic/my/ur + sister locales.

  Phase 2 — token-level audit gate against token_language_bitmask.parquet
    - Groups fall-through tokens (status text_with_unmodeled_letters
      and no_in_scope_chars) by Unicode script of their letter/mark
      content, and asserts <50 per out-of-scope script.
"""

from __future__ import annotations

import collections
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from _common import (
    BITMASK_BYTES,
    BITMASK_MAX_BIT,
    build_script_to_lang_mask,
    codepoint_bits as _cp_bits,
    compute_all_bits,
    decode_mask,
    derive_family_bits,
    derive_script_bits,
)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    spec = yaml.safe_load((here / "languages.yaml").read_text())
    bit_for = {L["code"]: L["bit"] for L in spec["languages"]}

    # Dense / wire-format sanity
    bits_used = [L["bit"] for L in spec["languages"]]
    assert len(set(bits_used)) == len(bits_used), \
        f"duplicate bit ids in languages.yaml: {bits_used}"
    assert all(0 <= b < BITMASK_MAX_BIT for b in bits_used), \
        f"bit ids must be in [0, {BITMASK_MAX_BIT}); got {bits_used}"

    all_bits = compute_all_bits(spec["languages"])

    families = yaml.safe_load((here / "families.yaml").read_text())["families"]
    scripts = yaml.safe_load((here / "scripts.yaml").read_text())["scripts"]
    family_bit = {f["code"]: f["bit"] for f in families}
    script_bit = {s["code"]: s["bit"] for s in scripts}
    all_family = compute_all_bits(families)
    all_script = compute_all_bits(scripts)
    lang_code_to_bit = {L["code"]: L["bit"] for L in spec["languages"]}
    script_to_lang_mask = build_script_to_lang_mask(
        scripts, families, lang_code_to_bit
    )

    t = pq.read_table(here / "artifacts" / "char_language_bitmask.parquet")
    expected_type = pa.binary(BITMASK_BYTES)
    for col in ("bitmask", "family_bits", "script_bits"):
        if not t.schema.field(col).type.equals(expected_type):
            print(
                f"FAIL — char `{col}` column type is "
                f"{t.schema.field(col).type}, expected {expected_type}"
            )
            sys.exit(1)
    by_cp = {
        cp: decode_mask(m)
        for cp, m in zip(
            t["codepoint"].to_pylist(), t["bitmask"].to_pylist()
        )
    }
    by_cp_family = {
        cp: decode_mask(m)
        for cp, m in zip(
            t["codepoint"].to_pylist(), t["family_bits"].to_pylist()
        )
    }
    by_cp_script = {
        cp: decode_mask(m)
        for cp, m in zip(
            t["codepoint"].to_pylist(), t["script_bits"].to_pylist()
        )
    }

    failures: list[str] = []

    def check_eq(cp: int, expected_set: set[str], note: str) -> None:
        m = by_cp.get(cp, 0)
        actual = {code for code, bit in bit_for.items() if m & (1 << bit)}
        if actual != expected_set:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} ({note}): "
                f"expected={sorted(expected_set)} actual={sorted(actual)}"
            )

    def check_superset(cp: int, required_subset: set[str], note: str) -> None:
        m = by_cp.get(cp, 0)
        actual = {code for code, bit in bit_for.items() if m & (1 << bit)}
        missing = required_subset - actual
        if missing:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} ({note}): "
                f"missing required bits={sorted(missing)}; actual={sorted(actual)}"
            )

    def check_all_bits(cp: int, note: str) -> None:
        m = by_cp.get(cp, 0)
        if m != all_bits:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} ({note}): "
                f"expected ALL_BITS={hex(all_bits)} actual={hex(m)}"
            )

    LATN_V1 = {
        "en", "cs", "da", "de", "es", "fr", "hu", "id",
        "it", "nl", "pl", "pt", "sv", "tr", "vi",
    }
    LATN_V2_EXTRA = {
        "ro", "sr-Latn", "az", "fi", "nb", "sl", "hr", "sk", "et",
        "lt", "lv", "ca", "is",
    }
    LATN_V3_1_EXTRA = {"eo"}
    LATN_ALL = LATN_V1 | LATN_V2_EXTRA | LATN_V3_1_EXTRA
    CYRL_V1 = {"ru", "uk", "bg", "mk", "sr-Cyrl"}
    CYRL_V3_1 = {"be", "kk", "mn"}
    CYRL_ALL = CYRL_V1 | CYRL_V3_1
    ARAB_ALL = {"ar", "fa", "ur", "ps"}

    # --- v1 spot checks (still must pass) ---
    check_eq(ord("a"), LATN_ALL, "'a' — every Latin locale")
    check_eq(ord("ł"), {"pl"}, "Polish-only ł")
    check_eq(ord("ñ"), {"es"}, "Spanish-only ñ")
    check_eq(0x03B1, {"el", "el-polyton"}, "Greek α — modern + polytonic")
    check_eq(0x1F00, {"el-polyton"}, "Polytonic ἀ — ONLY polytonic")
    check_eq(0x0627, ARAB_ALL, "Arabic alef — ar + fa + ur + ps")
    check_eq(0x4E2D, {"zh-Hans", "zh-Hant", "ja"}, "中 — in all 3 CJK CLDR sets")
    # v3.1 refined Han fallback — Hans/Hant precision preserved:
    check_eq(0x56FD, {"zh-Hans", "ja"}, "国 — Simplified + Joyo, NOT Hant")
    check_eq(0x570B, {"zh-Hant"}, "國 — Traditional only, NOT Hans, NOT Joyo")
    check_eq(0x6447, {"zh-Hans", "zh-Hant", "ja"}, "摇 — rare Han, fallback to all 3")
    check_eq(0x3042, {"ja"}, "Japanese hiragana あ")

    # --- v2: case closure ---
    # Lowercase already covered by CLDR exemplar; uppercase must inherit.
    check_superset(0x0386, {"el"}, "Ά — Greek capital A with tonos (case closure)")
    check_superset(0x0388, {"el"}, "Έ — Greek capital E with tonos (case closure)")
    check_superset(0x038F, {"el"}, "Ώ — Greek capital Omega with tonos")
    check_superset(0x00C9, {"cs", "es", "fr", "hu", "it", "nl", "pt", "sv"},
                   "É — uppercase of é")
    check_superset(0x00C7, {"fr", "pt", "tr"}, "Ç — uppercase of ç")
    check_superset(0x00D1, {"es"}, "Ñ — uppercase of ñ")
    check_superset(0x1EDC, {"vi"}, "Ờ — Vietnamese uppercase O+circumflex+grave")
    check_superset(0x0141, {"pl"}, "Ł — Polish uppercase")
    check_superset(0x0410, {"ru", "uk", "bg", "mk", "sr-Cyrl"}, "Russian А capital")

    # --- v2: NFD closure ---
    # The combining acute U+0301 appears in every locale whose alphabet
    # has acute-accented letters that NFD-decompose to base + 0301.
    check_superset(
        0x0301,
        {"cs", "es", "fr", "hu", "it", "nl", "pt"},
        "U+0301 combining acute — via NFD of á/é/í/ó/ú etc.",
    )

    # --- v2: script-range fallback ---
    # 摇 (U+6447) is a Han codepoint not in CLDR's curated zh-Hans
    # exemplar. Script-range fallback must give it the CJK bits.
    check_superset(
        0x6447,
        {"zh-Hans", "zh-Hant", "ja"},
        "摇 — Han codepoint outside CLDR exemplar (script-range fallback)",
    )
    check_superset(
        0x5220,
        {"zh-Hans", "zh-Hant", "ja"},
        "删 — same",
    )

    # --- v2: substrate = all bits ---
    check_all_bits(ord("5"), "digit '5'")
    check_all_bits(ord("."), "period")
    check_all_bits(ord("-"), "hyphen")
    check_all_bits(ord(" "), "ASCII space")
    check_all_bits(0x000A, "newline")
    check_all_bits(ord("="), "equals — was zh-Hans-only in v1")
    check_all_bits(ord("_"), "underscore — was CJK-only in v1")
    check_all_bits(ord("{"), "left brace")
    check_all_bits(ord("}"), "right brace")
    check_all_bits(ord("$"), "dollar — no CLDR locale in v1")

    # --- v2: new locale spot checks ---
    check_eq(0xB2E4, {"ko"}, "Korean Hangul 다")
    check_eq(0x0915, {"hi"}, "Hindi Devanagari क")
    check_eq(0x05E9, {"he"}, "Hebrew shin ש")
    check_eq(0x0E1E, {"th"}, "Thai phor phan พ")
    check_eq(0x0561, {"hy"}, "Armenian small ա")
    check_eq(0x10D0, {"ka"}, "Georgian ა")
    check_eq(0x0995, {"bn"}, "Bengali ক")
    check_eq(0x0B95, {"ta"}, "Tamil க")
    check_eq(0x0C95, {"kn"}, "Kannada ಕ")
    check_eq(0x0D15, {"ml"}, "Malayalam ക")
    check_eq(0x0A95, {"gu"}, "Gujarati ક")
    check_eq(0x0A15, {"pa"}, "Punjabi Gurmukhi ਕ")
    check_eq(0x1000, {"my"}, "Burmese က")

    # --- v2: sister-language attribution ---
    # ø was attributed only to da in v1; should now be da + nb (Norwegian).
    check_superset(ord("ø"), {"da", "nb"}, "ø — Danish + Norwegian Bokmål")
    check_superset(ord("å"), {"da", "nb", "sv", "fi"},
                   "å — Danish + Norwegian + Swedish + Finnish")
    # ș ț are Romanian. v1 had no Romanian; v2 adds it.
    check_superset(ord("ș"), {"ro"}, "ș — Romanian")
    check_superset(ord("ț"), {"ro"}, "ț — Romanian")
    # ə is Azerbaijani.
    check_superset(ord("ə"), {"az"}, "ə — Azerbaijani schwa")
    # і ї are Ukrainian.
    check_superset(ord("ð"), {"is"}, "ð — Icelandic eth")

    # NFD-post-fallback: polytonic precomposed → combining ypogegrammeni
    check_superset(
        0x0345,
        {"el-polyton"},
        "U+0345 ypogegrammeni — via NFD of polytonic ᾳ ῃ ῳ etc.",
    )

    # Cyrillic/Arabic fallback REMOVED — chars without positive CLDR
    # evidence still must be 0-bit. v3.1 added kk (Kazakh) and ps
    # (Pashto), so chars previously 0-bit now carry their locale.
    check_eq(0x049A, {"kk"}, "Қ Kazakh — kk added in v3.1")
    check_eq(0x06A4, set(), "ڤ Sindhi/Pashto-Sindhi — sd still out of scope")

    # Extra substrate codepoints
    check_all_bits(0x00B5, "µ MICRO SIGN — extra substrate")
    check_all_bits(0x00AA, "ª FEMININE ORDINAL — extra substrate")
    check_all_bits(0x00BA, "º MASCULINE ORDINAL — extra substrate")
    check_all_bits(0xFF21, "Ａ FULLWIDTH LATIN A — extra substrate")
    check_all_bits(0xFF41, "ａ FULLWIDTH LATIN a — extra substrate")
    check_all_bits(0xFF15, "５ FULLWIDTH DIGIT FIVE — extra substrate")

    # NFKC-aware script detection: halfwidth Katakana ｦ should resolve
    # to Kana → Jpan via the script-range fallback.
    check_superset(
        0xFF66,
        {"ja"},
        "ｦ HALFWIDTH KATAKANA WO — Jpan via NFKC-aware script detection",
    )

    check_superset(0x0456, {"uk"}, "і — Ukrainian i")
    check_superset(0x0457, {"uk"}, "ї — Ukrainian yi")
    # ј is Macedonian / Serbian-Cyrillic.
    check_superset(0x0458, {"mk", "sr-Cyrl"}, "ј — Cyrillic je")

    # --- Out-of-scope ---
    check_eq(0x1F600, set(), "emoji symbol is substrate but not in table?")
    # Actually emoji has category So — should be substrate-all-bits via apply
    # script fallback, not in the table. So 0 in the table is OK.
    # Re-check: 0x1F600 happens to be category So (Symbol, Other) → apply
    # treats as substrate. Not stored in build. Validate-time: by_cp.get → 0.
    # This is correct behaviour, not a failure.

    # --- v3 per-level char assertions ---
    def check_family(cp: int, expected: set[str], note: str) -> None:
        m = by_cp_family.get(cp, 0)
        actual = {c for c, b in family_bit.items() if m & (1 << b)}
        if actual != expected:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} family ({note}): "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )

    def check_script(cp: int, expected: set[str], note: str) -> None:
        m = by_cp_script.get(cp, 0)
        actual = {c for c, b in script_bit.items() if m & (1 << b)}
        if actual != expected:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} script ({note}): "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )

    check_family(ord("ñ"), {"Romance-Latn"}, "ñ Spanish-only")
    check_family(ord("ß"), {"Germanic-Latn"}, "ß German-only")
    check_family(ord("ł"), {"Slavic-Latn"}, "ł Polish-only")
    check_family(0x03B1, {"Grek-modern", "Grek-polyton"}, "α both Greek")
    check_family(0x1F00, {"Grek-polyton"}, "ἀ polytonic-only")
    check_family(0x4E2D, {"Sinitic-Hans", "Sinitic-Hant", "Japonic"}, "中 CJK")
    check_family(0xB2E4, {"Hangul-family"}, "다 Korean")
    check_family(0x0915, {"Devanagari-family"}, "क Hindi")

    check_script(ord("a"), {"Latn"}, "'a' Latin only")
    check_script(ord("ñ"), {"Latn"}, "ñ Latin only at script level")
    check_script(0x03B1, {"Grek-modern", "Grek-polyton"}, "α both Greek scripts")
    check_script(0x1F00, {"Grek-polyton"}, "ἀ polytonic Greek only")
    check_script(0x4E2D, {"Hans", "Hant", "Jpan"}, "中 three CJK scripts")
    check_script(0x0430, {"Cyrl"}, "Russian а — Cyrillic")
    check_script(0x0035, set(script_bit), "'5' substrate — all scripts")

    # --- derivation consistency: family_bits == projection of bitmask
    # across the family-locale mapping for a sample of codepoints.
    derivation_samples = [
        ord("a"), ord("ñ"), ord("ß"), ord("ł"), ord("ä"),
        0x03B1, 0x1F00, 0x0430, 0x4E2D, 0xB2E4,
        0x0035, ord("."), ord(" "),
    ]
    for cp in derivation_samples:
        lang_bits = by_cp.get(cp, 0)
        if lang_bits == 0:
            continue
        is_substrate = (lang_bits == all_bits)
        expected_family = (
            all_family if is_substrate
            else derive_family_bits(lang_bits, families, lang_code_to_bit)
        )
        expected_script = (
            all_script if is_substrate
            else derive_script_bits(lang_bits, scripts, script_to_lang_mask)
        )
        if by_cp_family.get(cp, 0) != expected_family:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} derivation: "
                f"family stored={by_cp_family.get(cp, 0):#x} "
                f"expected={expected_family:#x}"
            )
        if by_cp_script.get(cp, 0) != expected_script:
            failures.append(
                f"  U+{cp:04X} {chr(cp)!r} derivation: "
                f"script stored={by_cp_script.get(cp, 0):#x} "
                f"expected={expected_script:#x}"
            )

    if failures:
        print(f"FAIL phase 1 — {len(failures)} assertion(s):")
        for line in failures:
            print(line)
        sys.exit(1)
    print(
        f"phase 1 OK — {len(by_cp)} codepoints in table, "
        f"{len(bit_for)} language bits / {len(family_bit)} family bits / "
        f"{len(script_bit)} script bits, all char checks passed."
    )

    # ---------------- Phase 2: token-level audit gate ----------------
    token_path = here / "artifacts" / "token_language_bitmask.parquet"
    if not token_path.exists():
        print(
            f"phase 2 SKIP — {token_path.name} does not exist; "
            "run apply_to_apertus_vocab.py to produce it."
        )
        return

    token_audit_threshold = 50

    # Wire-format sanity: mask columns are fixed-width binary[16] at
    # all three levels.
    tt_full = pq.read_table(token_path)
    mask_cols = (
        "script_and", "script_or",
        "family_and", "family_or",
        "bitmask_and", "bitmask_or",
    )
    for col in mask_cols:
        actual = tt_full.schema.field(col).type
        if not actual.equals(pa.binary(BITMASK_BYTES)):
            print(
                f"FAIL phase 2 — token column `{col}` type is {actual}, "
                f"expected {pa.binary(BITMASK_BYTES)}"
            )
            sys.exit(1)

    # Recompute a handful of tokens from the char table and check the
    # stored AND/OR match at all three levels — catches wire-format
    # drift between build and apply.
    token_recheck_failures = _recheck_token_masks(
        tt_full,
        by_cp_script, by_cp_family, by_cp,
        all_script, all_family, all_bits,
    )
    if token_recheck_failures:
        print(
            f"FAIL phase 2 — {len(token_recheck_failures)} token "
            f"mask mismatch(es):"
        )
        for line in token_recheck_failures:
            print(line)
        sys.exit(1)
    print(
        "phase 2 wire-format OK — bitmask_and / bitmask_or are "
        f"binary({BITMASK_BYTES}); spot-checked tokens recompute."
    )

    in_scope_scripts = collect_in_scope_scripts(spec["languages"])
    txts = tt_full["decoded_text"].to_pylist()
    sts = tt_full["status"].to_pylist()

    script_counter: collections.Counter[str] = collections.Counter()
    for txt, s in zip(txts, sts):
        if s not in ("text_with_unmodeled_letters", "no_in_scope_chars"):
            continue
        if not txt:
            continue
        scripts_in_token: set[str] = set()
        for ch in txt:
            cat = unicodedata.category(ch)
            if cat[0] not in "LM" or cat == "Lm":
                continue
            scripts_in_token.add(_codepoint_script(ord(ch)))
        for sc in scripts_in_token:
            script_counter[sc] += 1

    # Pass criterion: each OUT-OF-SCOPE script has <threshold tokens.
    # In-scope scripts with high counts are reported (they reflect
    # language-coverage gaps within a covered script — e.g. Kazakh
    # letters in Cyrillic) but don't fail the gate, because the
    # rejection behaviour for those is correct: tokens get 0 bits.
    out_of_scope_breaches: list[str] = []
    for sc, n in script_counter.most_common():
        in_scope = sc in in_scope_scripts
        if n >= token_audit_threshold and not in_scope:
            out_of_scope_breaches.append(f"  [BREACH] {sc:18s} {n:6d}")
            continue
        tag = (
            "in-scope, coverage-gap" if (in_scope and n >= token_audit_threshold)
            else "in-scope" if in_scope
            else "out-of-scope ok"
        )
        print(f"  [{tag:22s}] {sc:18s} {n:6d}")

    if out_of_scope_breaches:
        print(
            f"\nFAIL phase 2 — {len(out_of_scope_breaches)} out-of-scope "
            f"script(s) over <{token_audit_threshold} threshold:"
        )
        for line in out_of_scope_breaches:
            print(line)
        sys.exit(1)

    print(
        f"phase 2 OK — all out-of-scope scripts have "
        f"<{token_audit_threshold} fall-through tokens."
    )


def collect_in_scope_scripts(languages: list[dict]) -> set[str]:
    return {L["script"] for L in languages}


def _recheck_token_masks(
    tt: "pa.Table",
    by_cp_script: dict[int, int],
    by_cp_family: dict[int, int],
    by_cp_lang: dict[int, int],
    all_script: int,
    all_family: int,
    all_lang: int,
) -> list[str]:
    """Pick representative `text` tokens, recompute their AND/OR at
    all three levels from the per-level char tables + substrate-
    fallback rule, and assert the stored bytes match. Catches wire-
    format drift between build and apply at every level.
    """
    ids = tt["token_id"].to_pylist()
    txts = tt["decoded_text"].to_pylist()
    sts = tt["status"].to_pylist()
    s_and = tt["script_and"].to_pylist()
    s_or = tt["script_or"].to_pylist()
    f_and = tt["family_and"].to_pylist()
    f_or = tt["family_or"].to_pylist()
    l_and = tt["bitmask_and"].to_pylist()
    l_or = tt["bitmask_or"].to_pylist()

    candidates: list[int] = []
    for i, (txt, s) in enumerate(zip(txts, sts)):
        if s != "text" or not txt:
            continue
        candidates.append(i)
        if len(candidates) >= 12:
            break
    for i, s in enumerate(sts):
        if s == "text_with_unmodeled_letters":
            candidates.append(i)
            break

    failures: list[str] = []
    for i in candidates:
        txt = txts[i]
        if txt is None:
            continue
        exp_s_and, exp_s_or = all_script, 0
        exp_f_and, exp_f_or = all_family, 0
        exp_l_and, exp_l_or = all_lang, 0
        for ch in txt:
            ms, _ = _cp_bits(ord(ch), by_cp_script, all_script)
            mf, _ = _cp_bits(ord(ch), by_cp_family, all_family)
            ml, _ = _cp_bits(ord(ch), by_cp_lang, all_lang)
            exp_s_and &= ms; exp_s_or |= ms
            exp_f_and &= mf; exp_f_or |= mf
            exp_l_and &= ml; exp_l_or |= ml

        checks = [
            ("script_and", decode_mask(s_and[i]), exp_s_and),
            ("script_or", decode_mask(s_or[i]), exp_s_or),
            ("family_and", decode_mask(f_and[i]), exp_f_and),
            ("family_or", decode_mask(f_or[i]), exp_f_or),
            ("bitmask_and", decode_mask(l_and[i]), exp_l_and),
            ("bitmask_or", decode_mask(l_or[i]), exp_l_or),
        ]
        for label, stored, expected in checks:
            if stored != expected:
                failures.append(
                    f"  token_id={ids[i]} text={txt!r} {label}: "
                    f"stored={hex(stored)} expected={hex(expected)}"
                )
    return failures


_SCRIPT_PREFIXES: list[tuple[str, str]] = [
    ("LATIN", "Latn"), ("GREEK", "Grek"), ("COPTIC", "Copt"),
    ("CYRILLIC", "Cyrl"), ("ARABIC", "Arab"), ("HEBREW", "Hebr"),
    ("DEVANAGARI", "Deva"), ("BENGALI", "Beng"), ("TAMIL", "Taml"),
    ("TELUGU", "Telu"), ("KANNADA", "Knda"), ("MALAYALAM", "Mlym"),
    ("GUJARATI", "Gujr"), ("GURMUKHI", "Guru"), ("THAI", "Thai"),
    ("MYANMAR", "Mymr"), ("HANGUL", "Hang"), ("CJK", "Hans"),
    ("KANGXI", "Hans"), ("HIRAGANA", "Jpan"), ("KATAKANA", "Jpan"),
    ("ARMENIAN", "Armn"), ("GEORGIAN", "Geor"), ("LAO", "Laoo"),
    ("KHMER", "Khmr"), ("TIBETAN", "Tibt"), ("MONGOLIAN", "Mong"),
    ("SINHALA", "Sinh"), ("ETHIOPIC", "Ethi"), ("CHEROKEE", "Cher"),
    ("SYRIAC", "Syrc"), ("THAANA", "Thaa"), ("YI", "Yiii"),
]


def _codepoint_script(cp: int) -> str:
    try:
        name = unicodedata.name(chr(cp))
    except ValueError:
        return "Unnamed"
    for prefix, script in _SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return script
    return "Other"


if __name__ == "__main__":
    main()

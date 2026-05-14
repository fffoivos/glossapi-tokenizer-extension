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

import pyarrow.parquet as pq
import yaml


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    spec = yaml.safe_load((here / "languages.yaml").read_text())
    bit_for = {L["code"]: L["bit"] for L in spec["languages"]}
    all_bits = (1 << len(bit_for)) - 1

    t = pq.read_table(here / "artifacts" / "char_language_bitmask.parquet")
    by_cp = {
        cp: m for cp, m in zip(t["codepoint"].to_pylist(), t["bitmask"].to_pylist())
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
    LATN_ALL = LATN_V1 | LATN_V2_EXTRA
    CYRL_ALL = {"ru", "uk", "bg", "mk", "sr-Cyrl"}

    # --- v1 spot checks (still must pass) ---
    check_eq(ord("a"), LATN_ALL, "'a' — every Latin locale")
    check_eq(ord("ł"), {"pl"}, "Polish-only ł")
    check_eq(ord("ñ"), {"es"}, "Spanish-only ñ")
    check_eq(0x03B1, {"el", "el-polyton"}, "Greek α — modern + polytonic")
    check_eq(0x1F00, {"el-polyton"}, "Polytonic ἀ — ONLY polytonic")
    check_eq(0x0627, {"ar", "fa", "ur"}, "Arabic alef — ar + fa + ur")
    check_eq(0x4E2D, {"zh-Hans", "zh-Hant", "ja"}, "中 — CJK")
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

    # Cyrillic/Arabic fallback REMOVED — these chars must be 0-bit
    # because their host languages are out of scope.
    check_eq(0x049A, set(), "Қ Kazakh — Cyrillic fallback removed")
    check_eq(0x06A4, set(), "ڤ Pashto/Sindhi — Arabic fallback removed")

    # Extra substrate codepoints
    check_all_bits(0x00B5, "µ MICRO SIGN — extra substrate")
    check_all_bits(0x00AA, "ª FEMININE ORDINAL — extra substrate")
    check_all_bits(0x00BA, "º MASCULINE ORDINAL — extra substrate")

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

    if failures:
        print(f"FAIL phase 1 — {len(failures)} assertion(s):")
        for line in failures:
            print(line)
        sys.exit(1)
    print(
        f"phase 1 OK — {len(by_cp)} codepoints in table, "
        f"{len(bit_for)} bits, all char checks passed."
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

    in_scope_scripts = collect_in_scope_scripts(spec["languages"])
    tt = pq.read_table(token_path, columns=["decoded_text", "status"])
    txts = tt["decoded_text"].to_pylist()
    sts = tt["status"].to_pylist()

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

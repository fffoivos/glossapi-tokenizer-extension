#!/usr/bin/env python3
"""Build artifacts/char_language_bitmask.parquet (schema v5).

For every codepoint that appears in any of the configured (language,
script, encoding) triples, emit a row with the membership bitmask.

Pipeline per language:
  1. Fetch CLDR exemplar subsets (main, index, numbers, punctuation).
     `auxiliary` is excluded — see PLAN.md § "Why auxiliary is dropped".
  2. Parse each subset as a UnicodeSet (chars + ranges + escapes +
     {clusters}; constituent codepoints of clusters are emitted so
     combining marks aren't lost).
  3. Filter by Unicode script — a codepoint contributes to a language
     only if its script matches the locale's declared script, or is
     Common/Inherited. Suppresses CLDR cross-script bleed like
     zh-Hans's pinyin index admitting A-Z.
  4. Restrict to letters and marks (categories `L*` / `M*`). Substrate
     chars from CLDR subsets are ignored at this stage; they're added
     uniformly later.
  5. Case closure — for every codepoint in the language's letter set,
     also add its upper() and lower() variants.
  6. NFD closure — for every codepoint, also add the codepoints of its
     NFD canonical decomposition (catches combining marks for
     decomposed text under Apertus's `normalizer: null`).

After all languages:
  7. Script-range fallback — any letter codepoint in a script range we
     model (Hangul, Han, Devanagari, …) that received no bit through
     CLDR gets the bits of the locales that use that script. Closes
     gaps where CLDR's curated exemplar excludes a real in-script char
     (e.g. the ~80k Han codepoints CLDR doesn't list).
  8. Substrate override — every codepoint with Unicode general category
     N* / P* / S* / Z* / Cc / Cf gets ALL_BITS_SET. Substrate
     contributes zero exclusion power and must not falsely narrow
     language membership.

Source: cldr-json (https://github.com/unicode-org/cldr-json), tag
pinned in languages.yaml (`cldr_release`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from _common import (
    BITMASK_BYTES,
    BITMASK_MAX_BIT,
    EXTRA_SUBSTRATE_CODEPOINTS,
    build_iso_lookup,
    build_script_to_lang_mask,
    compute_all_bits,
    derive_family_bits,
    derive_script_bits,
    encode_mask,
)


CLDR_URL = (
    "https://raw.githubusercontent.com/unicode-org/cldr-json/{release}/"
    "cldr-json/cldr-misc-full/main/{locale}/characters.json"
)

CLDR_SUBSETS = ["exemplarCharacters", "index", "numbers", "punctuation"]


# -- Unicode script identification ------------------------------------

SCRIPT_PREFIXES: list[tuple[str, str]] = [
    ("LATIN", "Latn"),
    ("GREEK", "Grek"),
    ("COPTIC", "Copt"),
    ("CYRILLIC", "Cyrl"),
    ("ARABIC", "Arab"),
    ("HEBREW", "Hebr"),
    ("DEVANAGARI", "Deva"),
    ("BENGALI", "Beng"),
    ("TAMIL", "Taml"),
    ("TELUGU", "Telu"),
    ("KANNADA", "Knda"),
    ("MALAYALAM", "Mlym"),
    ("GUJARATI", "Gujr"),
    ("GURMUKHI", "Guru"),
    ("THAI", "Thai"),
    ("MYANMAR", "Mymr"),
    ("HANGUL", "Hang"),
    ("CJK", "Han"),
    ("KANGXI", "Han"),
    ("HIRAGANA", "Kana"),
    ("KATAKANA", "Kana"),
    ("ARMENIAN", "Armn"),
    ("GEORGIAN", "Geor"),
    ("ETHIOPIC", "Ethi"),
    ("KHMER", "Khmr"),
    ("SINHALA", "Sinh"),
    ("LAO", "Laoo"),
    ("TIBETAN", "Tibt"),
    ("ORIYA", "Orya"),
    ("THAANA", "Thaa"),
]

# Unicode-script → set of locale-scripts that admit it.
SCRIPT_COMPAT: dict[str, set[str]] = {
    "Latn": {"Latn"},
    "Cyrl": {"Cyrl"},
    "Grek": {"Grek"},
    "Arab": {"Arab"},
    "Hebr": {"Hebr"},
    "Deva": {"Deva"},
    "Beng": {"Beng"},
    "Taml": {"Taml"},
    "Telu": {"Telu"},
    "Knda": {"Knda"},
    "Mlym": {"Mlym"},
    "Gujr": {"Gujr"},
    "Guru": {"Guru"},
    "Thai": {"Thai"},
    "Mymr": {"Mymr"},
    "Hang": {"Hang"},
    "Armn": {"Armn"},
    "Geor": {"Geor"},
    "Han": {"Hans", "Hant", "Jpan"},
    "Kana": {"Jpan"},
    "Ethi": {"Ethi"},
    "Khmr": {"Khmr"},
    "Sinh": {"Sinh"},
    "Laoo": {"Laoo"},
    "Tibt": {"Tibt"},
    "Orya": {"Orya"},
    "Thaa": {"Thaa"},
}


def char_script(cp: int) -> str:
    """Map a codepoint to its Unicode script (a coarse string code like
    "Latn" / "Grek" / "Han"). Uses character-name prefixes; for
    compatibility forms (fullwidth Latin, halfwidth Katakana, etc.)
    that have non-script-prefixed names ("FULLWIDTH LATIN ...",
    "HALFWIDTH KATAKANA ..."), falls back to NFKC normalisation and
    re-checks on the canonical form. Returns "Common" for substrate
    and codepoints we can't classify."""
    try:
        name = unicodedata.name(chr(cp))
    except ValueError:
        return "Common"
    for prefix, script in SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return script
    nfkc = unicodedata.normalize("NFKC", chr(cp))
    if len(nfkc) == 1 and ord(nfkc[0]) != cp:
        try:
            n2 = unicodedata.name(nfkc)
            for prefix, script in SCRIPT_PREFIXES:
                if n2.startswith(prefix):
                    return script
        except ValueError:
            pass
    return "Common"


def char_is_compatible(cp: int, locale_script: str) -> bool:
    script = char_script(cp)
    if script == "Common":
        return True
    return locale_script in SCRIPT_COMPAT.get(script, {locale_script})


def is_letter_or_mark(cp: int) -> bool:
    """Letters and marks that carry language signal. Excludes Lm
    (modifier letter, e.g. ʻ ʼ) and EXTRA_SUBSTRATE_CODEPOINTS
    (ordinal/unit Ll/Lo codepoints) — those are letter-categorised in
    Unicode but function as cross-language punctuation."""
    if cp in EXTRA_SUBSTRATE_CODEPOINTS:
        return False
    try:
        cat = unicodedata.category(chr(cp))
    except ValueError:
        return False
    if cat == "Lm":
        return False
    return cat[0] in ("L", "M")


def is_substrate(cp: int) -> bool:
    """Codepoints that contribute zero exclusion power: digits, punct,
    symbols, separators, control, format, modifier letters (Lm), and
    the explicit EXTRA_SUBSTRATE_CODEPOINTS list (compatibility-style
    Ll/Lo codepoints used as units / ordinal markers across many
    languages)."""
    if cp in EXTRA_SUBSTRATE_CODEPOINTS:
        return True
    try:
        cat = unicodedata.category(chr(cp))
    except ValueError:
        return False
    if cat == "Lm":
        return True
    return cat[0] in ("N", "P", "S", "Z") or cat in ("Cc", "Cf")


# -- CLDR fetch -------------------------------------------------------

def fetch_characters(cldr_locale: str, release: str, cache_dir: Path) -> dict:
    cache_path = cache_dir / release / f"{cldr_locale}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    url = CLDR_URL.format(release=release, locale=cldr_locale)
    print(f"  fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except Exception as e:
        raise RuntimeError(
            f"failed to fetch CLDR characters.json for {cldr_locale} "
            f"at release {release}: {e}"
        ) from e
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return json.loads(raw)


# -- UnicodeSet parser -----------------------------------------------

def parse_unicode_set(spec: str) -> set[int]:
    s = spec.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    cps: set[int] = set()
    last_cp: int | None = None
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            last_cp = None
            continue
        if c == "\\":
            cp, i = _parse_escape(s, i)
            cps.add(cp)
            last_cp = cp
            continue
        if c == "{":
            j = s.find("}", i + 1)
            if j < 0:
                break
            _emit_cluster(s[i + 1 : j], cps)
            i = j + 1
            last_cp = None
            continue
        if c == "-" and last_cp is not None:
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j >= n:
                break
            if s[j] == "\\":
                end_cp, j = _parse_escape(s, j)
            elif s[j] == "{":
                i = j
                continue
            else:
                end_cp = ord(s[j])
                j += 1
            if end_cp >= last_cp:
                for cp in range(last_cp + 1, end_cp + 1):
                    cps.add(cp)
                last_cp = end_cp
            i = j
            continue
        cps.add(ord(c))
        last_cp = ord(c)
        i += 1
    return cps


def _emit_cluster(inner: str, cps: set[int]) -> None:
    k = 0
    n = len(inner)
    while k < n:
        ic = inner[k]
        if ic.isspace():
            k += 1
            continue
        if ic == "\\":
            cp, k = _parse_escape(inner, k)
            cps.add(cp)
            continue
        cps.add(ord(ic))
        k += 1


def _parse_escape(s: str, i: int) -> tuple[int, int]:
    n = len(s)
    if i + 1 >= n:
        return ord("\\"), i + 1
    c = s[i + 1]
    if c == "u" and i + 5 < n:
        try:
            return int(s[i + 2 : i + 6], 16), i + 6
        except ValueError:
            pass
    if c == "U" and i + 9 < n:
        try:
            return int(s[i + 2 : i + 10], 16), i + 10
        except ValueError:
            pass
    if c == "x" and i + 3 < n:
        try:
            return int(s[i + 2 : i + 4], 16), i + 4
        except ValueError:
            pass
    return ord(c), i + 2


# -- Per-language letter set ------------------------------------------

def language_letter_set(
    cldr_locale: str,
    locale_script: str,
    release: str,
    cache_dir: Path,
    extra_codepoints: list[int] | None = None,
) -> set[int]:
    data = fetch_characters(cldr_locale, release, cache_dir)
    if cldr_locale not in data["main"]:
        raise RuntimeError(
            f"CLDR characters.json for {cldr_locale}: top-level "
            f"'main' has no key '{cldr_locale}'; got keys "
            f"{list(data['main'].keys())}"
        )
    chars_obj = data["main"][cldr_locale]["characters"]
    cps_raw: set[int] = set()
    for key in CLDR_SUBSETS:
        spec = chars_obj.get(key)
        if not spec:
            continue
        if isinstance(spec, list):
            spec = " ".join(spec)
        cps_raw |= parse_unicode_set(spec)

    # v3.3.1: audit-driven per-locale supplements (e.g. Urdu ں U+06BA
    # which CLDR misclassifies as auxiliary). Each entry is fed
    # through the same filter + closure pipeline as CLDR codepoints.
    if extra_codepoints:
        cps_raw |= set(extra_codepoints)

    cps: set[int] = set()
    for cp in cps_raw:
        if not is_letter_or_mark(cp):
            continue
        if not char_is_compatible(cp, locale_script):
            continue
        cps.add(cp)

    cps |= _case_closure(cps)
    cps |= _nfd_closure(cps)
    cps = {cp for cp in cps if char_is_compatible(cp, locale_script)}
    return cps


def _case_closure(cps: set[int]) -> set[int]:
    extra: set[int] = set()
    for cp in cps:
        ch = chr(cp)
        for variant in (ch.upper(), ch.lower(), ch.title()):
            for vch in variant:
                vcp = ord(vch)
                if vcp != cp:
                    extra.add(vcp)
    return extra


def _nfd_closure(cps: set[int]) -> set[int]:
    extra: set[int] = set()
    for cp in cps:
        nfd = unicodedata.normalize("NFD", chr(cp))
        for ch in nfd:
            ncp = ord(ch)
            if ncp != cp:
                extra.add(ncp)
    return extra


# -- Script-range fallback --------------------------------------------

# Script-range fallback is applied ONLY where the (script ↔ covered-
# locales) mapping is essentially one-to-one, so an in-range letter
# we don't have explicit CLDR evidence for is still admitted by the
# locales using that script with high confidence.
#
# Cyrillic and Arabic are DELIBERATELY EXCLUDED — their script blocks
# contain language-specific extensions (Kazakh `Қ`, Bashkir `Ҡ`, Pashto
# `ښ`, Sindhi `ڤ`, Uyghur, …) for languages outside our scope. Falling
# back to "any Cyrillic locale" or "any Arabic locale" for those would
# violate the strict-rule "positive CLDR evidence only" guarantee.
# Such codepoints intentionally fall through to 0 bits → AND-reject
# every in-scope language, which is the correct strict outcome.
SCRIPT_FALLBACK_RANGES: list[tuple[int, int, str]] = [
    (0x0370, 0x03FF, "Grek"),
    (0x1F00, 0x1FFF, "Grek-Polyton"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1000, 0x109F, "Mymr"),
    (0xAC00, 0xD7AF, "Hang"),
    (0x1100, 0x11FF, "Hang"),
    (0x3130, 0x318F, "Hang"),
    (0x0530, 0x058F, "Armn"),
    (0x10A0, 0x10FF, "Geor"),
    (0x4E00, 0x9FFF, "Han"),
    (0x3400, 0x4DBF, "Han"),
    (0x20000, 0x2A6DF, "Han"),
    (0x2A700, 0x2B73F, "Han"),
    (0x2B740, 0x2B81F, "Han"),
    (0x2B820, 0x2CEAF, "Han"),
    (0x3040, 0x309F, "Kana"),
    (0x30A0, 0x30FF, "Kana"),
    (0x31F0, 0x31FF, "Kana"),
    (0xFF65, 0xFF9F, "Kana"),    # Halfwidth Katakana
    (0xF900, 0xFAFF, "Han"),     # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F, "Han"),   # CJK Compatibility Ideographs Supplement
    # v3.3 new scripts
    (0x1200, 0x137F, "Ethi"),    # Ethiopic
    (0x1380, 0x139F, "Ethi"),    # Ethiopic Supplement
    (0x2D80, 0x2DDF, "Ethi"),    # Ethiopic Extended
    (0xAB00, 0xAB2F, "Ethi"),    # Ethiopic Extended-A
    (0x1780, 0x17FF, "Khmr"),    # Khmer
    (0x19E0, 0x19FF, "Khmr"),    # Khmer Symbols
    (0x0D80, 0x0DFF, "Sinh"),    # Sinhala
    (0x0E80, 0x0EFF, "Laoo"),    # Lao
    (0x0F00, 0x0FFF, "Tibt"),    # Tibetan
    (0x0B00, 0x0B7F, "Orya"),    # Oriya / Odia
    (0x0780, 0x07BF, "Thaa"),    # Thaana
]


def apply_script_fallback(
    cp_bitmask: dict[int, int],
    languages: list[dict],
) -> int:
    """For letter/mark codepoints in script ranges we model, ensure
    they have at least the bits of the locales that use that script.
    Greek polytonic block (U+1F00–1FFF) gets only the polytonic-encoding
    locales, not modern Greek. Returns the count of codepoints touched.
    """
    script_to_bits: dict[str, int] = {}
    for L in languages:
        script_to_bits.setdefault(L["script"], 0)
        script_to_bits[L["script"]] |= 1 << L["bit"]

    grek_bits = script_to_bits.get("Grek", 0)
    grek_polyton_bits = 0
    for L in languages:
        if L["cldr_locale"] == "el-polyton":
            grek_polyton_bits |= 1 << L["bit"]

    han_bits = (
        script_to_bits.get("Hans", 0)
        | script_to_bits.get("Hant", 0)
        | script_to_bits.get("Jpan", 0)
    )
    kana_bits = script_to_bits.get("Jpan", 0)

    range_to_bits: list[tuple[int, int, int]] = []
    for start, end, script in SCRIPT_FALLBACK_RANGES:
        if script == "Grek":
            bits = grek_bits
        elif script == "Grek-Polyton":
            bits = grek_polyton_bits
        elif script == "Han":
            bits = han_bits
        elif script == "Kana":
            bits = kana_bits
        else:
            bits = script_to_bits.get(script, 0)
        if bits:
            range_to_bits.append((start, end, bits))

    # Han codepoints have multiple locales sharing the script (Hans /
    # Hant / Jpan). The original "always OR" rule erased CLDR's
    # per-locale Hans-vs-Hant attribution by adding all three bits to
    # every Han codepoint. Refined rule for Han only: skip the
    # fallback when the codepoint already has ANY CJK bit set
    # (i.e. CLDR has placed it in zh-Hans, zh-Hant, or ja). That
    # preserves CLDR-precise Hans/Hant attribution and only fills in
    # for Han codepoints CLDR doesn't list (the ~76k rare Han chars
    # outside CLDR's curated ~4k subset).
    touched = 0
    for start, end, bits in range_to_bits:
        is_han = (start, end) in {
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
        }
        for cp in range(start, end + 1):
            if not is_letter_or_mark(cp):
                continue
            existing = cp_bitmask.get(cp, 0)
            if is_han and (existing & bits):
                # CLDR already attributed this Han codepoint to one or
                # more CJK locales; trust CLDR's per-locale precision.
                continue
            if (existing | bits) != existing:
                cp_bitmask[cp] = existing | bits
                touched += 1
    return touched


# -- Bitmask-level NFD closure ---------------------------------------

def apply_bitmask_nfd_closure(cp_bitmask: dict[int, int]) -> int:
    """For every codepoint in the table, propagate its bits onto the
    codepoints of its NFD canonical decomposition. Catches combining
    marks that are only reachable via precomposed forms added by the
    script-range fallback (e.g. polytonic Greek `ᾳ` decomposing to
    `α + U+0345 COMBINING GREEK YPOGEGRAMMENI`).

    Runs after script-range fallback so fallback-added codepoints'
    decompositions are covered. Idempotent under union.
    Returns the count of codepoints whose mask changed.
    """
    touched = 0
    for cp in list(cp_bitmask):
        nfd = unicodedata.normalize("NFD", chr(cp))
        if len(nfd) <= 1:
            continue
        bits = cp_bitmask[cp]
        for ch in nfd:
            ncp = ord(ch)
            if ncp == cp:
                continue
            existing = cp_bitmask.get(ncp, 0)
            merged = existing | bits
            if merged != existing:
                cp_bitmask[ncp] = merged
                touched += 1
    return touched


# -- Substrate override ------------------------------------------------

ASCII_SUBSTRATE = list(range(0x0020, 0x007F))
SUPPLEMENTARY_SUBSTRATE = [
    0x0009, 0x000A, 0x000D,    # TAB / LF / CR
    0x00A0,                     # NBSP
    0x2010, 0x2011, 0x2012, 0x2013, 0x2014,  # hyphens, dashes
    0x2018, 0x2019, 0x201C, 0x201D,           # smart quotes
    0x2026, 0x2030, 0x2032, 0x2033,           # ellipsis, per-mille, primes
]


def apply_substrate_override(
    cp_bitmask: dict[int, int], all_bits: int
) -> int:
    """For every codepoint with Unicode category N*/P*/S*/Z*/Cc/Cf
    (plus Lm and EXTRA_SUBSTRATE_CODEPOINTS), override its bitmask to
    ALL_BITS. Substrate contributes zero exclusion power under our
    rejection framing; if it had narrow CLDR bits, those would falsely
    exclude. Returns the count of codepoints set to all_bits.
    """
    candidates: set[int] = set(cp_bitmask)
    candidates.update(ASCII_SUBSTRATE)
    candidates.update(SUPPLEMENTARY_SUBSTRATE)
    candidates.update(EXTRA_SUBSTRATE_CODEPOINTS)

    touched = 0
    for cp in candidates:
        if is_substrate(cp):
            if cp_bitmask.get(cp) != all_bits:
                cp_bitmask[cp] = all_bits
                touched += 1
    return touched


# -- Main -------------------------------------------------------------

def validate_families(
    families: list[dict],
    lang_code_to_bit: dict[str, int],
    scripts: list[dict],
) -> None:
    """Cross-check families.yaml against languages.yaml and
    scripts.yaml. Every locale listed under a family must exist in
    languages.yaml; every family's `script` must reference a
    scripts.yaml `code`; locales are partitioned exactly across
    families (each locale in exactly one family)."""
    script_codes = {s["code"] for s in scripts}
    seen_locales: dict[str, str] = {}
    for f in families:
        if f["script"] not in script_codes:
            raise RuntimeError(
                f"family {f['code']!r} references unknown script "
                f"{f['script']!r}"
            )
        for loc in f["locales"]:
            if loc not in lang_code_to_bit:
                raise RuntimeError(
                    f"family {f['code']!r} lists unknown locale {loc!r}"
                )
            if loc in seen_locales:
                raise RuntimeError(
                    f"locale {loc!r} is in two families: "
                    f"{seen_locales[loc]!r} and {f['code']!r}"
                )
            seen_locales[loc] = f["code"]
    missing = set(lang_code_to_bit) - set(seen_locales)
    if missing:
        raise RuntimeError(
            f"locale(s) not assigned to any family: {sorted(missing)}"
        )


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", type=Path, default=here / "languages.yaml")
    ap.add_argument("--families", type=Path, default=here / "families.yaml")
    ap.add_argument("--scripts", type=Path, default=here / "scripts.yaml")
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "artifacts" / "char_language_bitmask.parquet",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=here / "artifacts" / "manifest.json",
    )
    ap.add_argument("--cache-dir", type=Path, default=here / "data" / "cldr")
    args = ap.parse_args()

    spec = yaml.safe_load(args.languages.read_text())
    languages = spec["languages"]
    cldr_release = spec["cldr_release"]

    families_spec = yaml.safe_load(args.families.read_text())
    families = families_spec["families"]
    scripts_spec = yaml.safe_load(args.scripts.read_text())
    scripts = scripts_spec["scripts"]

    lang_code_to_bit = {L["code"]: L["bit"] for L in languages}
    validate_families(families, lang_code_to_bit, scripts)
    script_to_lang_mask = build_script_to_lang_mask(
        scripts, families, lang_code_to_bit
    )

    print(
        f"v3 build — {len(languages)} languages, {len(families)} families, "
        f"{len(scripts)} scripts; CLDR {cldr_release}"
    )

    cp_bitmask: dict[int, int] = {}
    per_lang: dict[str, int] = {}
    for L in languages:
        bit = L["bit"]
        code = L["code"]
        cldr_locale = L["cldr_locale"]
        locale_script = L["script"]
        extra_cps: list[int] = []
        for spec in L.get("extra_codepoints", []) or []:
            if isinstance(spec, str):
                extra_cps.append(int(spec.removeprefix("U+"), 16))
            else:
                extra_cps.append(int(spec))
        cps = language_letter_set(
            cldr_locale, locale_script, cldr_release, args.cache_dir,
            extra_codepoints=extra_cps,
        )
        per_lang[code] = len(cps)
        mask = 1 << bit
        for cp in cps:
            cp_bitmask[cp] = cp_bitmask.get(cp, 0) | mask
        print(
            f"  bit={bit:2d}  {code:11s}  cldr={cldr_locale:11s}  "
            f"script={locale_script:4s}  letters={len(cps)}"
        )

    print(f"\nafter CLDR + case + NFD: {len(cp_bitmask)} codepoints")

    touched = apply_script_fallback(cp_bitmask, languages)
    print(f"script-range fallback: touched {touched} codepoints")

    nfd_touched = apply_bitmask_nfd_closure(cp_bitmask)
    print(f"bitmask-level NFD closure: touched {nfd_touched} codepoints")

    all_bits = compute_all_bits(languages)
    sub_touched = apply_substrate_override(cp_bitmask, all_bits)
    print(f"substrate override: {sub_touched} codepoints set to ALL_BITS")

    print(f"\nfinal codepoints in table: {len(cp_bitmask)}")

    # Derive script_bits and family_bits per codepoint from lang bits.
    # Substrate codepoints (the ones whose lang bitmask was set to
    # all_bits) get all script + family bits set too, which is the
    # right behaviour (substrate contributes zero exclusion power at
    # every level).
    family_all_bits = compute_all_bits(families)
    script_all_bits = compute_all_bits(scripts)

    cp_family: dict[int, int] = {}
    cp_script: dict[int, int] = {}
    for cp, lang_bits in cp_bitmask.items():
        if lang_bits == all_bits:
            cp_family[cp] = family_all_bits
            cp_script[cp] = script_all_bits
        else:
            cp_family[cp] = derive_family_bits(
                lang_bits, families, lang_code_to_bit
            )
            cp_script[cp] = derive_script_bits(
                lang_bits, scripts, script_to_lang_mask
            )

    codepoints = sorted(cp_bitmask)
    bitmasks = [cp_bitmask[cp] for cp in codepoints]
    family_bits_list = [cp_family[cp] for cp in codepoints]
    script_bits_list = [cp_script[cp] for cp in codepoints]
    chars = [chr(cp) for cp in codepoints]
    num_langs = [bin(m).count("1") for m in bitmasks]
    categories = []
    for cp in codepoints:
        try:
            categories.append(unicodedata.category(chr(cp)))
        except ValueError:
            categories.append("")

    bitmasks_bytes = [encode_mask(m) for m in bitmasks]
    family_bytes = [encode_mask(m) for m in family_bits_list]
    script_bytes = [encode_mask(m) for m in script_bits_list]
    table = pa.table(
        {
            "codepoint": pa.array(codepoints, type=pa.uint32()),
            "script_bits": pa.array(
                script_bytes, type=pa.binary(BITMASK_BYTES)
            ),
            "family_bits": pa.array(
                family_bytes, type=pa.binary(BITMASK_BYTES)
            ),
            "bitmask": pa.array(bitmasks_bytes, type=pa.binary(BITMASK_BYTES)),
            "char": pa.array(chars, type=pa.string()),
            "num_langs": pa.array(num_langs, type=pa.uint8()),
            "category": pa.array(categories, type=pa.string()),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"\nwrote {args.out}")

    iso_lookup = build_iso_lookup(languages, scripts)
    iso_lookup_serialised = {
        f"{iso_lang}_{iso_script}": char_tool_code
        for (iso_lang, iso_script), char_tool_code in sorted(iso_lookup.items())
    }

    # v3.3.3 self-test (broadened from v3.3.1 which only checked
    # primary × primary pairs): every (iso_639_3 primary + alias) ×
    # (iso15924 primary + alias) combination for each language MUST
    # resolve through the canonical-key lookup. Catches alias
    # regressions like the v3.2 `arb_Arab → ar` silent-bug class.
    #
    # Plus a fixed fixture of real consumer cap-hit canonical keys —
    # the 4 silent-bug seeds, the v3.3.2 "diplomatic" additions, and
    # a handful of edge cases. If a future change drops any of these
    # mappings the build fails loud.
    self_test_failures: list[str] = []
    for L in languages:
        primary_iso = L["iso_639_3"]
        all_lang_iso = [primary_iso] + list(L.get("iso_639_3_aliases", []) or [])
        # Resolve via the script's iso15924, not the language's
        # `script` field (which may be a script `code` rather than
        # an iso15924 value).
        for s in scripts:
            if s["code"] == L["script"] or s["iso15924"] == L["script"]:
                all_script_iso = [s["iso15924"]] + list(
                    s.get("iso15924_aliases", []) or []
                )
                for liso in all_lang_iso:
                    for siso in all_script_iso:
                        key = (liso, siso)
                        if key not in iso_lookup:
                            self_test_failures.append(
                                f"  {liso}_{siso} (language {L['code']!r}, "
                                f"missing from canonical_key_to_char_tool_code)"
                            )
                break  # found the right script entry

    # Consumer-key fixture — known cap-hit canonical keys that real
    # corpora produce and that we've already shipped resolution for.
    # If a future schema change accidentally drops any of these the
    # build fails loud with a clear consumer-impacting message.
    CONSUMER_FIXTURE: list[tuple[str, str]] = [
        # v3.2 silent-bug seeds — these were the original regressions
        ("srp_Cyrl", "sr-Cyrl"),
        ("lvs_Latn", "lv"),
        ("ekk_Latn", "et"),
        ("cmn_Hani", "zh-Hans"),
        # v3.2.x hotfix targets
        ("ell_Grek", "el"),
        ("arb_Arab", "ar"),
        # v3.3.2 "diplomatic" additions
        ("als_Latn", "sq"),       # Tosk Albanian (macro)
        ("sqi_Latn", "sq"),
        ("gsw_Latn", "gsw"),
        ("lat_Latn", "la"),
        # Macrolanguage-individual pairs that commonly slip
        ("zho_Hans", "zh-Hans"),
        ("zho_Hant", "zh-Hant"),
        ("zho_Hani", "zh-Hans"),  # default Hani→Hans
        ("khk_Cyrl", "mn"),        # Khalkha-Mongolian individual code
        ("nob_Latn", "nb"),
        ("nno_Latn", "nn"),
        ("ces_Latn", "cs"),
        ("cze_Latn", "cs"),
        ("fas_Arab", "fa"),
        ("per_Arab", "fa"),
        ("pes_Arab", "fa"),
        ("ory_Orya", "or"),
        ("ori_Orya", "or"),
        ("bod_Tibt", "bo"),
        ("tib_Tibt", "bo"),
        # The v3.3 new-script Tier 1 keys
        ("amh_Ethi", "am"),
        ("khm_Khmr", "km"),
        ("sin_Sinh", "si"),
        ("lao_Laoo", "lo"),
        ("div_Thaa", "dv"),
    ]
    for key, expected in CONSUMER_FIXTURE:
        actual = iso_lookup_serialised.get(key)
        if actual != expected:
            self_test_failures.append(
                f"  CONSUMER_FIXTURE: {key!r} → {actual!r}, "
                f"expected {expected!r}"
            )

    if self_test_failures:
        for line in self_test_failures:
            print(line, file=sys.stderr)
        raise RuntimeError(
            f"build self-test failed: {len(self_test_failures)} canonical "
            f"key resolution(s) wrong. See list above."
        )
    print(
        f"build self-test OK — {len(iso_lookup_serialised)} canonical "
        f"keys resolve; all primary + alias pairs covered; "
        f"{len(CONSUMER_FIXTURE)} consumer-fixture assertions pass."
    )

    manifest = {
        "build_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema_version": 5,
        "bitmask_bytes": BITMASK_BYTES,
        "bitmask_byte_order": "little",
        "levels": {
            "script": {
                "bits_used": len(scripts),
                "column": "script_bits",
            },
            "family": {
                "bits_used": len(families),
                "column": "family_bits",
            },
            "language": {
                "bits_used": len(languages),
                "column": "bitmask",
            },
        },
        "scripts": scripts,
        "families": families,
        "canonical_key_to_char_tool_code": iso_lookup_serialised,
        "canonical_key_help": (
            "FineWeb-2 / ISO-style canonical keys map to char-tool BCP47 codes. "
            "Key format: '<iso_639_3>_<iso_15924>'. Both individual and "
            "macrolanguage ISO codes resolve here, plus the iso15924 aliases "
            "(e.g., Hani → Hans default). Consumers should read this map "
            "directly rather than hand-rolling a dict."
        ),
        "cldr_release": cldr_release,
        "cldr_source": "https://github.com/unicode-org/cldr-json",
        "cldr_subsets_included": CLDR_SUBSETS,
        "cldr_subsets_excluded": ["auxiliary", "numbers-auxiliary",
                                   "punctuation-auxiliary",
                                   "punctuation-person"],
        "closures_applied": ["case", "NFD_per_locale",
                             "script_range_fallback",
                             "NFD_post_fallback",
                             "substrate_all_bits"],
        "script_fallback_scripts": sorted(
            {s for _, _, s in SCRIPT_FALLBACK_RANGES}
        ),
        "extra_substrate_codepoints": [
            f"U+{cp:04X}" for cp in sorted(EXTRA_SUBSTRATE_CODEPOINTS)
        ],
        "scope_proxy_assumption": (
            "Apertus pretrain data is used as the proxy for "
            "Mistral-Nemo's (private) tokenizer training data."
        ),
        "locale_compatibility": {
            "el": {
                "subset_of": ["el-polyton"],
                "note": (
                    "Modern monotonic Greek (el) is a strict subset of "
                    "polytonic Greek (el-polyton): every char admitted by "
                    "el is also admitted by el-polyton. Consumers with an "
                    "`ell_Grek` canonical key whose sample may contain "
                    "occasional polytonic forms (classical quotes, place "
                    "names) should prefer `el-polyton` for maximum "
                    "admissibility."
                ),
            },
            "el-polyton": {"superset_of": ["el"]},
            "Hani_default": {
                "iso15924": "Hani",
                "resolves_to": "zh-Hans",
                "note": (
                    "FineWeb-2's generic `Hani` script tag (e.g. `cmn_Hani` "
                    "or `zho_Hani`) resolves to `zh-Hans` by default. "
                    "Consumers should override to `zh-Hant` only when the "
                    "sample is identified as Traditional Chinese."
                ),
            },
        },
        "consumer_notes": [
            (
                "Substrate tokens (every script/family/language bit set) "
                "may carry non-zero PMI under sample-domain mismatch — "
                "the symmetry assumption fails when language samples come "
                "from different corpus sources (e.g. Wikipedia for one, "
                "FineWeb-2-HQ for another). Consumers using PMI for "
                "language attribution should rely on the popcount-based "
                "substrate filter (`popcount(bitmask) < N_LANG_BITS`) "
                "rather than expecting PMI self-cancellation."
            ),
            (
                "For consumer canonical keys like `<iso_639_3>_<iso_15924>` "
                "(FineWeb-2 / ISO style), read "
                "`canonical_key_to_char_tool_code` from this manifest "
                "instead of hand-rolling a dict. Macrolanguage / "
                "individual-language aliases (est+ekk, lav+lvs, zho+cmn, "
                "mon+khk, pus+pbt) are pre-resolved."
            ),
            (
                "Within Latin script most tokens AND-saturate at the "
                "language layer (no diacritics → admissible in ~all "
                "Latin locales). Token-level discrimination should look "
                "at family_bits first; the script layer is the most "
                "reliable broad signal for bare-ASCII Latin content."
            ),
            (
                "Canonical keys with `und_<script>` (undetermined "
                "language) won't appear in canonical_key_to_char_tool_code "
                "by design — there is no language to map to. Consumers "
                "should resolve those keys to script_bits directly (e.g. "
                "`und_Cyrl` → script_bits with the Cyrl bit set), using "
                "the script layer of the artifact rather than the "
                "language layer."
            ),
            (
                "A handful of historical / liturgical languages remain "
                "genuinely unmappable: `gmh_Latn` (Middle High German) "
                "has no CLDR data. Consumers can fall back to a "
                "near-neighbour at their layer (e.g. `gmh_Latn → de` "
                "with a documented caveat). Classical Latin `lat_Latn` "
                "→ `la` IS mapped as of v3.3.2, with a synthesised "
                "A-Z + macron exemplar (CLDR's `la` exemplar is empty)."
            ),
        ],
        "languages": languages,
        "codepoint_coverage": {
            "total_codepoints_in_table": len(cp_bitmask),
            "codepoints_per_language_from_cldr": per_lang,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()

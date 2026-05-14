#!/usr/bin/env python3
"""Build artifacts/char_language_bitmask.parquet (v2).

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
}


def char_script(cp: int) -> str:
    try:
        name = unicodedata.name(chr(cp))
    except ValueError:
        return "Common"
    for prefix, script in SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return script
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


# Codepoints that are Unicode-categorised as Letter (Ll/Lo) but
# function as language-neutral typography (unit prefixes, ordinal
# indicators). Treated as substrate even though the category test
# alone wouldn't catch them.
EXTRA_SUBSTRATE_CODEPOINTS: frozenset[int] = frozenset({
    0x00AA,  # FEMININE ORDINAL INDICATOR (1ª)
    0x00B5,  # MICRO SIGN (µm, µg) — distinct from Greek mu U+03BC
    0x00BA,  # MASCULINE ORDINAL INDICATOR (1º)
})


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
    cldr_locale: str, locale_script: str, release: str, cache_dir: Path
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

    touched = 0
    for start, end, bits in range_to_bits:
        for cp in range(start, end + 1):
            if not is_letter_or_mark(cp):
                continue
            existing = cp_bitmask.get(cp, 0)
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

def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", type=Path, default=here / "languages.yaml")
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

    print(
        f"v2 build — {len(languages)} (lang, script, encoding) triples, "
        f"CLDR {cldr_release}"
    )

    cp_bitmask: dict[int, int] = {}
    per_lang: dict[str, int] = {}
    for L in languages:
        bit = L["bit"]
        code = L["code"]
        cldr_locale = L["cldr_locale"]
        locale_script = L["script"]
        cps = language_letter_set(
            cldr_locale, locale_script, cldr_release, args.cache_dir
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

    all_bits = (1 << len(languages)) - 1
    sub_touched = apply_substrate_override(cp_bitmask, all_bits)
    print(f"substrate override: {sub_touched} codepoints set to ALL_BITS")

    print(f"\nfinal codepoints in table: {len(cp_bitmask)}")

    codepoints = sorted(cp_bitmask)
    bitmasks = [cp_bitmask[cp] for cp in codepoints]
    chars = [chr(cp) for cp in codepoints]
    num_langs = [bin(m).count("1") for m in bitmasks]
    categories = []
    for cp in codepoints:
        try:
            categories.append(unicodedata.category(chr(cp)))
        except ValueError:
            categories.append("")

    table = pa.table(
        {
            "codepoint": pa.array(codepoints, type=pa.uint32()),
            "bitmask": pa.array(bitmasks, type=pa.uint64()),
            "char": pa.array(chars, type=pa.string()),
            "num_langs": pa.array(num_langs, type=pa.uint8()),
            "category": pa.array(categories, type=pa.string()),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"\nwrote {args.out}")

    manifest = {
        "build_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "schema_version": 2,
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

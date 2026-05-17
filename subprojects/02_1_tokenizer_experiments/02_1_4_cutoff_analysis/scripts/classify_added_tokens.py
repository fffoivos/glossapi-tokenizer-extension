"""Combined classification of C3's 25,600 added tokens.

For each added token, we have two independent labels:
  - GLOSSARY category (Gemini-pass): greek_word / greek_fragment /
    greek_morpheme / proper_noun / greek_acronym / latin_word /
    latin_fragment / latin_acronym / latin_abbreviation /
    mixed_script_token / mojibake / encoding_artifact /
    table_separator / punctuation_run / escaped_character_run /
    math_symbol / dingbat_or_symbol / postscript_glyph /
    control_or_invisible / whitespace_only / code_identifier /
    url_or_path / unit_or_measure
  - CHAR-MASK bucket (from 02_2_1_char_language_membership):
    el_or_polyton / single:el-polyton / en_plus_others /
    universal_substrate / multi_other / no_lang / el_plus_others

We define a higher-level "function" label by combining both:
  - GREEK_CONTENT: glossary in {greek_word, greek_fragment,
    greek_morpheme, greek_acronym, proper_noun} AND char-mask is
    Greek-script (el_or_polyton or single:el-polyton). These are the
    primary payload of the extension.
  - USEFUL_STRUCTURAL: glossary in {table_separator, punctuation_run,
    escaped_character_run, math_symbol, dingbat_or_symbol,
    whitespace_only, url_or_path, code_identifier,
    latin_acronym, latin_abbreviation, unit_or_measure,
    postscript_glyph}. These are MD/code/URL/structural patterns
    that genuinely appear in Greek text and have legitimate use.
  - NOISE: glossary in {mojibake, encoding_artifact,
    control_or_invisible}. Real undesirable artifacts.
  - AMBIGUOUS: glossary in {mixed_script_token, latin_fragment,
    latin_word}. Latin pieces that may or may not be useful (loanwords
    vs noise).

Then per-cutoff (every 1024) we tabulate:
  - GREEK_CONTENT subbreakdown by glossary category
  - USEFUL_STRUCTURAL subbreakdown
  - NOISE breakdown (with actual examples)
  - AMBIGUOUS breakdown

For the final cutoff-decision view we also report:
  - Greek content as % of added budget
  - Useful-structural budget
  - Noise budget
  - Marginal Greek-content per +1024 step
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import os

REPO = Path(__file__).resolve().parents[4]
TOOL = REPO / "subprojects/02_2_tokenizer_implementation/02_2_1_char_language_membership"
GLOSSARY = Path("/home/foivos/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl")
CHAR_BM = TOOL / "artifacts/char_language_bitmask.parquet"
LANG_YAML = TOOL / "languages.yaml"
SCRIPT_YAML = TOOL / "scripts.yaml"
MANIFEST = TOOL / "artifacts/manifest.json"

# Default output to this sub-subproject's local artifacts/ dir; override
# with CLASSIFY_OUT_DIR env var if you want a different sink.
_HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get(
    "CLASSIFY_OUT_DIR",
    str((_HERE / "../artifacts").resolve()),
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTRATE_GC_PREFIXES = {"N", "P", "S", "Z"}
SUBSTRATE_GC_FULL = {"Cc", "Cf"}


def load_layer_widths() -> tuple[int, int]:
    """Read live language- and script-layer bit widths from the manifest
    instead of hardcoding. Falls back to the (88, 29) widths used at
    schema v5 if the manifest is missing.
    """
    try:
        m = json.loads(MANIFEST.read_text())
        lang_bits = int(m["levels"]["language"]["bits_used"])
        script_bits = int(m["levels"]["script"]["bits_used"])
    except (FileNotFoundError, KeyError, ValueError):
        lang_bits, script_bits = 88, 29
    return lang_bits, script_bits


LANG_BITS, SCRIPT_BITS = load_layer_widths()
ALL_BITS_LANG = (1 << LANG_BITS) - 1
ALL_BITS_SCRIPT = (1 << SCRIPT_BITS) - 1

GREEK_GLOSSARY_CATS = {
    "greek_word", "greek_fragment", "greek_morpheme",
    "greek_acronym", "proper_noun",
}
USEFUL_GLOSSARY_CATS = {
    "table_separator", "punctuation_run", "escaped_character_run",
    "math_symbol", "dingbat_or_symbol", "whitespace_only",
    "url_or_path", "code_identifier",
    "latin_acronym", "latin_abbreviation", "unit_or_measure",
    "postscript_glyph",
}
NOISE_GLOSSARY_CATS = {
    "mojibake", "encoding_artifact", "control_or_invisible",
}
AMBIGUOUS_GLOSSARY_CATS = {
    "mixed_script_token", "latin_fragment", "latin_word",
}


def load_yaml_codes(path: Path, key: str) -> dict[int, str]:
    data = yaml.safe_load(path.read_text())
    return {entry["bit"]: entry["code"] for entry in data[key]}


def little_endian_to_int(b: bytes) -> int:
    return int.from_bytes(b, "little")


def classify_lang(mask_and: int, bit_to_code: dict[int, str]) -> str:
    bits = [b for b in range(LANG_BITS) if mask_and & (1 << b)]
    if not bits:
        return "no_lang"
    codes = {bit_to_code[b] for b in bits}
    if codes == {"en"}:
        return "en_exclusive"
    if codes == {"el"}:
        return "el_exclusive"
    if codes == {"el", "el-polyton"}:
        return "el_or_polyton"
    # treat "substrate" as everything within 90% of the live language layer width
    if len(bits) >= int(0.9 * LANG_BITS):
        return "universal_substrate"
    if len(bits) == 1:
        return f"single:{next(iter(codes))}"
    if "el" in codes or "el-polyton" in codes:
        return "el_plus_others"
    if "en" in codes:
        return "en_plus_others"
    return "multi_other"


def load_char_table() -> tuple[dict[int, int], dict[int, str]]:
    cols = ["codepoint", "bitmask", "category"]
    t = pq.read_table(CHAR_BM, columns=cols).to_pylist()
    cp_lang = {}
    cp_gc = {}
    for r in t:
        cp = r["codepoint"]
        cp_lang[cp] = little_endian_to_int(r["bitmask"])
        cp_gc[cp] = r["category"]
    return cp_lang, cp_gc


def codepoint_bits_local(cp: int, table: dict[int, int]) -> int:
    if cp in table:
        return table[cp]
    import unicodedata
    gc = unicodedata.category(chr(cp))
    if gc[0] in SUBSTRATE_GC_PREFIXES or gc in SUBSTRATE_GC_FULL:
        return ALL_BITS_LANG
    return 0


def compute_token_lang_and(text: str, cp_lang: dict[int, int]) -> int:
    if not text:
        return 0
    m_and = ALL_BITS_LANG
    for ch in text:
        m_and &= codepoint_bits_local(ord(ch), cp_lang)
    return m_and


def function_label(glossary_cat: str, lang_bucket: str) -> str:
    if glossary_cat in GREEK_GLOSSARY_CATS:
        # Greek by glossary; sanity-check char-mask agrees
        if lang_bucket in ("el_or_polyton", "single:el-polyton", "el_plus_others"):
            return "GREEK"
        # Glossary says Greek but char-mask says no Greek: edge case
        # (e.g., Ohm sign Ω classified as math_symbol gets glossary=greek_acronym
        # in some weird cases; or strict-glossary disagreement)
        return "GREEK_disagree"
    if glossary_cat in USEFUL_GLOSSARY_CATS:
        return "USEFUL_STRUCTURAL"
    if glossary_cat in NOISE_GLOSSARY_CATS:
        return "NOISE"
    if glossary_cat in AMBIGUOUS_GLOSSARY_CATS:
        return "AMBIGUOUS_LATIN_OR_MIXED"
    return f"OTHER:{glossary_cat}"


def main() -> None:
    bit_to_lang = load_yaml_codes(LANG_YAML, "languages")
    cp_lang, gc_table = load_char_table()

    rows = []
    with GLOSSARY.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(r)
    rows.sort(key=lambda r: int(r["id"]))

    # Annotate every added token
    classified = []
    for r in rows:
        text = r["decoded"]
        m_and = compute_token_lang_and(text, cp_lang)
        lang_bucket = classify_lang(m_and, bit_to_lang)
        glossary_cat = r["category"]
        fn = function_label(glossary_cat, lang_bucket)
        classified.append({
            "id": int(r["id"]),
            "decoded": text,
            "glossary": glossary_cat,
            "lang_bucket": lang_bucket,
            "function": fn,
        })

    cutoffs = [n * 1024 for n in range(1, 12)]  # 1k..11k

    # Per-cutoff: function breakdown + Greek subbreakdown by glossary
    rep = {}
    for n in cutoffs:
        upto = classified[:n]
        fn_counts = Counter(x["function"] for x in upto)
        greek_sub = Counter(x["glossary"] for x in upto if x["function"] == "GREEK")
        useful_sub = Counter(x["glossary"] for x in upto if x["function"] == "USEFUL_STRUCTURAL")
        noise_sub = Counter(x["glossary"] for x in upto if x["function"] == "NOISE")
        ambig_sub = Counter(x["glossary"] for x in upto if x["function"] == "AMBIGUOUS_LATIN_OR_MIXED")
        rep[n] = {
            "n_added": n,
            "function_counts": dict(fn_counts),
            "greek_breakdown": dict(greek_sub),
            "useful_breakdown": dict(useful_sub),
            "noise_breakdown": dict(noise_sub),
            "ambiguous_breakdown": dict(ambig_sub),
            "greek_examples": [x["decoded"] for x in upto if x["function"] == "GREEK"][:0],  # too many
            "noise_examples": [(x["decoded"], x["glossary"]) for x in upto if x["function"] == "NOISE"],
            "ambiguous_examples": [(x["decoded"], x["glossary"]) for x in upto if x["function"] == "AMBIGUOUS_LATIN_OR_MIXED"],
            "useful_examples": [(x["decoded"], x["glossary"]) for x in upto if x["function"] == "USEFUL_STRUCTURAL"],
        }

    # Print readable summary
    print("=" * 100)
    print("FUNCTION-LEVEL BREAKDOWN PER CUTOFF (1k..11k)")
    print("=" * 100)
    header = f"{'cutoff':>8} | {'GREEK':>6} | {'USEFUL':>7} | {'NOISE':>6} | {'AMBIG':>6} | {'GR%':>5} | {'NOISE%':>7}"
    print(header)
    print("-" * len(header))
    for n in cutoffs:
        c = rep[n]["function_counts"]
        gr = c.get("GREEK", 0)
        gd = c.get("GREEK_disagree", 0)
        uf = c.get("USEFUL_STRUCTURAL", 0)
        ns = c.get("NOISE", 0)
        ab = c.get("AMBIGUOUS_LATIN_OR_MIXED", 0)
        gr_pct = (gr + gd) / n * 100
        ns_pct = ns / n * 100
        print(f"{n:>8d} | {gr+gd:>6d} | {uf:>7d} | {ns:>6d} | {ab:>6d} | {gr_pct:>4.1f}% | {ns_pct:>6.2f}%")

    print("\n" + "=" * 100)
    print("GREEK SUB-BREAKDOWN PER CUTOFF (by glossary category)")
    print("=" * 100)
    greek_cats_seen = sorted({c for n in cutoffs for c in rep[n]["greek_breakdown"]})
    print(f"{'cutoff':>8} | " + " | ".join(f"{c:>16s}" for c in greek_cats_seen))
    for n in cutoffs:
        gb = rep[n]["greek_breakdown"]
        print(f"{n:>8d} | " + " | ".join(f"{gb.get(c, 0):>16d}" for c in greek_cats_seen))

    print("\n" + "=" * 100)
    print("USEFUL-STRUCTURAL BREAKDOWN PER CUTOFF")
    print("=" * 100)
    useful_cats_seen = sorted({c for n in cutoffs for c in rep[n]["useful_breakdown"]})
    print(f"{'cutoff':>8} | " + " | ".join(f"{c:>22s}" for c in useful_cats_seen))
    for n in cutoffs:
        ub = rep[n]["useful_breakdown"]
        print(f"{n:>8d} | " + " | ".join(f"{ub.get(c, 0):>22d}" for c in useful_cats_seen))

    print("\n" + "=" * 100)
    print("NOISE BREAKDOWN PER CUTOFF + EXAMPLES AT 11264")
    print("=" * 100)
    noise_cats_seen = sorted({c for n in cutoffs for c in rep[n]["noise_breakdown"]})
    print(f"{'cutoff':>8} | " + " | ".join(f"{c:>22s}" for c in noise_cats_seen))
    for n in cutoffs:
        nb = rep[n]["noise_breakdown"]
        print(f"{n:>8d} | " + " | ".join(f"{nb.get(c, 0):>22d}" for c in noise_cats_seen))

    print("\nNOISE TOKENS AT 11264:")
    for tok, cat in rep[11264]["noise_examples"]:
        print(f"  [{cat:25s}] {tok!r}")

    print("\n" + "=" * 100)
    print("AMBIGUOUS LATIN/MIXED BREAKDOWN + EXAMPLES AT 11264")
    print("=" * 100)
    ambig_cats_seen = sorted({c for n in cutoffs for c in rep[n]["ambiguous_breakdown"]})
    print(f"{'cutoff':>8} | " + " | ".join(f"{c:>22s}" for c in ambig_cats_seen))
    for n in cutoffs:
        ab = rep[n]["ambiguous_breakdown"]
        print(f"{n:>8d} | " + " | ".join(f"{ab.get(c, 0):>22d}" for c in ambig_cats_seen))

    print("\nAMBIGUOUS TOKENS AT 11264:")
    for tok, cat in rep[11264]["ambiguous_examples"]:
        print(f"  [{cat:25s}] {tok!r}")

    print("\nUSEFUL_STRUCTURAL TOKENS AT 11264 (sampled, first 30):")
    for tok, cat in rep[11264]["useful_examples"][:30]:
        print(f"  [{cat:25s}] {tok!r}")

    # Marginal Greek-content per +1024 step
    print("\n" + "=" * 100)
    print("MARGINAL GREEK-CONTENT GAIN PER +1024 STEP")
    print("=" * 100)
    print(f"{'step':>16} | {'Δgreek':>8} | {'Δuseful':>8} | {'Δnoise':>8} | {'Δambig':>8}")
    prev_gr = prev_uf = prev_ns = prev_ab = 0
    for n in cutoffs:
        c = rep[n]["function_counts"]
        gr = c.get("GREEK", 0) + c.get("GREEK_disagree", 0)
        uf = c.get("USEFUL_STRUCTURAL", 0)
        ns = c.get("NOISE", 0)
        ab = c.get("AMBIGUOUS_LATIN_OR_MIXED", 0)
        print(f"{n-1024:>5d} -> {n:>5d} | {gr-prev_gr:>8d} | {uf-prev_uf:>8d} | {ns-prev_ns:>8d} | {ab-prev_ab:>8d}")
        prev_gr, prev_uf, prev_ns, prev_ab = gr, uf, ns, ab

    # Save full per-token classification
    with (OUT_DIR / "classified_added_tokens.jsonl").open("w") as fh:
        for x in classified:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    (OUT_DIR / "per_cutoff_report.json").write_text(
        json.dumps({str(k): v for k, v in rep.items()}, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {OUT_DIR / 'classified_added_tokens.jsonl'}")
    print(f"wrote {OUT_DIR / 'per_cutoff_report.json'}")


if __name__ == "__main__":
    main()

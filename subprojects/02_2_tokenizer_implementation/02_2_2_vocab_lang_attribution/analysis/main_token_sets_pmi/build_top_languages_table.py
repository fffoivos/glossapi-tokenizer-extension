"""Build the per-language vocab-share ranking with token-uniqueness counts.

Reads:
  - summary.tsv                          (per-key masked/unmasked counts)
  - tables/<key>__masked.txt             (per-key token ids; Variant A set)
  - ../../outputs/lang_metadata.json     (per-key name + family)

Writes:
  - TOP_LANGUAGES_BY_VOCAB_SHARE.md      (ranked table, one row per language)

Language-vs-key model
---------------------
The PMI pass tracks 87 cap-hit *keys*. One of those keys (English) is
duplicated across two source corpora — `eng_Latn` (general crawl) and
`eng_Latn_fineweb_hq` (FineWeb-HQ); both keys describe the same
language. To answer "share of vocab by language" honestly we must
collapse those two keys into a single English row whose masked set is
the union of the two key sets, and whose uniqueness is measured against
every NON-English key. All other 85 keys are distinct languages and
stay as their own row.

Definitions used in the output table:
  - "masked tokens" = size of the language's Variant-A set (PMI/count
    test passed + char-mask admissible), unioned across all keys that
    map to that language. One token may appear in MANY languages' sets;
    the per-language sets are NOT a disjoint partition of vocab.
  - "% of vocab" = masked / 131,072 (Apertus base vocab).
  - "unique tokens" = tokens that are admissible under THIS language
    and under NO other language in the cap-hit set. Counted in
    language space, not key space, so the dual-key English collapse is
    applied before counting.
  - "% unique" = unique / masked.
"""
from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY_TSV = HERE / "summary.tsv"
TABLES_DIR = HERE / "tables"
LANG_META = (
    HERE.parent.parent / "outputs" / "lang_metadata.json"
)
OUT_MD = HERE / "TOP_LANGUAGES_BY_VOCAB_SHARE.md"

APERTUS_VOCAB = 131_072


def load_masked_token_ids(path: Path) -> set[int]:
    """Each non-comment line is a Python dict literal: ``{<id>: <token>}``."""
    ids: set[int] = set()
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            d = ast.literal_eval(line)
            ids.update(int(k) for k in d.keys())
    return ids


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Per-key rows from summary.tsv (raw, before collapsing).
    # ------------------------------------------------------------------
    key_rows: list[dict] = []
    with SUMMARY_TSV.open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            key_rows.append(
                {
                    "key": r["target_key"],
                    "iso": r["lang_code"],
                    "masked": int(r["masked_count"]),
                    "unmasked": int(r["unmasked_count"]),
                }
            )

    # ------------------------------------------------------------------
    # 2. Names + families. `lang_metadata.json` uses internal placeholder
    #    strings for both English keys; override them so the table reads
    #    cleanly.
    # ------------------------------------------------------------------
    NAME_OVERRIDES = {
        "eng_Latn":             "English",
        "eng_Latn_fineweb_hq":  "English",
    }
    meta = json.loads(LANG_META.read_text())
    for r in key_rows:
        m = meta.get(r["key"], {})
        # Name: prefer override (for the placeholder-named English keys),
        # else lang_metadata's `name`, else blank.
        if r["key"] in NAME_OVERRIDES:
            r["name"] = NAME_OVERRIDES[r["key"]]
        else:
            n = m.get("name") or ""
            r["name"] = n if not n.startswith("<") else ""
        # Script: pull ISO 15924 from metadata; fall back to parsing the
        # `<iso639_3>_<Script>` key when metadata is missing/placeholder.
        script = m.get("script_iso15924") or ""
        if not script:
            parts = r["key"].split("_")
            script = parts[1] if len(parts) >= 2 else ""
        r["script"] = script

    # ------------------------------------------------------------------
    # 3. Load masked sets per key.
    # ------------------------------------------------------------------
    per_key_ids: dict[str, set[int]] = {}
    for r in key_rows:
        p = TABLES_DIR / f"{r['key']}__masked.txt"
        per_key_ids[r["key"]] = load_masked_token_ids(p) if p.exists() else set()

    # ------------------------------------------------------------------
    # 4. Collapse keys → languages.
    #    The only dual-keyed language in the cap-hit set is English
    #    (`eng_Latn` + `eng_Latn_fineweb_hq`). Every other key is its own
    #    language. The 5 `unmapped`/`und_*` keys stay each as their own
    #    row — they are not really "languages" but they belong to the
    #    cap-hit set.
    # ------------------------------------------------------------------
    LANGUAGE_COLLAPSE = {
        "eng_Latn":            "English",
        "eng_Latn_fineweb_hq": "English",
    }

    def lang_id_for(key: str) -> str:
        return LANGUAGE_COLLAPSE.get(key, key)

    lang_keys: dict[str, list[str]] = {}
    for kr in key_rows:
        lang_keys.setdefault(lang_id_for(kr["key"]), []).append(kr["key"])

    # ------------------------------------------------------------------
    # 5. Per-language masked union + a representative row's display
    #    fields (iso, name, family). For English, prefer the FineWeb-HQ
    #    iso/name; both English rows already point to the same display
    #    name via NAME_OVERRIDES.
    # ------------------------------------------------------------------
    lang_rows: list[dict] = []
    for lid, keys in lang_keys.items():
        ids_union: set[int] = set()
        for k in keys:
            ids_union |= per_key_ids[k]
        rep = next(kr for kr in key_rows if kr["key"] == keys[0])
        if lid == "English":
            iso, name, script = "en", "English", "Latn"
        else:
            iso, name, script = rep["iso"], rep["name"], rep["script"]
        lang_rows.append(
            {
                "lang_id": lid,
                "keys": keys,
                "iso": iso,
                "name": name,
                "script": script,
                "ids": ids_union,
                "masked": len(ids_union),
            }
        )

    # ------------------------------------------------------------------
    # 6. Uniqueness in LANGUAGE space:
    #    a token is "unique to language L" iff it belongs to L's masked
    #    union AND to no other language's masked union.
    # ------------------------------------------------------------------
    multiplicity: Counter[int] = Counter()
    for lr in lang_rows:
        multiplicity.update(lr["ids"])
    for lr in lang_rows:
        lr["unique"] = sum(1 for tid in lr["ids"] if multiplicity[tid] == 1)
        assert lr["unique"] <= lr["masked"]

    # ------------------------------------------------------------------
    # 7. Union / coverage stats (still over the full vocab).
    # ------------------------------------------------------------------
    covered_ids: set[int] = set()
    for lr in lang_rows:
        covered_ids |= lr["ids"]
    coverage = len(covered_ids)
    uncovered = APERTUS_VOCAB - coverage
    sum_masked = sum(lr["masked"] for lr in lang_rows)

    lang_rows.sort(key=lambda r: -r["masked"])

    # Compat alias for the markdown emit block below (still iterates as
    # `rows`).
    rows = lang_rows

    # ------------------------------------------------------------------
    # 8. Emit markdown.
    # ------------------------------------------------------------------
    lines: list[str] = []
    a = lines.append
    a("# Apertus tokenizer — per-language vocab share")
    a("")
    a("**Source**: `summary.tsv` + per-key `tables/<key>__masked.txt`")
    a("(this directory). **Apertus base vocab**: 131,072 tokens.")
    a("**Rebuild date**: 2026-05-15 (PMI/char-mask schema v5).")
    a("")
    a("This table ranks every language in the cap-hit set by the size")
    a("of its **masked-A set** — the tokens that pass *both* the")
    a("PMI/count admissibility test against the count-pooled cap-hit")
    a("marginal AND the strict char-admissibility mask for that")
    a("language. The cap-hit set is every language with at least 1 B")
    a("observed Apertus-token firings in the attribution pass.")
    a("")
    a("Per the attribution methodology, English is double-keyed across")
    a("two source corpora (`eng_Latn` = general crawl, `eng_Latn_fineweb_hq`")
    a("= FineWeb-HQ). Both keys describe the same language, so the row")
    a("below merges them: the English `masked` count is the union of the")
    a("two key sets (23,774), and English `unique` is computed against")
    a(f"every non-English language. After the merge there are **{len(rows)}**")
    a("language rows.")
    a("")
    a("## Reading caveats")
    a("")
    a("1. **The sets overlap heavily.** A single token like `_a`, `s`,")
    a("   or `,` is admissible under many Latin-script languages at")
    a(f"   once. Sum of masked counts across all {len(rows)} languages =")
    a(f"   **{sum_masked:,}** ({sum_masked / APERTUS_VOCAB * 100:.1f} %")
    a("   of vocab) — the excess over 100 % is double-counting.")
    a(f"2. **Union coverage**: **{coverage:,} / {APERTUS_VOCAB:,}**")
    a(f"   ({coverage / APERTUS_VOCAB * 100:.2f} %) tokens are")
    a(f"   admissible under at least one language. **{uncovered:,}**")
    a(f"   ({uncovered / APERTUS_VOCAB * 100:.2f} %) are unattributed")
    a("   (byte-fallback, punct, code, mojibake, residual long tail).")
    a("3. **Script matters more than family for tokens.** A BPE token's")
    a("   admissibility set is fixed by the script its codepoints lie in,")
    a("   not by linguistic family. The `script` column carries the ISO")
    a("   15924 code (`Latn`, `Grek`, `Cyrl`, `Arab`, `Hang`, `Jpan`,")
    a("   `Hani`, …). Languages on isolated scripts (Korean Hangul,")
    a("   Greek, Japanese kana) trivially get 100 %-unique sets because")
    a("   no other cap-hit language shares any codepoint with them.")
    a("   Latin-script languages share a large substrate and have low")
    a("   uniqueness even when very different linguistically.")
    a(f"4. **`und_*` keys** (`und_Mong`, `und_Kana`, `und_Grek`,")
    a(f"   `und_Cyrl`) and `gmh_Latn` are unmapped to ISO-639-1 codes;")
    a(f"   `iso` = `unmapped` in the row. Kept in for completeness.")
    a("")
    a("## Methodology pointers")
    a("")
    a("- Builder: `build.py` (this directory)")
    a("- PMI predicate / knobs: `manifest.json` (`alpha = 0.5`,")
    a("  `delta = 1.0`, `min_count = 100`, `marginal_floor = 1 e9`)")
    a("- Char-mask source: `02_2_1_char_language_membership/artifacts/`")
    a("  `token_language_bitmask.parquet`")
    a("- Firing counts: `02_2_2_vocab_lang_attribution/outputs/`")
    a("  `histogram_matrix.npz`")
    a("- Full methodology: `02_2_4_language_category_promotion/`")
    a("  `METHODOLOGY.md` + `PMI_PROMOTION_SPEC.md`")
    a("- Table reproduction: `build_top_languages_table.py` (this dir).")
    a("")
    def emit_table(sorted_rows):
        a("| # | iso | language | script | masked | % vocab | unique | % unique |")
        a("|---:|---|---|---|---:|---:|---:|---:|")
        for i, r in enumerate(sorted_rows, 1):
            pct = r["masked"] / APERTUS_VOCAB * 100
            upct = r["unique"] / r["masked"] * 100 if r["masked"] else 0.0
            name = r["name"] or "—"
            script = r["script"] or "—"
            a(
                f"| {i} | `{r['iso']}` | {name} | `{script}` "
                f"| {r['masked']:,} | {pct:.2f} % "
                f"| {r['unique']:,} | {upct:.2f} % |"
            )

    a("## Table — ranked by masked-set size")
    a("")
    a("How big is the language's PMI/mask-admissible set as a fraction")
    a("of the 131,072-token Apertus vocab. Languages on shared scripts")
    a("(Latin, Cyrillic, Arabic) rank high here because they inherit the")
    a("shared substrate; this is the \"how much of the vocab *touches*")
    a("this language\" view.")
    a("")
    emit_table(sorted(rows, key=lambda r: -r["masked"]))
    a("")
    a("## Table — ranked by unique-set size")
    a("")
    a("How many tokens the language owns ALONE — admissible under this")
    a("language and under no other cap-hit language. This is the \"how")
    a("much of the vocab is *dedicated* to this language\" view. Script-")
    a("isolated languages (own writing system) dominate the top of this")
    a("ranking because no other language can compete for their tokens.")
    a("Latin-script languages sit lower even when their masked set is")
    a("large, because their tokens are shared across the Latin family.")
    a("")
    emit_table(sorted(rows, key=lambda r: -r["unique"]))
    a("")

    # ------------------------------------------------------------------
    # 9. Rank-delta analysis: who climbed and who fell between the two
    #    rankings, cross-tabulated against whether the language sits on
    #    a script no other cap-hit language uses.
    # ------------------------------------------------------------------
    # "Script alone" = the language is the only cap-hit language with
    # masked_count > 0 on this script. The `und_*` keys (und_Grek,
    # und_Cyrl, und_Mong, und_Kana) and `gmh_Latn` all have masked = 0
    # so they nominally claim a script slot but contribute no token
    # competition. Counting them as competitors would falsely report
    # Greek as "shared" when functionally no other language admits
    # Grek-script tokens.
    script_lang_counts: Counter[str] = Counter(
        r["script"] for r in rows if r["masked"] > 0
    )
    masked_rank = {
        r["lang_id"]: i
        for i, r in enumerate(sorted(rows, key=lambda r: -r["masked"]), 1)
    }
    unique_rank = {
        r["lang_id"]: i
        for i, r in enumerate(sorted(rows, key=lambda r: -r["unique"]), 1)
    }
    for r in rows:
        r["masked_rank"] = masked_rank[r["lang_id"]]
        r["unique_rank"] = unique_rank[r["lang_id"]]
        r["delta"] = r["masked_rank"] - r["unique_rank"]  # +ve = climbed
        r["script_alone"] = (
            r["masked"] > 0 and script_lang_counts[r["script"]] == 1
        )

    upgraded = [r for r in rows if r["delta"] > 0]
    downgraded = [r for r in rows if r["delta"] < 0]
    same = [r for r in rows if r["delta"] == 0]

    def split(rs):
        alone = [r for r in rs if r["script_alone"]]
        shared = [r for r in rs if not r["script_alone"]]
        return alone, shared

    up_alone, up_shared = split(upgraded)
    dn_alone, dn_shared = split(downgraded)
    sm_alone, sm_shared = split(same)

    a("## Rank deltas — who climbed, who fell")
    a("")
    a("Side-by-side comparison of the two rankings. **Δ = masked rank −")
    a("unique rank**: positive Δ means the language climbed when we re-")
    a("ranked by unique tokens, negative means it dropped. \"Script")
    a("alone\" = no other language in the cap-hit set uses that ISO 15924")
    a("script; \"script shared\" = at least one other cap-hit language")
    a("uses it (e.g. Latn, Cyrl, Arab, Deva).")
    a("")
    a("### Summary")
    a("")
    a("| direction | script alone | script shared | total |")
    a("|---|---:|---:|---:|")
    a(f"| **climbed** (Δ > 0) | {len(up_alone)} | {len(up_shared)} "
      f"| {len(upgraded)} |")
    a(f"| **unchanged** (Δ = 0) | {len(sm_alone)} | {len(sm_shared)} "
      f"| {len(same)} |")
    a(f"| **dropped** (Δ < 0) | {len(dn_alone)} | {len(dn_shared)} "
      f"| {len(downgraded)} |")
    a(f"| **total** | "
      f"{len(up_alone)+len(sm_alone)+len(dn_alone)} | "
      f"{len(up_shared)+len(sm_shared)+len(dn_shared)} | "
      f"{len(rows)} |")
    a("")
    a("The pattern is very clean:")
    a("")
    a("- **Every effectively-alone language climbed** (16/16). 100 % of")
    a("  their masked set is unique by definition, so the unique ranking")
    a("  surfaces them above shared-script languages that lose tokens")
    a("  to neighbors.")
    a("- **Every dropper is on a shared script** (37/37 — Latn, Cyrl,")
    a("  Arab). They lose tokens to their script neighbors when we re-")
    a("  rank by unique-only.")
    a("- **17 shared-script languages climbed anyway**. These are")
    a("  languages whose vocab carries diacritic/letter combinations")
    a("  rare in the rest of their script family — e.g. Polish & Czech")
    a("  ł/ż/ę, Hungarian ő/ű, Turkish ş/ğ/ı, Vietnamese tone marks,")
    a("  Welsh ŵ/ŷ. Their unique tokens come from those distinctive")
    a("  subsets even though they share the Latin substrate.")
    a("- The **16 unchanged** rows split into two clusters:")
    a("  - **3 dominant Latin-script languages** (English, French,")
    a("    German) keep their #1/#2/#3 positions in both rankings —")
    a("    their masked sets are so large that even after losing the")
    a("    shared-substrate tokens to neighbors they still beat the")
    a("    next contender.")
    a("  - **13 zero-masked rows at the bottom** — zero-promoted scripts")
    a("    (Lao, Khmer, Odia, Tibetan, Sinhala, Amharic, Dhivehi, Hindi,")
    a("    Middle High German) and the four `und_*` keys.")
    a("")
    a("### Climbers (Δ > 0)")
    a("")

    def emit_delta_table(rs):
        a("| iso | language | script | script-share | masked rank | unique rank | Δ |")
        a("|---|---|---|---|---:|---:|---:|")
        for r in rs:
            ss = "alone" if r["script_alone"] else f"shared ({script_lang_counts[r['script']]} langs)"
            sign = "+" if r["delta"] > 0 else ("" if r["delta"] == 0 else "")
            a(f"| `{r['iso']}` | {r['name'] or '—'} | `{r['script']}` | "
              f"{ss} | {r['masked_rank']} | {r['unique_rank']} | "
              f"{sign}{r['delta']} |")

    emit_delta_table(sorted(upgraded, key=lambda r: -r["delta"]))
    a("")
    a("### Droppers (Δ < 0)")
    a("")
    emit_delta_table(sorted(downgraded, key=lambda r: r["delta"]))
    a("")
    a("### Unchanged (Δ = 0)")
    a("")
    emit_delta_table(sorted(same, key=lambda r: r["masked_rank"]))

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD} ({len(rows)} rows)")
    print(f"  union coverage: {coverage:,} / {APERTUS_VOCAB:,}")
    print(f"  sum of masked : {sum_masked:,} ({sum_masked/APERTUS_VOCAB*100:.1f}%)")


if __name__ == "__main__":
    main()

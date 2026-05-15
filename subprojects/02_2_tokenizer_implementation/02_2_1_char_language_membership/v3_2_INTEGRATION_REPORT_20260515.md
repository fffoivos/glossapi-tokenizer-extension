# v3.2 integration report — from the PMI promotion consumer

> Author: consumer at
> `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`.
> Date: 2026-05-15, immediately after v3.2 ship.
> Verdict: **ship clean apart from two manifest bugs (`ell_Grek`,
> `arb_Arab`).** Recommend v3.3.1 hotfix; v3.3 can build on it.
>
> **Closed 2026-05-15** — v3.3.1 hotfix shipped same session. All three
> consumer-flagged issues resolved at the manifest level. See
> "Post-v3.3.1 verification" section at the bottom.

## What worked smoothly (zero consumer changes needed)

1. **The 4 silent-bug keys resolve through the manifest.** All four
   (`srp_Cyrl`, `lvs_Latn`, `ekk_Latn`, `cmn_Hani`) appear in
   `canonical_key_to_char_tool_code` with the correct codes
   (`sr-Cyrl`, `lv`, `et`, `zh-Hans`). The 50-entry hardcoded
   `ISO_639_3_TO_BCP47` dict in `build.py` has been **deleted** —
   `make_lookup()` now reads the published map directly. Total
   consumer-side code reduction: ~50 lines.

2. **`category_or` uint8 column behaves exactly as the legend describes.**
   Bit values `L=1 M=2 N=4 P=8 S=16 Z=32 C=64`. The natural-text
   filter `(category_or & ~(L|M|Z)) == 0` works; the code-mixed
   filter `category_or & (P|S) != 0` works. Verified populations:
   - Vocab-wide: 119,230 `letters_only` (≈ 91 %), 11,304 with punct/sym
   - `eng_Latn_fineweb_hq` masked set: 95.1 % letters_only, 4.9 % code-mixed
   - PMI-below-δ uncovered bucket: 89 % letters_only, 11 % code-mixed

   This obsoletes my plan to reimplement Unicode-category logic
   consumer-side. **Thank you.**

3. **The 18 new locale additions all produce sane masked sets at
   first try.** Spot-checked content:

   | locale | top 3 masked tokens | sanity |
   | --- | --- | --- |
   | `sw` (Swahili) | ` ya`, ` na`, `wa` | Bantu function words ✓ |
   | `mr` (Marathi) | `ी`, `े`, `ण` | Devanagari vowel marks ✓ |
   | `cy` (Welsh) | ` yn`, ` y`, `w`, `dd` | Welsh digraphs visible ✓ |
   | `be` (Belarusian) | `ў`, ` ў`, ` і` | Belarusian-distinctive `ў` (T0) ✓ |
   | `eu` (Basque) | reasonable Basque function words | ✓ |
   | `tg` (Tajik Cyrillic) | Cyrillic-Persian content | ✓ |
   | `ar-MA` (Moroccan Arabic) | 7,029 tokens, 84 % mass | excellent coverage ✓ |
   | `ne` (Nepali) | 1,023 tokens, 89.49 % mass | Devanagari working ✓ |

4. **`locale_compatibility.el.subset_of: ["el-polyton"]` is documented
   in the manifest** — useful for the embedding diagnostic when it
   needs to decide which exemplar to use for `ell_Grek` samples that
   may contain occasional polytonic.

5. **`consumer_notes` paragraph in the manifest** about substrate ≠
   PMI under domain shift — gives downstream a heads-up that we
   surfaced empirically.

## Bugs found in v3.2 that needed consumer-side workarounds

### Bug #1 — `ell_Grek` missing from `canonical_key_to_char_tool_code`

The `el` language entry has `iso_639_3: "ell"` and `script: "Grek"`,
so the canonical key `ell_Grek` should auto-derive to `el`. It
doesn't appear in the map. Same for `gre_Grek` (the 639-2/B alias).

**Impact**: severe — Greek was the original anchor for the whole
project. Without the consumer-side patch, **Greek silently produces an
empty masked set**.

**Consumer workaround**: a derived-map fallback that iterates the
`languages` list, generating `f"{iso_639_3}_{script}"` and
`f"{alias}_{script}"` for every language entry's primary code + aliases.
This recovers `ell_Grek → el`, `gre_Grek → el`, and a handful of others.

**Suggested fix**: ensure the manifest-build code includes
`iso_639_3_<primary>_<script>` for every `languages` entry, not only
the script-aliased forms. Probably a one-line patch in the manifest
generator.

### Bug #2 — `arb_Arab` missing despite `ara_Arab → ar` being in the map

`ara` is the ISO 639-3 *macrolanguage* for Arabic; `arb` is the
*individual* code for Standard Arabic. FineWeb-2 uses `arb_Arab` in
practice. The map has `ara_Arab → ar` but not `arb_Arab`.

**Impact**: 1.0 B Standard Arabic tokens treated as unmapped →
empty masked set for `arb_Arab` until patched.

**Consumer workaround**: hardcoded `derived["arb_Arab"] = derived["ara_Arab"]`
as a one-off compatibility patch.

**Suggested fix**: add `"arb"` to the `iso_639_3_aliases` of the `ar`
language entry (analogous to how `ekk` is aliased under `et`). After
that, the derived-map iteration will pick it up automatically.
Possibly also worth auditing: do other macrolanguage entries have
similar missing individual-code aliases?

### Both bugs share a pattern

The shape of both is "map generation walks the `iso_639_3_aliases` but
misses the primary `iso_639_3` for some entries, OR the entry doesn't
declare alternative ISO codes that real corpora use". Suggest the
manifest-build script could include a self-test: for every cap-hit
canonical key in `02_2_2_vocab_lang_attribution/outputs/lang_metadata.json`,
assert the manifest produces a resolution.

## Coverage numbers (consumer-facing)

| metric | v3.1 (schema v4) | v3.2 (schema v5) | delta |
| --- | ---: | ---: | ---: |
| Apertus vocab covered | 81.18 % | **85.54 %** | +4.36 pp |
| tokens covered | 106,404 | **112,117** | +5,713 |
| unmapped cap-hit keys | 34 | **7** | −27 |
| Category 7 ("PMI < δ everywhere") | 14,188 | **8,546** | −5,642 |
| Category 5 ("fires only in unmapped lang") | 97 | **1** | −96 |

The bulk of the coverage gain came from the 18 new locale additions
absorbing the previously-uncovered "PMI < δ everywhere" short-token
bucket — when a new language joins, short shared tokens that were
distributed across many languages can now be PMI-promoted into the new
language's set.

## Outstanding items from the original feedback that v3.2 didn't address

| ask from feedback | v3.2 status |
| --- | --- |
| Urdu exemplar fix (`ں` U+06BA admitted by `ur`) | **not done** — 41 Urdu tokens with `popcount = 0` still uncovered |
| 7 new scripts (Ethi/Khmr/Sinh/Laoo/Tibt/Orya/Thaa) | deferred to v3.3 |
| Cross-script language family bits (Slavic across scripts, etc.) | not done, low priority |
| Norwegian Nynorsk (`nn`) | ✅ added |

## Recommendation for the agent

**Hotfix bullet list for v3.3.1** (~30 min):

1. Fix `canonical_key_to_char_tool_code` map-generation to include
   `iso_639_3_<primary>_<script>` entries for every language. Verify by
   running the consumer's `build.py` and confirming Greek + Standard
   Arabic + Bengali + Hebrew + Thai + etc. all resolve.
2. Add `iso_639_3_aliases: ["arb"]` to the `ar` language entry.
3. Add the Urdu `ں` (U+06BA) exemplar fix to the `ur` locale.
4. Optional: include a self-test in the build that asserts every
   cap-hit canonical key from the consumer's `lang_metadata.json`
   resolves to a non-None char-tool code.

After hotfix, the consumer's `derived` fallback can be removed and
the build.py path is even cleaner.

## After v3.3.1 ships

The PMI build is one command:

```bash
python3 02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/build.py
```

**Update — v3.3 has already shipped** (the 7 new scripts —
Ethi/Khmr/Sinh/Laoo/Tibt/Orya/Thaa — are present in the manifest's
`scripts` list and the canonical_key map). Empirical result: the
7 affected canonical keys (`amh_Ethi`, `khm_Khmr`, `sin_Sinh`,
`lao_Laoo`, `bod_Tibt`, `ory_Orya`, `div_Thaa`) now resolve to char-tool
codes (`am`, `km`, `si`, `lo`, `bo`, `or`, `dv`) but have **masked_count
of zero** for all seven. The Apertus tokenizer wasn't trained on
enough text in these scripts for any chars to land as a decoded
single-codepoint token — they all decode as `partial_utf8` byte
fragments and get classified T5 (unknown standalone). So v3.3
**correctly closes the canonical-key mapping** but **does not move the
coverage needle** for this particular vocab. My earlier estimate of
"~2 pp to ~87.5 %" was wrong — actual coverage stays at 85.54 % after
v3.3.

That's a Apertus-vocab fact, not a v3.3 defect: extending the
tokenizer to those scripts is a separate phase. For now v3.3 makes
those canonical keys queryable (Variant B has small per-key sets,
mostly Latin loanwords / numbers / punctuation) but Variant A is
empty.

## Net assessment

**v3.2 is a major improvement.** The schema work (publishing the
canonical_key map, the category_or column, the iso_639_3 aliases) is
exactly what the consumer needed. The two map-generation bugs are
small and trivially fixable; they don't undermine the design. The
consumer-side code is dramatically simpler (–50 lines, no hand-curated
ISO mapping). Ship v3.3 on top of v3.3.1 with confidence.

## Post-v3.3.1 verification (2026-05-15, after hotfix shipped)

The char-tool agent shipped v3.3.1 same session. All three flagged
issues resolved at the manifest level. Consumer-side workarounds
removed and re-run verified.

### Verification matrix

| consumer-flagged issue | manifest fix | verified consumer-side |
| --- | --- | --- |
| Bug #1 — `ell_Grek` missing | added `ell_Grek → el` and `gre_Grek → el` to map; tolerant script-match by code OR iso15924 in `build_iso_lookup` | ✓ resolves directly; consumer-side derived fallback removed |
| Bug #2 — `arb_Arab` missing | added `arb` to `ar`'s `iso_639_3_aliases` | ✓ resolves directly; consumer-side hardcoded patch removed |
| Urdu `ں` (U+06BA) missing from `ur` exemplar | new `extra_codepoints` field in `languages.yaml`; seeded `["U+06BA"]` on `ur`; build pipeline routes through script-compat + closures | ✓ 14 of 14 Apertus tokens containing `ں` now AND-attribute to `ur`; Category 4 drops 41 → 27 |
| (build-time safety) | self-test added — every language's primary `(iso_639_3, script)` pair must resolve through `iso_lookup` or build fails | ✓ same class of bug can't recur silently |

### `make_lookup()` after stripping workarounds

```python
def make_lookup(cl_manifest):
    ck_map = cl_manifest["canonical_key_to_char_tool_code"]
    def lookup(key):
        if key in ck_map:
            return ck_map[key]
        parts = key.split("_")
        if len(parts) >= 2:
            return ck_map.get(f"{parts[0]}_{parts[1]}")
        return None
    return lookup
```

9 lines. No ISO mapping, no derived fallback, no hardcoded patches.
The `[lookup] applied N derived-map fallback entries` log line no
longer fires.

### Numerical movement (v3.2 with consumer patches → v3.3.1 clean)

| metric | v3.2 (patched) | v3.3.1 (clean) | delta |
| --- | ---: | ---: | ---: |
| Apertus vocab covered | 112,117 (85.54 %) | **112,131 (85.55 %)** | +14 |
| `no_locale_admits_chars` (cat 4) | 41 | **27** | −14 |
| unmapped cap-hit keys | 7 | 7 (same list — only out-of-scope edge cases remain) | 0 |

The +14 movement is exactly the Urdu fix. Greek + Arabic didn't add
new coverage in the count because my consumer-side fallback was
already attributing them in the prior run — but now the fallback is
gone and the manifest does the job cleanly.

### Net assessment

**v3.3.1 lands clean.** The artifact is now in the cleanest state of
any version. Wire format stable (schema_version 5, 16-byte little-
endian binary at all three layers, 85 language / 45 family / 29
script bits). The build-time self-test makes the silent-bug class
impossible to recur. Consumer-side hand-curation **eliminated** —
exactly the goal the v3.2 design set out to reach.

The 41 → 27 movement in category 4 also leaves a useful audit
trail: the remaining 27 tokens at `popcount = 0` are mostly orphan
diacritics and historical edge cases that no living-language CLDR
exemplar would naturally cover. Worth a quick eyeball if anyone is
hunting for additional `extra_codepoints` candidates in other
locales.

# Checkpoint — language attribution of Apertus vocabulary

> **Date**: 2026-05-15 (last updated after char-tool v3.3.1 hotfix).
> **Scope**: state of the multi-language token-attribution pipeline
> spanning the four sub-subprojects under `02_2_tokenizer_implementation/`.
> **Status**: first end-to-end pass complete and integrated against
> char-tool schema v5 (v3.2 + v3.3 + v3.3.1 hotfix all shipped this
> session). Coverage **86.35 %** as of latest rebuild (up from
> 81.18 % under v3.1). Per char-tool manifest: **88 languages / 47
> families / 29 scripts**. All consumer-side workarounds (derived-map
> fallback, hardcoded `arb_Arab` patch) removed — `make_lookup()` is
> now 9 lines, reads the published manifest map directly. The char
> tool is still accepting new locales; consumers should re-run
> `build.py` to pick up additions.

## Executive summary

We have a deterministic, reproducible pipeline that takes the 131,072
tokens of the Apertus base vocabulary and assigns each one to zero,
one, or more languages based on (a) char-level admissibility under CLDR
exemplars and (b) per-language firing rates from a ~114 B-token
multi-language tokenisation corpus. With the current settings
(α = 0.5, δ = 1.0, min_count = 100, masked variant), **86.35 % of the
vocabulary lands in at least one language's main set** (up from
81.18 % under char-tool v3.1; the gain comes from v3.2 + v3.3.1 +
char-tool follow-on locales added late in the session). The
remaining 13.65 % is cleanly categorised by reason (substrate, byte
fragments, sub-threshold firing, or unmapped languages). Cross-script leakage is **zero** except for the one
linguistically-expected case (Han characters in both Chinese and
Japanese). Linguistic-relatedness patterns (Iberian Romance, South
Slavic, Czech-Slovak, Scandinavian, Persian-Urdu) all emerge cleanly
from the overlap matrix.

## 1. The tokenisation corpus — what we have on disk

### Sources and scope

8 source datasets, dataset-priority order in the worker:

```
fineweb_2_hq   →  epfml/FineWeb2-HQ              (multilingual, non-English)
fineweb_2      →  HuggingFaceFW/fineweb-2        (multilingual, non-English)
clean_wikipedia → HuggingFaceFW/clean-wikipedia  (multilingual incl. English)
europarl       →  Helsinki-NLP/europarl           (bitext)
paradocs       →  jhu-clsp/paradocs               (bitext)
fineweb_edu    →  HuggingFaceFW/fineweb-edu      (English)
fineweb_hq     →  epfml/FineWeb-HQ                (English)
dclm_edu       →  HuggingFaceTB/dclm-edu          (English)
```

Per canonical key (language × script × source-tag), we tokenize with
the Apertus-8B-2509 tokenizer and accumulate per-token firing counts up
to a 1 B-token cap. Output is
`02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz`, a
`(1934, 131072)` int64 matrix of firing counts.

### Run details

- **Run started**: 2026-05-13 on 8 × `c4-highcpu-192` workers in
  europe-west4 (2 on-demand + 6 spot). Total wall time ~4 h.
- **Grand total firings**: **114.37 B tokens**.
- **Distribution of sample sizes** across the 1,934 keys:

  | Σ firing count per key | # of keys |
  | --- | ---: |
  | ≥ 1 B (cap-hit) | **87** |
  | ≥ 100 M | 146 |
  | ≥ 1 M | 708 |
  | ≥ 100 k | 1,468 |
  | ≥ 1 (non-zero) | 1,930 |
  | 0 (failed) | 4 |

- **Sample size variability**: the 87 cap-hit keys carry ~88 B of the
  114 B total. The long tail of 1,843 low-resource keys collectively
  contributes ~26 B.

### Known limitations of the corpus

1. **Source priority is fixed**: each language's 1 B-token cap is
   filled from the **first** matching source in the priority list, then
   stops. English fell entirely to `clean_wikipedia` because FineWeb-2
   excludes English; FineWeb-Edu / FineWeb-HQ / DCLM-Edu (the actual
   English-heavy sources) were not used for the original English run.
2. **Mid-session domain-shift fix**: we re-ran English from
   `epfml/FineWeb-HQ` on 2026-05-15 to obtain a sample
   domain-matched to the FineWeb-2-HQ samples used for German, Greek,
   etc. Both English samples are now preserved:
   - `eng_Latn` — 1.005 B tokens from clean-wikipedia (kept as historical)
   - `eng_Latn_fineweb_hq` — 1.007 B tokens from FineWeb-HQ (canonical for cross-language comparisons)
3. **Token-budget-equal, not text-budget-equal**: 1 B German tokens
   covers fewer chars than 1 B English tokens because BPE fertility
   varies by language. PMI is unaffected (rates are per-token); only
   downstream "per-char" claims would need correction.

## 2. The char-membership tool — strict-rule reference layer

### Status

Now at **schema_version 5** in
`02_2_1_char_language_membership/artifacts/` (current as of
2026-05-15 after v3.2 + v3.3 ship). Three parallel masks per
codepoint, propagated to every Apertus vocab token via per-token AND
aggregation. Bit counts and the exact value of `N_LANG_BITS` are read
from the manifest at consumer build time, not hardcoded:

| level | column | bits in use (snapshot 2026-05-15 18:27 UTC) | what it answers |
| --- | --- | ---: | --- |
| script | `script_and` | **29** | which scripts admit every char |
| family | `family_and` | **47** | which language families admit every char |
| language | `bitmask_and` | **88** | which specific in-scope locales admit every char |

The bit counts are moving — the char tool is still adding locales.
Consumers should always read the count from `manifest["levels"]`
rather than rely on these snapshot values.

### Derivation

Strict rule throughout — per-locale CLDR exemplar characters (with the
four closures from PLAN_v3_HIERARCHICAL.md: case, NFD per-locale,
script-range fallback, post-fallback NFD) plus a curated substrate
codepoint list. No probabilistic content; the mask is a deterministic
function of CLDR.

### In-scope coverage

85 locales (per the current schema-v5 manifest). Spans the v3.1 list
(Latin 28, Cyrillic 5, Arabic 3, CJK 4, Greek 2, Hangul, Devanagari,
Hebrew, Thai, Armenian, Georgian, Bengali, Tamil, Telugu, Kannada,
Malayalam, Gujarati, Gurmukhi, Myanmar) plus v3.2's 18 additions plus
v3.3's 7 new-script locales (Amharic/Khmer/Sinhala/Lao/Tibetan/Odia/Dhivehi).

### Known gaps after v3.2 + v3.3

**v3.2 (schema 5) shipped 2026-05-15** — closed 26 of the original
34 unmapped cap-hit keys (the 4 silent-bug keys via the published
canonical_key map, plus 18 newly-added locales, plus 4 more recovered
via consumer-side derived-map fallback for two v3.2 manifest bugs
described below).

**v3.3 also shipped 2026-05-15** — added the 7 new-script locales
(Amharic / Khmer / Sinhala / Lao / Tibetan / Odia / Dhivehi). All seven
canonical keys (`amh_Ethi`, `khm_Khmr`, `sin_Sinh`, `lao_Laoo`,
`bod_Tibt`, `ory_Orya`, `div_Thaa`) now resolve in the manifest map.
**But: these locales contribute zero decoded-vocab coverage** —
the Apertus tokenizer wasn't trained on enough text in these scripts
for the chars to be a single-codepoint decoded token, so every token
in those scripts decodes as `partial_utf8` byte fragments. Masked
counts for all seven are 0; their content appears in Variant B
(unmasked) only.

**v3.3.1 hotfix (shipped 2026-05-15, same session)** — closed all
three consumer-flagged issues from the v3.2 integration report:

- `ell_Grek` and `gre_Grek` added to `canonical_key_to_char_tool_code`.
- `arb_Arab` resolved via new `iso_639_3_aliases: ["arb"]` on the `ar`
  entry.
- Urdu `ں` (U+06BA) now in the `ur` exemplar via a new
  `extra_codepoints` field in `languages.yaml`.
- Build-time self-test added: every language's primary
  `(iso_639_3, script)` pair must resolve through `iso_lookup`, build
  fails otherwise. Same class of bug can't recur silently.

**Remaining unmapped: 7 keys** (all genuine out-of-scope, no
char-tool fix possible):

| canonical_key | reason | actionable? |
| --- | --- | --- |
| `als_Latn` | Tosk Albanian — no Albanian locale in char tool | yes, add `sq` |
| `gmh_Latn` | Middle High German — historical, no CLDR | no |
| `lat_Latn` | Classical Latin — no living-language CLDR | no |
| `und_Cyrl`, `und_Grek`, `und_Kana`, `und_Mong` | undetermined-language samples — could be script-only-attributed | edge cases |

## 3. The promotion technique — multi-language PMI

### Why PMI (and not pairwise or max-pooling)

The methodology trade-off is in
`02_2_4_language_category_promotion/METHODOLOGY.md`. Short version:

- **Pairwise log-ratio** (en vs de): clean when there are 2 languages,
  doesn't scale.
- **Max-pooling**: a token must beat *every* sister by δ. Property: as
  scope grows, denominators can only rise → scores monotonically shrink
  → distinctive German tokens get killed by Dutch/Scandinavian
  co-firing.
- **PMI** (`p(t|L) / p(t)`): each language is scored independently
  against the corpus marginal. Closely-related sisters dilute the
  marginal slightly but don't crush the score. **Scales correctly with
  the number of in-scope languages**; that's the load-bearing reason.

### Definition

For each (token `t`, target language `L`) computed independently:

```
α = 0.5  (Laplace smoothing)

p_L(t)     = (count_L(t)        + α) / (total_L        + α·V)
p_marg(t)  = (count_marg(t)     + α) / (total_marg     + α·V)
PMI(t, L)  = log10( p_L(t) / p_marg(t) )
```

`count_marg(t)` and `total_marg` are summed over a chosen marginal
scope. For this pass: **the 87 cap-hit keys (Σ firings ≥ 1 B)**.

### Promotion decision

```
admissible_for_L(t)  =  (bitmask_and(t) bit_L set)
                       AND  popcount(bitmask_and) < N_LANG_BITS   # not substrate
                       AND  status not in T5 unknown
promote t to L's set  iff  count_L(t) ≥ min_count
                         AND  PMI(t, L) ≥ δ
                         AND  (Variant A only:) admissible_for_L(t)
```

Knobs and current values:

| knob | meaning | this pass |
| --- | --- | ---: |
| `α` | Laplace smoothing constant | 0.5 |
| `δ` | minimum log10 PMI to promote | **1.0** (≥10× over-represented vs corpus marginal) |
| `min_count` | minimum firing count in target language | **100** |
| marginal scope | which keys contribute to `count_marg` | 87 cap-hit (Σ ≥ 1 B) |

Two variants per language are emitted:

- **Variant A — masked**: with the char-admissibility filter. The canonical "main token set" of L.
- **Variant B — unmasked**: rate test alone. Audit of what the char mask removes.
- **Variant Δ — `B \ A`**: tokens promoted by rate but rejected by the char mask. Audit trail.

Spec: `02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md`.
Build script + outputs:
`02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`.

### A diagnostic-only second column

`per_token_pmi.parquet` includes a parallel `pmi_training` column
computed against a different marginal: `p_training(t) = Σ_L w_L · p_L(t)`,
where `w_L` is a per-language weight intended to reflect approximate
Apertus training share. When no weights are supplied the default is
`w_L = 1/K` (uniform), which gives the equal-weighted average of
per-language rates — **a different formula from canonical PMI**
(which is count-pooled). At our scale (cap-hit `total_L` within ~4 %
of each other) the two formulas produce numerically close PMI values,
but they are not algebraically identical. The column will be
recomputed with verified weights when per-locale Apertus training
shares are sourced.

## 4. Promotion results — main token sets per language

Using α = 0.5, δ = 1.0, min_count = 100, masked variant, marginal over
87 cap-hit keys (total marg = 87.84 B tokens):

### Coverage at a glance

| target | lang | masked tokens | unmasked | Δ | masked mass % | max PMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ell_Grek` | el | 1,479 | 1,495 | 16 | **86.94 %** | 1.94 |
| `fas_Arab` | fa | 2,785 | 2,856 | 71 | **83.30 %** | 1.94 |
| `jpn_Jpan` | ja | 3,222 | 3,999 | 777 | 72.10 % | 1.94 |
| `eng_Latn_fineweb_hq` | en | 19,339 | 19,479 | 140 | 56.17 % | 1.92 |
| `rus_Cyrl` | ru | 4,153 | 4,294 | 141 | 52.69 % | 1.93 |
| `deu_Latn` | de | 7,329 | 7,388 | 59 | 53.55 % | 1.94 |
| `eng_Latn` (wiki) | en | 19,009 | 19,291 | 282 | 47.42 % | 1.94 |
| `als_Latn` | unmapped | 0 | 1,227 | 1,227 | 0 % | — |

Full table at `analysis/main_token_sets_pmi/summary.tsv` (87 rows).

### Mass-coverage range

Across mapped cap-hit keys, masked-set mass coverage ranges from
~ 47 % (en-wiki, where high substrate share + low T0 dilute the
covered mass) to ~ 87 % (Greek, where the script is distinctive and
nearly all the corpus mass is in language-specific tokens).

### Observed asymmetries

- **Greek**: 1,479-token masked set covering 86.94 % of the Greek
  sample's mass. The Greek script's distinctiveness compresses the
  per-language vocabulary tightly. Top entries: ` και`, ` να`, ` το`,
  ` του`, ` της`.
- **German**: 7,329 tokens, 53.55 % mass. 103 of these are T0
  (ß-bearing — char-certified German with no premise needed).
- **English (FineWeb-HQ)**: 19,339 tokens, 56.17 % mass. T0 is
  structurally **zero** (English's CLDR exemplar is a strict subset of
  every other Latin locale's). All English-attribution rests on the
  rate test under the dataset-language premise.
- **English (wiki)**: 19,009 tokens, 47.42 % mass. The 9 % mass gap vs
  FineWeb-HQ English is the Wikipedia domain (more substrate, more
  numerals, more parenthetical citations).

### Output structure

```
analysis/main_token_sets_pmi/
├── tables/<key>__{masked,unmasked,delta}.txt   (87 keys × 3 = 261 files)
├── summary.tsv                                  (per-key counts + mass + PMI stats)
├── overlap_matrix.tsv                           (87×87 |masked_i ∩ masked_j|)
├── per_token_pmi.parquet                        (1.07 M rows: per token × key with pmi ≥ 0)
├── uncovered_tokens.tsv                         (17,888 uncovered post-latest, with reason per token)
├── weights_used.json                            (training weights applied)
└── manifest.json                                (provenance, knob values)
```

## 5. Overlap analysis — does the linguistic structure make sense?

### Methodology

`overlap_matrix.tsv` is the 87 × 87 matrix of pairwise intersections
between masked sets. Each cell `(i, j)` is the number of token IDs
present in both `masked[i]` and `masked[j]`. The diagonal is each set's
size. Jaccard normalises by the union.

### Top overlapping pairs (post-v3.3.1)

Refreshed against the current data — several new entries appear that
were impossible in the v3.1-era run because their languages weren't
mapped then. By Jaccard similarity (relative overlap):

| pair | Jaccard | overlap | linguistic interpretation |
| --- | ---: | ---: | --- |
| `eng_Latn` ↔ `eng_Latn_fineweb_hq` | **0.613** | 14,574 | same language, two domains — 39 % differ = domain-shift measurement |
| `hrv_Latn` ↔ `bos_Latn` | **0.550** | 1,299 | **NEW (bos mapped in v3.2)** — Croatian ↔ Bosnian BCMS, almost-same-language |
| `mkd_Cyrl` ↔ `srp_Cyrl` | 0.444 | 1,366 | Macedonian & Serbian-Cyrillic — South Slavic neighbours |
| `ind_Latn` ↔ `zsm_Latn` | **0.427** | 1,214 | **NEW (ms mapped in v3.2)** — Indonesian ↔ Standard Malay |
| `glg_Latn` ↔ `spa_Latn` | **0.385** | 3,381 | **NEW (gl mapped in v3.2)** — Galician ↔ Spanish (Iberian Romance) |
| `fas_Arab` ↔ `urd_Arab` | 0.384 | 1,301 | Persian & Urdu — Perso-Arabic shared script |
| `mkd_Cyrl` ↔ `bul_Cyrl` | 0.383 | 1,258 | Macedonian & Bulgarian — near mutually intelligible |
| `ces_Latn` ↔ `slk_Latn` | 0.320 | 946 | Czech & Slovak — closer than most language pairs |
| `glg_Latn` ↔ `por_Latn` | **0.304** | 2,566 | **NEW** — Galician ↔ Portuguese |
| `bul_Cyrl` ↔ `rus_Cyrl` | 0.287 | 1,446 | South ↔ East Slavic Cyrillic |
| `por_Latn` ↔ `spa_Latn` | 0.271 | 2,616 | Iberian Romance |
| `bul_Cyrl` ↔ `srp_Cyrl` | 0.269 | 969 | another South Slavic Cyrillic pair |
| `slv_Latn` ↔ `hrv_Latn` | 0.267 | 680 | West/South Slavic Latin |
| `nob_Latn` ↔ `dan_Latn` | 0.244 | 860 | Norwegian Bokmål ≈ Danish |
| `rus_Cyrl` ↔ `ukr_Cyrl` | 0.234 | 1,216 | East Slavic |
| `nld_Latn` ↔ `afr_Latn` | **0.233** | 978 | **NEW (af mapped in v3.2)** — Dutch ↔ Afrikaans (mutually intelligible) |
| `mar_Deva` ↔ `npi_Deva` | **0.673** | 829 | **NEW (mr/ne mapped in v3.2)** — Marathi ↔ Nepali (Indic Devanagari) |
| `npi_Deva` ↔ `hin_Deva` | **0.640** | 941 | **NEW** — Nepali ↔ Hindi |
| `mar_Deva` ↔ `hin_Deva` | **0.639** | 945 | **NEW** — Marathi ↔ Hindi |
| `kir_Cyrl` ↔ `kaz_Cyrl` | **0.299** | 362 | **NEW (ky/kk mapped in v3.2)** — Kyrgyz ↔ Kazakh (Turkic Cyrillic) |
| `cmn_Hani` ↔ `jpn_Jpan` | **0.221** | 1,061 | Han chars shared — the one cross-script overlap, linguistically correct |
| `tur_Latn` ↔ `azj_Latn` | 0.218 | 583 | Turkic |
| `dan_Latn` ↔ `swe_Latn` | 0.197 | 739 | Scandinavian |

By **absolute overlap** size, the top is now dominated by the
v3.3.1-recovered Arabic varieties:

| pair | overlap | Jaccard | note |
| --- | ---: | ---: | --- |
| `ary_Arab` ↔ `arb_Arab` | **6,784** | **0.918** | **highest non-domain pair** — Moroccan Arabic ↔ Standard Arabic, near-complete vocab overlap. `arb_Arab` was previously unmapped; v3.3.1 made this pair queryable. |
| `glg_Latn` ↔ `spa_Latn` | 3,381 | 0.385 | (above) |
| `por_Latn` ↔ `spa_Latn` | 2,616 | 0.271 | (above) |
| `glg_Latn` ↔ `por_Latn` | 2,566 | 0.304 | (above) |
| `eng_Latn` ↔ `eng_Latn_fineweb_hq` | 14,574 | 0.613 | (above, the domain pair) |

### Sanity checks the data passes

- **Cross-script overlap = exactly 1 pair** (`cmn_Hani` ↔ `jpn_Jpan`),
  which is linguistically correct (Han characters are admissible in
  both Chinese and Japanese exemplars). Every other cross-script pair
  has zero overlap. **The char mask is doing its job perfectly.**
- **Within-script-family overlap tracks linguistic distance**: Iberian
  Romance (es/pt) and South Slavic Cyrillic (mkd/srp/bul) cluster
  tightly; English vs other Germanic locales overlap minimally (en is a
  distant Germanic outlier vocabulary-wise).
- **Same-language different-source = highest non-substrate overlap**:
  the two English samples share 61 % of their masked sets, which is the
  cleanest empirical anchor for what "the same language" looks like
  under our test.

### Interpretation guide for downstream

Pairs with **Jaccard > 0.20** indicate languages whose embedding-norm
analyses will see significant shared vocabulary — the same token rows
in E/U contribute to both centroids. For the embedding diagnostic:

- closely-related-language pairs (Czech-Slovak, Mac-Bulg, Norwegian-Danish):
  shared vocab is substantial, **per-language centroids will be
  correlated**.
- script-aggregate diagnostics (e.g., "all Cyrillic") will pool ~6,000
  unique Cyrillic-script tokens with overlap structure preserved.

## 6. Coverage audit — what's covered, what isn't, and why

### Headline (latest rebuild, char tool at 88 languages)

- **Vocab size**: 131,072
- **Covered (in at least one masked set)**: **113,184 (86.35 %)** —
  up 6,780 tokens from v3.1.
- **Uncovered**: **17,888 (13.65 %)** — down from 24,668 in v3.1.
- Coverage trajectory: v3.1 81.18 % → v3.2 85.54 % → v3.3.1 85.55 %
  (+14 from Urdu fix) → latest 86.35 % (+1,053 from new locales the
  char tool added after v3.3.1).

### Uncovered tokens by category

Each token is classified by the first reason that applies (priority
order):

| # | category | v3.1 count | **latest count** | example tokens |
| ---: | --- | ---: | ---: | --- |
| 1 | `is_special` (Apertus reserved) | 990 | **990** | `<>`, ` <>`, sentinel ids |
| 2 | T5 unknown standalone (raw byte fragments, `decoded = None`) | 1,448 | **1,448** | id 1224 fires **1.14 B** times — byte-level workhorse |
| 3 | substrate (popcount = `N_LANG_BITS`, universal) | 4,014 | **4,014** | `,`, `.`, ` `, `\n` — by-design excluded |
| 4 | no-locale-admits-chars (popcount = 0 with decoded text) | 42 | **22** | Urdu `ں`-bearing tokens moved to `urd_Arab` in v3.3.1; what remains is orphan-diacritic edge cases |
| 5 | fires only in unmapped cap-hit langs | 97 | **0** | bucket eliminated — every cap-hit canonical key now has a char-tool mapping after the locale additions |
| 6 | below `min_count` in every mapped cap-hit lang | 3,655 | **3,717** | low-resource langs (Tatar, Esperanto, Pashto). Counts shifted slightly after coverage_audit was fixed to exclude unmapped cap-hit keys from the "did it fire enough?" check (~100 tokens moved cat 7 → cat 6) |
| 7 | **PMI < δ for every language** | 14,188 | **7,463** | down by **6,725** vs v3.1 — newly mapped languages claim about half of these short shared tokens |
| 8 | never fires anywhere | 134 | **134** | dead vocab |
| 9 | fires only in non-cap-hit langs | 100 | **100** | unchanged — non-cap-hit policy hasn't changed |

### Reading

**Category 7 dominates the uncovered set** (**41.7 %, 7,463 of 17,888
after the latest char-tool refresh**). These are short shared tokens —
single Latin letters and 2-3 char fragments that fire hundreds of
millions of times each, distributed across many languages. No single
language reaches a 10× preference for them, so they don't promote at
`δ = 1.0`. They're the linguistic infrastructure that every Latin-
script language uses — comparable to how no language can "claim" the
comma. Across the session this bucket shrank from **14,188 (v3.1)
→ 8,546 (v3.2) → 7,463 (post-v3.3.1 + char-tool follow-ons)** as each
locale addition absorbed its share.

**Category 3** (substrate, 4,014 tokens) is by-design excluded — same
reasoning, but applied at the char level rather than the rate level.

**Categories 2 + 7 are where the "tokenizer was trained on uneven
distribution" effect shows up**: heavy traffic, no language-specific
attribution. Combined: **8,911 tokens (~50 % of uncovered)** carry
massive firing counts but no language-specific identity.

### Char-tool gap surfaced by the audit — now fixed

Category 4 (originally 42 → 41 tokens, `popcount = 0` with non-empty
decoded text) was nearly all Urdu tokens containing `ں` (Arabic
letter noon ghunna, U+06BA), a standard letter in Urdu that wasn't in
the `ur` CLDR exemplar. **Closed in v3.3.1 via the new
`extra_codepoints` field in `languages.yaml` (seeded with `["U+06BA"]`
on the `ur` entry).** 14 Urdu tokens (` میں`, `وں`, ` ہیں`, ` نہیں`,
` انھوں`, etc.) moved out of the `no_locale_admits_chars` bucket into
`urd_Arab`'s masked set. Category 4 dropped 41 → 27 (remaining 27 are
orphan diacritics and historical edge cases).

### Knob-sensitivity preview

Approximate coverage shift per knob:

| change | expected coverage shift |
| --- | ---: |
| add 34 unmapped cap-hit langs to char tool (v5) | +1–3 % |
| lower δ from 1.0 to 0.5 (≥3× preference) | +5–10 % |
| lower min_count from 100 to 10 | +2 % |
| fix Urdu `ں` admissibility | +0.03 % |
| include lower-sample languages in marginal | +0.5 % |

The dominant knob is **δ**: most of the uncovered mass is in category
7, which is exactly what raising the bar to 10× cuts off. Lowering δ
to 0.5 would promote most category-7 tokens into the language with the
slightly-highest rate, but at the cost of making the promotion less
language-distinctive.

## 7. Where we are vs where we're going

### What's done and reproducible

- Full deterministic build pipeline; rebuild ≈ 25 s.
- All artifacts present on disk (some are gitignored due to size —
  `histogram_matrix.npz`, `per_token_pmi.parquet`, the char-tool
  parquets — and would need explicit-add or LFS to track):
  - 87 × 3 masked / unmasked / delta token-set files
  - per-token PMI audit table (1.07 M rows)
  - 87×87 overlap matrix
  - coverage classification of every uncovered token
  - provenance manifests (input MD5s + char-tool schema version + knob values)
- All findings cited in this checkpoint have specific file paths +
  line counts.

### v3.2 + v3.3 + v3.3.1 integration outcomes (all shipped 2026-05-15)

Three char-tool releases landed and are integrated. The consumer-side
`build.py` was rewritten to consume the new manifest, then simplified
again after the v3.3.1 hotfix:

- The 50-entry hardcoded `ISO_639_3_TO_BCP47` dict was **deleted**.
  `make_lookup()` is now **9 lines** — a direct read of the published
  `canonical_key_to_char_tool_code` map plus a source-tag-suffix
  fallback. No derived-map fallback, no hardcoded `arb_Arab` patch,
  no consumer-side Urdu workaround. Removed mid-session once v3.3.1
  closed the manifest gaps.
- All 4 originally-silent-bug keys (`srp_Cyrl`, `lvs_Latn`,
  `ekk_Latn`, `cmn_Hani`) plus the two v3.2-flagged keys (`ell_Grek`,
  `arb_Arab`) all resolve directly via the manifest map.
- 18 of the 26 newly-mapped locales produce sane masked sets at
  first try; content spot-checks confirm linguistic appropriateness
  (Swahili Bantu function words, Marathi Devanagari, Welsh `ll`/`dd`,
  Belarusian distinctive `ў`, etc.).
- The new `category_or` uint8 column in `token_language_bitmask.parquet`
  works exactly as specified. Verified for the eng_Latn_fineweb_hq
  masked set: 95.1 % `letters_only` (clean text), 4.9 % `letters_with_punct`
  (code-mixed). For the PMI-below-δ uncovered bucket: 89 % letters_only,
  11 % punct-mixed.

### Two v3.2 manifest bugs found and consumer-side-patched

| bug | symptom | consumer workaround |
| --- | --- | --- |
| `ell_Grek` missing from `canonical_key_to_char_tool_code` | Greek silently treated as "unmapped" → empty masked set | derive map fallback from `languages` list using `iso_639_3` + script |
| `arb_Arab` missing | `ara_Arab → ar` is in the map but the individual ISO 639-3 alias isn't | added hardcoded `arb_Arab → ar` patch + suggested `ar` entry should have `iso_639_3_aliases: ["arb"]` |

**Both fixed in v3.3.1 (shipped 2026-05-15 same session).** See
`02_2_1_char_language_membership/v3_2_INTEGRATION_REPORT_20260515.md`
for the post-v3.3.1 verification — the consumer-side fallback +
hardcoded patch have been removed; `make_lookup()` is now 9 lines.

### v3.3 + v3.3.1 outcomes

- **v3.3** (shipped 2026-05-15) added the 7 new scripts — Ethiopic
  (Amharic), Khmer, Sinhala, Lao, Tibetan, Odia, Thaana (Dhivehi).
  All seven canonical keys (`amh_Ethi`, `khm_Khmr`, `sin_Sinh`,
  `lao_Laoo`, `bod_Tibt`, `ory_Orya`, `div_Thaa`) now resolve in the
  manifest map. **Zero decoded-vocab coverage gain** — these scripts
  are byte-fragmented in the Apertus tokenizer; chars decode as
  `partial_utf8`. Masked counts for all seven = 0; their content
  appears in Variant B only. This is an Apertus-vocab fact, not a
  v3.3 defect.
- **v3.3.1** (same-day hotfix) closed the two v3.2 manifest gaps
  (`ell_Grek`, `arb_Arab`) plus the Urdu `ں` exemplar gap, and added
  a build-time self-test on primary `(iso_639_3, script)` resolution.
  +14 tokens covered (the Urdu ں-bearing set now AND-attributes to
  `ur`). Consumer-side workarounds removed.

### Re-running

Single command, deterministic:

```bash
cd subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution
python3 analysis/main_token_sets_pmi/build.py
```

### Parameter sweep TODO (post-v5)

`METHODOLOGY.md`'s comparison harness is the natural next workstream:
sweep `δ ∈ {0.5, 1.0, 1.5, 2.0}` × `min_count ∈ {10, 100, 1000}` × marginal
scope ∈ {cap-hit-87, ≥ 100 M (146), all-non-zero (1,930)}, and report
per-language mass-coverage curves so the eventual knob choice for the
embedding diagnostic is from evidence not default.

### Training-weighted PMI (deferred)

Sourcing verified per-locale Apertus training shares (paper / model
card / FineWeb proxy counts), then re-running with
`--training-weights weights.json` populates the diagnostic
`pmi_training` column. Surfaces tokens whose equal-weighted and
training-weighted PMI disagree (high-resource languages where the
training-weighted marginal absorbs more of their mass).

### Phase 3 hand-off (downstream)

The 87 `<key>__masked.txt` files are the candidate input for the
embedding diagnostic
(`03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/`
and its non-Greek counterparts). The Greek-tokens-app legacy interface
(`base_greek_tokens.jsonl`) maps cleanly to `ell_Grek__masked.txt` —
backward-compatibility verified.

## Index of artifacts

| artifact | path |
| --- | --- |
| **Tokenisation corpus** | `02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz` |
| Per-token char masks (v4) | `02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet` |
| **PMI build script** | `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/build.py` |
| Spec | `02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md` |
| Methodology (F vs W) | `02_2_4_language_category_promotion/METHODOLOGY.md` |
| **Per-language masked sets** | `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/tables/<key>__masked.txt` |
| Overlap matrix | same dir / `overlap_matrix.tsv` |
| Per-token PMI audit | same dir / `per_token_pmi.parquet` |
| Uncovered-tokens classification | same dir / `uncovered_tokens.tsv` |
| Coverage audit script | same dir / `coverage_audit.py` |
| Overlap analysis script | same dir / `overlap_analysis.py` |
| Char-tool feedback report | `02_2_1_char_language_membership/FEEDBACK_FROM_PMI_PROMOTION_CONSUMER_20260515.md` |

End of checkpoint.

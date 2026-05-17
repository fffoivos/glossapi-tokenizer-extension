# C3 cutoff analysis — first-draft decision report

**Date**: 2026-05-17.

**Status**: Draft. C3 is the current **combined-Greek baseline**. This
report documents C3 cutoff evidence; it does not decide whether the next
tokenizer path should be an extension of an existing tokenizer or a
fresh modern-Greek tokenizer followed by a polytonic extension.

**Scope**: Recommend a cutoff for C3 given (1) comparable-language
empirical vocab footprints in Apertus, (2) fertility curves on three
verified-clean held-out slices, (3) how the glossary morphological
categorization of added tokens evolves per +1024 step, (4) how the
char-language-membership of those tokens evolves per +1024 step.

---

## TL;DR

For the C3 combined-Greek baseline under the design constraint
"Greek-payload total ≤ ~13k (the English-unique anchor) + total vocab
divisible by 256":

**Definition used throughout:**
**Greek-payload total** = (Apertus base Greek-PMI tokens = 1,479) +
(GREEK-bucket added tokens at the cutoff = glossary-says-Greek AND
char-mask agrees the token is Greek-script). This is the apples-to-
apples comparison against the per-language PMI anchors in §1. The
"added N" column is the raw cutoff including ~1 % non-Greek
structural/ambiguous tokens.

| pick | added N | Greek-payload added | Greek-payload total (base + added) | total vocab | fertility | unused | note |
|---|---:|---:|---:|---:|---:|---:|---|
| **recommended** | **11,264** | **11,167** | **12,646** | **142,336** | **1.47** | **5.5 %** | best fertility inside cap; 99 % Greek-payload share; 256-aligned |
| tight-to-cap | 11,520 | ~11,420 | ~12,899 | 142,592 | ~1.464 (interp.) | ~6.0 % | closest 256-step to the 13k cap; needs a new variant build |
| conservative | 8,192 | 8,128 | 9,607 | 139,264 | 1.55 | 3.5 % | ~ French footprint; 4 % below 11k fertility |
| German-tier | 6,144 | 6,095 | 7,574 | 137,216 | 1.63 | 2.4 % | ~ German footprint |

This is for C3 only. Any future modern/polytonic tokenizer path should
reanchor against its own training data, dedup state, and polytonic eval
slices before a cutoff grid is chosen.

> Note on the 13k cap. The ~13k figure is the unique-vs-non-English
> count of the `eng_Latn` PMI slice taken alone. The union of the two
> English slices (`eng_Latn` + `eng_Latn_fineweb_hq`) gives a larger
> unique-vs-non-English count (~16k). The 13k cap is therefore a
> **conservative design choice** — anchoring to the smaller of the two
> defensible English-unique counts — not a single empirical fact.
> Any cutoff up to ~17k (the larger English-total empirical anchor)
> can be defended; the 13k pick keeps Greek safely below English.

---

## (1) Comparable-language vocab footprints in Apertus

Source: `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv`.
Per-language PMI-promoted token counts after the char-admissibility
mask (`masked_count`):

| Language | PMI tokens | mass % captured | script | notes |
|---|---:|---:|---|---|
| **English (FineWeb-2)** | **19,009** | 47.4 % | Latn | ~13k uniquely English; ~6k overlap (sum-of-pairwise) with other Latin-script langs (cumulative bounds, not distinct) |
| English (FineWeb-HQ) | 19,339 | 56.5 % | Latn | same language; 14,574 overlap with FineWeb-2-en |
| French | 9,694 | 58.7 % | Latn | biggest pairwise share with English = 1,333 |
| German | 7,329 | 53.6 % | Latn | |
| Arabic | 7,146 | 80.1 % | Arab | script-isolated |
| Spanish | 6,714 | 57.5 % | Latn | |
| Portuguese | 5,549 | 52.4 % | Latn | |
| Italian | 4,712 | 50.2 % | Latn | |
| Korean | 4,438 | 76.7 % | Hang | script-isolated (largest among isolated) |
| Russian | 4,153 | 52.7 % | Cyrl | |
| Japanese | 3,222 | 72.1 % | Jpan | script-isolated |
| Dutch | 3,045 | 46.4 % | Latn | |
| Mandarin | 2,650 | 65.2 % | CJK | |
| Polish | 2,570 | 49.8 % | Latn | |
| Swedish | 2,212 | 46.5 % | Latn | |
| Czech | 2,058 | 44.4 % | Latn | |
| Turkish | 1,833 | 47.7 % | Latn | |
| Finnish | 1,767 | 39.9 % | Latn | |
| Vietnamese | 1,564 | 70.6 % | Latn | |
| **Greek** | **1,479** | **86.9 %** | Grek | script-isolated; every token is exclusively Greek |
| Hindi | 1,388 | 86.0 % | Deva | |

**Key implication**. "Greek vocab size should match English" is a real
question with a real answer: English's empirical footprint is ~13–19k
PMI-promoted tokens, NOT 88k. (88k is the Latin-script *admissibility*
ceiling, shared by ~30 languages — not anyone's exclusive allocation.)
See
`subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/english_review/membership_report.md`
§Popcount distribution for the full derivation: 69 % of English mass
lives in tokens admissible by **every** Latin-script locale, only
0.07 % of mass is uniquely admissible by English alone.

**Anchor points for the C3 cutoff**:

| target language | empirical tokens | needed C3 cutoff (added) | sweep-data cutoff |
|---|---:|---:|---:|
| match Korean (largest script-isolated peer) | 4,438 | +2,959 | **3,072** |
| match German | 7,329 | +5,850 | **6,144** |
| match French | 9,694 | +8,215 | **8,192** |
| match "uniquely English" (~13k) | 13,000 | +11,521 | **11,264** |
| match English total PMI (~19k) | 19,009 | +17,530 | **17,408** |

Constraint #1 says we cap at "English unique (~13k) or less", so we
work within added ≤ ~11,521.

---

## (2) Fertility on three verified-clean held-out slices

Source: `02_1_3_fertility_evaluation/` (the sweep at
`~/runs/c3_cutoff_eval_20260511/fertility_c3_full_25_clean_20260511/`
on the gcloud instance).

Slices:
- `virgin_hplt` (10,000 docs) — HPLT docs whose `source_doc_id` is not
  in the C3 mix; guaranteed unseen by C3 BPE
- `C3_val_clean` (7,624 docs) — C3 val with the 30 train-overlap rows
  removed
- `C3_test_clean` (7,246 docs) — C3 test minus 36 train-overlap rows

Averaged across the three clean slices, restricted to the 1k–11k range:

| cutoff | greek_word fertility ↓ | Δfert vs prev | chars/tok | tokens/byte | added-vocab util | unused added |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (apertus_base) | 2.413 | — | 2.59 | 0.171 | — | — |
| 1,024 | 2.089 | −0.324 | 2.93 | 0.151 | 0.991 | 10 (1.0 %) |
| 2,048 | 1.934 | −0.155 | 3.14 | 0.141 | 0.991 | 19 (0.9 %) |
| 3,072 | 1.828 | −0.106 | 3.29 | 0.135 | 0.987 | 39 (1.3 %) |
| 4,096 | 1.747 | −0.081 | 3.41 | 0.130 | 0.982 | 73 (1.8 %) |
| 5,120 | 1.683 | −0.064 | 3.52 | 0.126 | 0.980 | 104 (2.0 %) |
| 6,144 | 1.631 | −0.051 | 3.61 | 0.123 | 0.976 | 148 (2.4 %) |
| 7,168 | 1.589 | −0.042 | 3.69 | 0.121 | 0.971 | 207 (2.9 %) |
| 8,192 | 1.553 | −0.036 | 3.76 | 0.118 | 0.965 | 284 (3.5 %) |
| 9,216 | 1.523 | −0.030 | 3.83 | 0.116 | 0.959 | 380 (4.1 %) |
| 10,240 | 1.495 | −0.028 | 3.88 | 0.115 | 0.952 | 490 (4.8 %) |
| **11,264** | **1.471** | **−0.024** | **3.93** | **0.114** | **0.945** | **620 (5.5 %)** |

For reference past the cap (full sweep continues to 25,600):

| cutoff | fertility | unused | note |
|---:|---:|---:|---|
| 12,288 | 1.450 | 777 (6.3 %) | exceeds 13k cap |
| 15,360 | 1.397 | 1,382 (9.0 %) | XLM-R-class |
| 17,408 | 1.370 | 1,892 (10.9 %) | matches English total |
| 20,480 | 1.335 | 2,914 (14.2 %) | |
| 25,600 | 1.291 | 4,995 (19.5 %) | C3 raw vocab; ~20 % unused on eval |

**Elbow analysis (inside the cap)**.

Marginal fertility gain per +1024 step decays smoothly from −0.155 at
1k→2k to −0.024 at 10k→11k. At every step inside the cap the marginal
gain is **larger than any step past 16k in the full sweep** (where
gains drop below 0.015). The deep elbow lives outside our cap; inside,
fertility is still gaining usefully.

**Implication**: under the cap, push as high as the constraint allows.

---

## (3) Glossary morphological categorization × cutoff

Source: `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl`
— the corrected Gemini-pass per-token labels (category, morphological
structure, lexical role, confidence). Each cutoff prefix is the first
`N` added tokens by id (BPE merge order).

### Category × cutoff

| cutoff | greek_word | greek_fragment | greek_morpheme | greek_acronym | proper_noun | Greek subtotal |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 204 | 457 | 342 | 4 | 1 | 1,008 |
| 2,048 | 496 | 906 | 609 | 9 | 6 | 2,026 |
| 3,072 | 830 | 1,325 | 859 | 16 | 11 | 3,041 |
| 4,096 | 1,201 | 1,727 | 1,084 | 21 | 25 | 4,058 |
| 5,120 | 1,565 | 2,126 | 1,330 | 22 | 35 | 5,078 |
| 6,144 | 1,969 | 2,515 | 1,540 | 25 | 46 | 6,095 |
| 7,168 | 2,368 | 2,901 | 1,758 | 27 | 57 | 7,111 |
| 8,192 | 2,796 | 3,284 | 1,944 | 30 | 74 | 8,128 |
| 9,216 | 3,256 | 3,624 | 2,136 | 37 | 86 | 9,139 |
| 10,240 | 3,711 | 3,987 | 2,311 | 39 | 106 | 10,154 |
| **11,264** | **4,182** | **4,331** | **2,479** | **48** | **127** | **11,167** |

Growth ratios 1k → 11k:
- `greek_word`: **×20.5** (the fastest grower — whole inflected forms)
- `greek_fragment`: ×9.5 (early phonotactic pieces saturate)
- `greek_morpheme`: ×7.2 (steady — inflectional endings/prefixes)
- `greek_acronym`: ×12 (rare — ΔΙΑΒΑΣΤΕ, ΚΟΙΝΟ, etc.)
- `proper_noun`: ×127 absolute but tiny — long tail of named entities

Trajectory in plain words:
- **Early cutoffs (1k–4k)**: fragment-heavy. The BPE merges that come
  out first are subword pieces (single Greek letters with prefixes,
  bigrams, common inflectional endings).
- **Mid cutoffs (5k–10k)**: `greek_word` becomes dominant. Whole
  inflected forms enter the vocab (Greek conjugations, nominal-case
  forms).
- **Late cutoffs (past 11k, outside our cap)**: `proper_noun` and
  `greek_acronym` accumulate. By 25k, 505 proper nouns + 161 acronyms
  vs 127 + 47 at 11k.

At 11,264 we already have **4,182 distinct whole-word Greek tokens**.
That's the meaningful payload — the morphological skeleton, the most
common inflected forms, and 2,479 inflectional pieces filling in the
ending layer.

### Structural fingerprint at 11k (glossary `greek_morphology.structure`)

| structure | 1k | 4k | 8k | **11k** | meaning |
|---|---:|---:|---:|---:|---|
| `fragment` | 358 | 1,344 | 2,522 | **3,332** | subword piece |
| `stem+ending` | (low) | 1,108 | 2,481 | **3,432** | inflected form |
| `stem` | 174 | 527 | 952 | **1,226** | bare stem |
| `prefix+stem+ending` | 13 | 151 | 437 | **718** | prefix + stem + inflection |
| `stem_partial` | 261 | 535 | 850 | **1,015** | partial stem |
| `prefix+stem_partial` | 41 | 137 | 265 | **354** | prefix + partial stem |
| `ending` | 131 | 328 | 528 | **638** | bare inflectional ending |
| `prefix+stem` | 7 | 39 | 87 | **121** | prefix + stem |
| `prefix` | 21 | 64 | 110 | **132** | bare derivational prefix |

At 11k the vocab now contains most productive Greek inflectional
endings as standalone tokens, plus the morphological skeleton (stems,
fragments). Whole inflected words (`stem+ending` + `prefix+stem+ending`
+ `prefix+stem+stem+ending` etc.) sum to ~4,200 — matches the
`greek_word` count above.

### Lexical-role fingerprint at 11k

| lexical role | 1k | 4k | 8k | **11k** | examples |
|---|---:|---:|---:|---:|---|
| `function_word` | 116 | 326 | 521 | **637** | και, που, στην, τους |
| `loanword` | 7 | 51 | 132 | **193** | Greek-rendered loans |
| `proper_noun` | 0 | 16 | 60 | **97** | Greek-spelt names |
| `abbreviation` | 6 | 22 | 35 | **52** | π.χ., κλπ. |
| `none` | 879 | 3,643 | 7,380 | **10,188** | residual greek-but-not-tagged |

Function-word coverage at 11k is already strong — 637 of the most
common Greek function words have whole-token entries (matches what
Apertus-base would only have as 2–3-token compositions).

### Confidence × cutoff

The Gemini-pass per-token confidence stays high across cutoffs (≥0.9
share is the dominant bucket); no quality cliff at any point in the
1k–11k range. Full numbers in
`~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/cutoff_grid/distribution_at_<N>.json`.

---

## (4) Language-mapping × cutoff (char-mask + glossary cross-product)

Source: `02_2_1_char_language_membership/artifacts/char_language_bitmask.parquet`
(strict CLDR-evidence char masks) crossed with the glossary categories.

Combined per-token "function" label:
- **GREEK**: glossary in {greek_word, greek_fragment, greek_morpheme,
  greek_acronym, proper_noun} AND char-mask in {`el_or_polyton`,
  `single:el-polyton`, `el_plus_others`}. The payload.
- **USEFUL_STRUCTURAL**: glossary in {table_separator, punctuation_run,
  escaped_character_run, math_symbol, dingbat_or_symbol,
  whitespace_only, url_or_path, code_identifier, latin_acronym,
  latin_abbreviation, unit_or_measure, postscript_glyph}. MD / code /
  URL / structural patterns that legitimately appear in the GlossAPI
  corpus.
- **NOISE**: glossary in {mojibake, encoding_artifact,
  control_or_invisible}. Real undesirable artifacts.
- **AMBIGUOUS**: glossary in {mixed_script_token, latin_fragment,
  latin_word}. Often punctuation+Greek patterns like `.Ε` or `,τι`.

### Function-level budget × cutoff

| cutoff | GREEK | USEFUL | NOISE | AMBIG | GREEK % | NOISE % |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,008 | 11 | 4 | 1 | 98.4 % | 0.39 % |
| 2,048 | 2,026 | 12 | 4 | 6 | 98.9 % | 0.20 % |
| 3,072 | 3,041 | 17 | 6 | 8 | 99.0 % | 0.20 % |
| 4,096 | 4,058 | 20 | 6 | 12 | 99.1 % | 0.15 % |
| 5,120 | 5,078 | 22 | 6 | 14 | 99.2 % | 0.12 % |
| 6,144 | 6,095 | 27 | 7 | 15 | 99.2 % | 0.11 % |
| 7,168 | 7,111 | 29 | 9 | 19 | 99.2 % | 0.13 % |
| 8,192 | 8,128 | 32 | 11 | 21 | 99.2 % | 0.13 % |
| 9,216 | 9,139 | 40 | 11 | 26 | 99.2 % | 0.12 % |
| 10,240 | 10,154 | 43 | 13 | 30 | 99.2 % | 0.13 % |
| **11,264** | **11,167** | **49** | **15** | **33** | **99.1 %** | **0.13 %** |

**Noise stays at ~0.13 % flat across the entire range.** Adding more
tokens doesn't introduce more noise; the cleaner held it down at every
prefix.

### Marginal per +1024 step

| step | Δgreek | Δuseful | Δnoise | Δambig |
|---|---:|---:|---:|---:|
| 0 → 1k | 1,008 | 11 | 4 | 1 |
| 1k → 2k | 1,018 | 1 | 0 | 5 |
| 2k → 3k | 1,015 | 5 | 2 | 2 |
| 3k → 4k | 1,017 | 3 | 0 | 4 |
| 4k → 5k | 1,020 | 2 | 0 | 2 |
| 5k → 6k | 1,017 | 5 | 1 | 1 |
| 6k → 7k | 1,016 | 2 | 2 | 4 |
| 7k → 8k | 1,017 | 3 | 2 | 2 |
| 8k → 9k | 1,011 | 8 | 0 | 5 |
| 9k → 10k | 1,015 | 3 | 2 | 4 |
| 10k → 11k | 1,013 | 6 | 2 | 3 |

Each +1024 step gets ~1,011–1,020 actual Greek tokens and ~3–8
structural/ambiguous/noise. The non-Greek cost per step is bounded ≤
13 and doesn't rise with cutoff.

### What the non-Greek tokens at 11,264 actually are

**NOISE (15 tokens, 0.13 %)** — enumerated exactly from
`artifacts/classified_added_tokens.jsonl` at 11,264:
```
''  (empty / partial-UTF8 byte fragments, encoding_artifact)  ×9
'ε', 'ο', 'ο'  (variant-byte Greek letters tagged encoding_artifact)  ×3
'%CE'  (URL-encoded prefix, encoding_artifact)                ×1
'·\n\n'  (control + whitespace, control_or_invisible)         ×1
'́'  (combining acute alone, encoding_artifact)               ×1
```
Note: `%CF` is in the vocab at id 138,196 but is classified
`url_or_path`, not `encoding_artifact` — it's in USEFUL_STRUCTURAL, not
NOISE. Same UTF-8-byte-prefix family as `%CE`; glossary tagging is
inconsistent across the two but neither is removable safely (both
recur in URL-bearing text).

**USEFUL_STRUCTURAL (49 tokens, 0.4 %)** — relevant corpus markup:
- 19 punctuation runs: `..................`, `....................`,
  `.....`, `.Α`, `.Ο`, `.Τ`, `.Ν` — period-leader patterns from
  TOC/page-number leaders in academic PDFs converted to MD
- 9 table separators: `|-----`, `|--------------------`,
  `|-----|-----` — MD table syntax
- 6 math symbols: `∆`, `Ω`, `∧`
- 5 escape runs: `\_\_`, `\_\_\_\_`, `\_\_\_\_\_\_\_\_` — MD escapes
- 1 URL fragment: `.gr` — the Greek domain
- Plus `\n`, `%CF`, `·`, `΄`, `code_identifier`, `unit_or_measure`,
  `latin_acronym` (×3)

**AMBIGUOUS (33 tokens, 0.3 %)** — almost all are punctuation+Greek:
- 28 period+Greek tokens: `.Ε`, `.Π`, `.Σ`, `.Δ`, `.Κ`, … (sentence-
  start patterns)
- `,τι` (from `ό,τι` "whatever"), `/και` (slash+και), `/ΕΚ`
- 5 hyphen+capital: `-Α`, `-Κ`, `-Π`, `-Μ`, `-Σ` (compound name patterns)
- 2 code-tag fragments: `-missing`, `-decoded`

The classifier's "AMBIGUOUS" bucket is conservative — these tokens
function as Greek-context tokens in practice. If you reclassify them as
USEFUL the noise rate stays at 0.13 % and USEFUL+AMBIG rises to 0.73 %.

### Total useful-for-corpus budget per cutoff

If GREEK + USEFUL_STRUCTURAL + AMBIGUOUS counts as "useful for the
Greek corpus" (everything except true noise):

| cutoff | useful | useful % | noise | noise % |
|---:|---:|---:|---:|---:|
| 1,024 | 1,020 | 99.61 % | 4 | 0.39 % |
| 4,096 | 4,090 | 99.85 % | 6 | 0.15 % |
| 8,192 | 8,181 | 99.87 % | 11 | 0.13 % |
| **11,264** | **11,249** | **99.87 %** | **15** | **0.13 %** |

Effectively zero waste across the entire 1k–11k range.

---

## (5) Constraints recap

1. **Greek total ≤ English-unique (~13k)** → added budget ≤ ~11,521.
2. **Total vocab divisible by 128 (preferably 256)** → 1024-multiples
   (256 × 4) all qualify. Apertus base 131,072 = 1024 × 128.
3. **Decide by fertility + language %**.

---

## (6) Recommendation

### C3 baseline pick: 11,264 added units

| check | value | ✓ |
|---|---|---|
| (1) Greek total ≤ English unique (~13k) | 1,479 base + 11,167 Greek-payload added = **12,646 Greek tokens** ≤ 13,000 | ✓ |
| (2) Total vocab divisible by 256 | 131,072 + 11,264 = **142,336** = 256 × 556 | ✓ |
| (3) Best fertility achievable under cap | **1.47** (clean held-out) | ✓ |
| Greek-script share of added budget | 11,206 / 11,264 = **99.5 %** | ✓ |
| True noise share | 15 / 11,264 = **0.13 %** | ✓ |

Cumulative gain over Apertus base:
- fertility: 2.41 → 1.47 (−39 %)
- chars/token: 2.59 → 3.93 (+52 %)
- single-token Greek-word share: 0.44 → 0.92 (+109 %)

### Why not larger

12,288 (next 1024-step) → Greek total 13,767 = exceeds the 13k English-
unique cap. 15,360+ adds little fertility per step and grows the
unused-token cost faster than the fertility gain shrinks (see §2 full-
sweep table).

### Why not smaller

8,192 (~French) → fertility 1.55 vs 1.47 at 11k; that's a 5 % fertility
cost on every Greek inference call, applied forever, to save 3,072
embedding rows (~50 MB BF16). Fertility curve hasn't bent inside the
cap, so taking the cap is justified.

### Optional tight-to-cap variant

11,520 (= 256 × 45) → Greek total 12,999, touches the 13k cap exactly.
Fertility interpolates to ~1.464. Build + measure is one variant build +
~5 min sweep. Gain over 11,264: ~0.007 fertility. Worth doing only if
exact-cap alignment is desired as a defensible round number.

### Caveat — next decision not fixed here

C3 is the **combined-Greek baseline**. The next decision is whether the
polytonic/ancient Greek material should be added as an extension to an
existing tokenizer, or whether modern Greek should be freshly tokenized
first and then extended with a separate polytonic lane.

The C3 11,264 recommendation here is for **the C3 arm specifically**.
It informs the next design (the marginal-fertility shape, the glossary
trajectory, the 99 % Greek-purity finding) but it does not commit to a
shipping cutoff for a future arm. That decision should be reanchored
against the chosen training data, dedup state, and polytonic held-out
slices before freeze.

---

## Reproduction

All three scripts now default to this sub-subproject's local
`artifacts/` for both outputs and (when present) snapshotted inputs.
Each can be redirected with an env var (see script headers).

```bash
# (3) glossary categorisation per cutoff
python3 scripts/apply_cutoff_grid.py
#   reads:  ~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl
#   writes: artifacts/cutoff_grid/distribution_at_<N>.json
#           artifacts/cutoff_grid/cutoff_grid_summary.{md,json}
#   override: CUTOFF_GRID_OUT_DIR=<path>

# (4) combined char-mask + glossary classification
/home/foivos/.venvs/pq-probe/bin/python3 scripts/classify_added_tokens.py
#   reads:  same glossary as (3)
#           subprojects/02_2_tokenizer_implementation/02_2_1_char_language_membership/artifacts/char_language_bitmask.parquet
#           ...                                                                              /artifacts/manifest.json
#               (used to derive live language- and script-layer bit widths;
#                falls back to 88/29 if missing)
#   writes: artifacts/classified_added_tokens.jsonl
#           artifacts/per_cutoff_report.json
#   override: CLASSIFY_OUT_DIR=<path>

# (2) per-cutoff fertility report w/ plots
python3 scripts/build_cutoff_report.py
#   reads:  artifacts/c3_cutoff_metrics.json
#               (snapshot from the gcloud fertility run at
#                ~/runs/c3_cutoff_eval_20260511/fertility_c3_full_25_clean_20260511/metrics_by_slice.json;
#                if missing, the script falls back to /tmp/c3_cutoff_metrics.json,
#                so a fresh pull from the instance works without copying)
#           artifacts/cutoff_grid/distribution_at_<N>.json  (from step 3)
#   writes: docs/figures/c3_cutoff_*.png
#           docs/C3_CUTOFF_REPORT.md
#   overrides: CUTOFF_METRICS_JSON=<path>, CUTOFF_GRID_DIR=<path>
```

Inputs the scripts need available somewhere (defaults shown):
- corrected glossary at `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl` — currently local; not snapshotted into `artifacts/` because it's ~25 MB and gitignored. Regenerated by the upstream Gemini pass.
- char-language bitmask + manifest at `subprojects/02_2_tokenizer_implementation/02_2_1_char_language_membership/artifacts/` — already in repo.
- fertility metrics at `artifacts/c3_cutoff_metrics.json` — **not present in repo today**; pull from the gcloud instance (path above) and `cp` into `artifacts/` before running step (2). Or set `CUTOFF_METRICS_JSON` to the instance-side path if you've mounted it.

## See also

- [`../README.md`](../README.md) — parent subproject overview
- [`../../../docs/C3_CUTOFF_REPORT.md`](../../../docs/C3_CUTOFF_REPORT.md) — the full 1k–25k fertility sweep report with plots
- [`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) — C3 baseline status
- [`../../02_2_tokenizer_implementation/02_2_1_char_language_membership/README.md`](../../02_2_tokenizer_implementation/02_2_1_char_language_membership/README.md) — rejection masks (§4 source)
- [`../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/) — PMI counts (§1 source)
- [`../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/english_review/membership_report.md`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/english_review/membership_report.md) — derivation of "English unique = ~13k"

# Stage 1 / Stage 2 review — what's there, what it does well, what breaks

**Scope.** Code + behaviour of the rule-discovery pipeline through the
aggregator (Stage 3 rule implementation is being built by a separate agent
and is out of scope for this pass).

**Evidence base.** Direct reads of the Rust matcher (`glossapi_rs_noise`) and
its PyO3 bindings, the four Python scripts under
`/home/foivos/glossAPI-development/src/glossapi/scripts/`, plus run outputs
from `wave10_v5` (40 Gemini reviews across 4 categories, 10 cases each).

---

## 1. Stage 1 — Rust matcher

### What it does

- Category specs live in JSON (`specs/first_pass_glossapi_review.json`): four
  categories today — `glyph_font_like`, `control_private_use_replacement`,
  `short_nonascii_latin_like` as `literal_set` (Aho-Corasick,
  LeftmostLongest), `dot_leader_like` as a regex `(?:[.…][ \t\u00A0]*){4,}`.
- The literal sets for the first three categories come from
  `tokenizer_analysis/inspection/non_greek/fresh/glossapi_only/categories/*.json`
  — i.e. bad-token inventories derived from a **fresh**, **GlossAPI-only**
  discovery tokenizer.
- Synthetic pages: `real_page` if `PAGE_SPLIT_MARKER` is found → else
  `synthetic_header` (markdown `#` block, ≥ `min_header_chars = 1200`) →
  `synthetic_paragraph` (paragraph-split inside a block that exceeds
  `hard_max_chars = 6000`) → `synthetic_fallback` (char-budget chop).
  Default page target = 4000 chars.
- Raw spans from every spec are collected per page, then sorted and
  **merged**: overlapping or touching spans become one merged span that
  carries the *union* of `categories`, `pattern_families`, and `raw_texts`.
- Output per page: annotated `.md`, with per-page rows in `manifest.jsonl`,
  `page_metrics.jsonl` (match density, per-category/per-family counts), and
  per-match rows in `match_index.jsonl` (match_id, start/end char+byte+line,
  categories, pattern_families, matched_text, raw_texts, and a fixed
  **240-char** context_before / context_after / context_excerpt).

### What's solid

- Stateless, Rayon-parallel via the Python driver (`batch_size=256`
  parquet batches). Clean re-runs: stale `*.md` and the four JSONL files are
  deleted before every run.
- Two matcher kinds (regex + literal set) with HTML-entity-decoded matches
  caught by the cleaner downstream.
- Page-density signals (`match_count`, `match_density_per_1k_chars`) let
  downstream reviewers judge whether a match is isolated or part of a
  broadly-noisy page — important for the short-fragment case.

### What's concerning

1. **Sources are tokenizer-derived, not corpus-derived.** The literal files
   live under `…/fresh/glossapi_only/…`. That means:
   - artifacts visible only to *other* tokenizers (continuous BPE from
     Apertus, or GlossAPI + HPLT mixed) are structurally invisible to the
     matcher;
   - anything that exists in the corpus but didn't become a bad-token in
     that one tokenizer's vocabulary is missed by construction.
2. **Literal entries are multi-char token surfaces, not bare codepoints.**
   `control_private_use_replacement_decoded.json` has 391 literals; per-char
   breakdown: 350 × `U+FFFD`, 107 ASCII chars, 50 PUA, 5 Latin-1 Supplement,
   4 Greek, and assorted controls. The ASCII/Greek chars appear because
   literals carry the surrounding characters that the BPE tokenizer merged
   alongside the bad one. Consequence: an Aho-Corasick hit drags legitimate
   context into the matched span. This is the root of the misfires we see
   in wave10 (below).
3. **Context window is hardcoded to 240 chars** in
   `token_category_debug_common.py:233`. Not CLI-configurable. For
   "is the page broadly corrupted?" questions the model effectively has
   only the density numbers, not the evidence.
4. **Merged spans carry a union of categories.** When a span is flagged by
   both `glyph_font_like` and `short_nonascii_latin_like`, the bundler
   samples it *once per category*; the model sees the same physical span
   twice under two different prompts. This is a design feature (each category
   reviewed independently) but creates review duplication that isn't
   accounted for in sample budgets.
5. **The `dot_leader_like` regex `(?:[.…][ \t\u00A0]*){4,}`** matches runs
   of four-or-more *dots-with-optional-spaces-between*. That is broad
   enough to match an ellipsis `…` followed by dots, or even a dot-space-
   dot-space pattern. In TOC leaders this is what you want; in prose it can
   catch genuine multi-sentence ellipses with intervening spaces. No
   negatives were audited.
6. **Corpus-source coverage gap** — see
   `PIPELINE_REVIEW_PENDING.md` M7a: runs only touch `openarchives.gr` and
   `hplt` out of 17 GlossAPI sources.

---

## 2. Stage 2a — Review bundler

### What it does

`build_token_category_review_bundle.py`: reads the matcher's
`match_index.jsonl` + `page_metrics.jsonl`, joins page metrics, samples up
to `sample_size_per_category` matches per category with a
SHA1-deterministic shuffle, and writes a per-case `.txt` under
`cases/<category_slug>/<NNNN>__<cat>__<family>__<source>__pXXXXX.txt`.

Each case file contains metadata header (REVIEW_MODE, CATEGORY, PATTERN_FAMILY,
MATCH_ID, SOURCE_*, PAGE_*, MERGED_MATCHED_TEXT, RAW_TEXTS, CONTEXT_BEFORE,
CONTEXT_AFTER, PAGE_MATCH_COUNT, PAGE_MATCH_DENSITY_PER_1K,
PAGE_CATEGORY_MATCH_COUNTS, PAGE_PATTERN_FAMILY_MATCH_COUNTS) +
`REVIEW_GOAL` + `FIELD_NOTES` + `CATEGORY_QUESTIONS` +
`MATCH_CONTEXT_EXCERPT`.

`DEFAULT_REVIEW_SPECS` hardcodes four categories with explicit goal + 4–5
questions each.

### What's solid

- Deterministic, reproducible sampling.
- Metadata headers are rich and clearly labelled.
- Separation of goal / notes / questions / excerpt sections is tidy; the
  model can cite the excerpt.
- `FIELD_NOTES` explicitly explains the MERGED vs RAW_TEXTS distinction.
- `_slugify_label` for safe filesystem paths.

### What's concerning

1. **Only four category specs are hardcoded.** The plan names separator
   families, `table_border_ascii_art`, markdown table separator cells,
   `latin_word_like`, `latin_mixed`, etc. None are in DEFAULT_REVIEW_SPECS;
   they'd fall back to `review_mode=unknown`, which the driver treats
   passively.
2. **`--include-full-debug-page` defaults OFF.** In wave10 bundles
   (`…_bundle_20260416_v3`) the debug page with match tags is not included.
   Reviewers see only the 240-char excerpt plus the page-density numbers.
   The "is this whole page noise?" question is harder to answer with
   confidence from numbers alone — especially on `short_nonascii_latin_like`
   where the plan explicitly anticipates page-level judgments.
3. **Sampling is per-category, not stratified by `pattern_family`.** When a
   category has multiple families the sort-by-hash order may starve the
   smaller ones. The present spec files each declare one `pattern_family`
   per category, so this hasn't mattered yet; it will once the spec grows.
4. **Aspirational "existing cleaner context" is absent.** MODEL_REVIEW_SPEC
   §4.2 says the model should see "what current cleaner rule/pattern already
   exists for this family" — e.g. the existing `GLYPH_FONT_TAG_REGEX` in
   `glossapi_rs_cleaner/src/cleaning_module.rs:27-28`. The bundler does
   not inject this. Consequence: when the schema asks the model to choose
   between `existing_regex_extension` and `new_regex`, the model is
   guessing what "existing" means.

---

## 3. Stage 2b — Gemini Batch driver

### What it does

`review_token_category_with_gemini.py`: builds a prompt per case as

```
REVIEW_MODE: <mode>\n
<mode-specific one-line instruction>\n
Use the following case file exactly as provided.\n
<case_text>
```

sends it to `gemini-2.5-flash` (per the wave10 requests.jsonl) with
`response_mime_type: application/json` + `response_json_schema` (strict per
mode). Has REST + SDK paths, inline/batch modes, retry on 408/429/5xx,
thinking-budget/level support.

`COMMON_SYSTEM_INSTRUCTION` tells the model to be conservative and to return
`uncertain` when unsure.

### What's solid

- Strict schemas force typed JSON (enum values for all decisions).
- Both REST and SDK implementations, dry-run support, batch inline/file mode.
- Model ID is carried on every request (partial fix for the version-pinning
  concern).
- Conservative system instruction is the right framing.

### What's concerning

1. **The schema conflates two questions per cleaning case** —
   `match_update_type` asks the model to pick among
   `{existing_regex_extension, new_regex, no_regex_needed, keep, uncertain}`
   *without telling the model what the existing regex is*. So
   `existing_regex_extension` vs `new_regex` is asked blind. The
   `regex_scope_note` free-text field gets the model to hand-wave about
   broadness, but the decision lacks grounding.
2. **Normalization mode doesn't show before/after.** The prompt
   includes the canonical target in the CATEGORY_QUESTIONS text ("Is it
   safely interchangeable with the canonical leader form `.....` in this
   context?") but no materialised transformed version. The model's
   "interchangeable" judgment is an imagination, not a comparison.
3. **System instruction says "Be conservative. Return uncertain when not
   confident."** In wave10 the `uncertain` rate is ~5% across all cases.
   Either the cases are clear-cut (unlikely — see §4 below on
   `control_private_use_replacement`), or the model tends not to escalate
   to `uncertain` as often as the instruction requests. This is a
   calibration signal worth tracking.
4. **Temperature / thinking_budget defaults aren't in the excerpt I read**
   — they're in CLI args. Worth verifying they were set consistently across
   waves; drifting thermal settings across waves would make wave-to-wave
   comparisons unfair.

---

## 4. Empirical Gemini-output quality on wave10

### dot_leader_like (normalization, n=10) — **HIGH quality**

10/10 agreement on every field. Every case judged `layout_leader`,
`interchangeable_with_target=yes`, `preserves_semantics=yes`,
`canonical_target=".....`. Reasons are consistent and grounded ("layout
leader connecting title to page number", "typical of a Table of Contents").

**Caveat**: this is a trivially easy category. Long dot runs in TOC context
are unambiguous. This result validates that the pipeline *can* produce a
clean promotion — but it doesn't stress-test the review machinery.

### glyph_font_like (cleaning, n=10) — **MIXED quality**

- 10/10 agree on `is_noise=yes`. Good.
- 9/10 say `match_update_type=existing_regex_extension`; 1 says `new_regex`.
  But no evidence of what the existing regex IS was in the prompt — see §3.
- **Ten distinct regex proposals.** After `_normalize_candidate_regex`
  (HTML-unescape + strip) collapses `GLYPH<c=\d+,font=/[^>]+&gt;`,
  `GLYPH<c=\d+,font=/[^>]+>`, and ` GLYPH<c=\d+,font=/[^>]+>` into one,
  the dominant pattern gets 3 votes. The other seven proposals range from
  sensible (`font=/[A-Za-z0-9+]+`) to trivial (`;GLYPH`) to nonsensical
  (`[t;]?GLYP[H]?`).
- The existing library regex
  `cleaning_module.rs:27-28` is
  `glyph<c=\d+,font=/[^>]+>` (case-insensitive via `(?i)`). Gemini's
  "winning" proposal `GLYPH<c=\d+,font=/[^>]+>` is **case-variant of the
  existing rule**. The aggregator flagged this as an "existing_regex_extension"
  and that is semantically correct — but the rule change is cosmetic, not
  substantive.

**Reading this result honestly**: wave10 surfaced a case mismatch that's
plausibly already handled by the library's `(?i)` flag. A careful reviewer
would check before landing this as a "promoted rule."

### control_private_use_replacement (cleaning, n=10) — **LOW quality and exposes a real bug**

- 5 `no`, 4 `yes`, 1 `uncertain`. Does not promote.
- The disagreement reveals a deeper problem: the matches center on
  `U+03A2` (the **unassigned** codepoint in the Greek block — no
  character is encoded there). The model's explanations show hallucinated
  identity:
  - 3 cases: "`Σ` is a valid Greek letter" — wrong; `Σ` is U+03A3, not
    U+03A2.
  - 2 cases: "OCR error for Σ" — correct intuition about the source, but
    the character is still genuine noise.
  - 1 case: "Greek Capital Letter San" — san is U+03FA, not U+03A2.
  - 1 case: `U+2022` BULLET called "legitimate list marker", another case
    of the same bullet called "structural noise".

- **The matcher is correct** to flag `U+03A2` as reserved/noise, but
  because the literal set carries surrounding Greek characters (see Stage
  1 concern #2), the model sees the whole multi-char span and reasons
  about the *neighbouring* characters rather than the one flagged codepoint.
- Net: the aggregator's non-promotion decision is correct, but for the
  wrong reason — it's catching model disagreement, not model-identified
  errors. A human reviewer would see there's a real signal here that the
  model is missing.

### short_nonascii_latin_like (cleaning, n=10) — **MIXED quality, regex proposals unusable**

- 10/10 agree on `is_noise=yes` ("mojibake", "broken glyph decoding",
  "widespread page corruption").
- 4/10 `no_regex_needed` (reasoning: glyph_font_like regex already
  catches the page).
- 6/10 propose regexes, *none the same*:
  - `ǂ` (single literal),
  - `[ǎǐ]` (two-char class),
  - `[ǂǎǕŽǒǗƿǚǘžǐƽǖǓƮǝǑǍ]+` (all characters literally in the example),
  - `[\p{L}&&[^\p{ASCII}]]+` — **Java regex syntax; won't compile in
    Python `re`**,
  - `[\u0100-\u024F]+` (Latin Extended-A range),
  - `[\p{InLatin_Extended_A}\p{InLatin_Extended_B}]+` — **Java syntax
    again**.
- The aggregator's `_compile_regex` will reject the Java-syntax ones
  (they return `None` from `re.compile`, so `validated_hit_count=0` and they
  don't promote). The aggregator does not report to the user that the
  model produced uncompilable regexes — they're silently filtered.

### Summary of what the model did and didn't do well

| Category | Noise agreement | Regex discipline | Risk of a bad promotion |
|---|---|---|---|
| `dot_leader_like` | 10/10 | n/a (normalization) | low |
| `glyph_font_like` | 10/10 | weak — 10 proposals, one ties by HTML-unescape dedup | moderate — the promoted regex is a case-variant of an existing `(?i)` rule |
| `control_private_use_replacement` | 5/4/1 split | n/a (did not promote) | low for promotion, but the model's hallucinated codepoint identity is a general concern |
| `short_nonascii_latin_like` | 10/10 | weak — 6 distinct proposals, 2 in Java regex syntax | moderate — needs manual regex selection, not auto-union |

---

## 5. Aggregator — concrete issues

### What's solid

- Joins bundle + review cleanly on `review_case_id`.
- Computes rates per decision field.
- Promotion gates are explicit and conservative (cleaning: noise_yes ≥
  0.85 AND uncertain ≤ 0.15 AND top_update_type ∈ {extension, new} AND
  top_update_rate ≥ 0.5 AND supported_regex exists; normalization:
  normalizable, interchangeable, semantics, markdown all pass their
  thresholds).
- Manual review queue captures uncertain cases.
- HTML-unescape in `_normalize_candidate_regex` dedupes some
  near-duplicates.

### What's broken or misleading

1. **`_regex_hits_case` is not a validation.** It tests whether the regex
   hits *the case it came from* — which, by construction, it should. Every
   `validated_hit_rate_over_votes` is an upper bound at best. The spec's
   requirement "deterministic validation must not hit reviewed negatives"
   is not implemented. Calling this field "validated" is overstated.
2. **No negatives are tested anywhere.** There is no held-out set of
   "clean paragraphs that should not match." Without this the aggregator
   cannot detect over-reach.
3. **Regex-union is plain alternation** (`(?:p1)|(?:p2)|…`). The spec
   (§10.2) calls for "compile into an automaton, minimise/simplify". Not
   implemented. The `union_regex` value in a promotion is the alternation
   string as-is.
4. **`min_reviewed_per_family` default is 10, not 12** as
   MODEL_REVIEW_SPEC §10.4 says. Wave10_v5 ran at 10. Either the spec or
   the CLI default should move.
5. **Promotion ignores statistical weight.** A family with 10/10 agreement
   is promoted identically to 255/300 agreement; Wilson 95% lower bounds
   differ (≈72% vs ≈80% at 85% point). See `PIPELINE_REVIEW_PENDING.md` W3.
6. **`canonical_target_counts` uses raw strings** — trailing whitespace,
   Unicode variants, escaping differences split buckets. `".....` and
   `"....."` (with trailing space) would be separate targets. Should
   normalise before counting.
7. **Bullet (`U+2022`) collision.** In wave10 the literal
   `control_private_use_replacement_decoded.json` contains many bullets.
   The model's bullet judgments split (noise vs legitimate). The aggregator
   cannot currently distinguish "two model runs gave different answers"
   from "the same model saw two different contexts and correctly gave
   different answers" — both show up as split family rates.

---

## 6. What Stage 3 will inherit from this state

The separate agent building Stage 3 (rule implementation into `Corpus.clean`)
will be handed the following artefacts per promoted family:

- `candidate_{cleaning,normalization}_rules.jsonl` rows with
  `top_candidate_regex`, `candidate_regex_union`, `regex_inventory`,
  `canonical_target`, and the rates.
- No negatives.
- No regression examples.
- No before/after byte/token impact measurement.
- Rates that conflate agreement with statistical confidence.

Practical consequence: the Stage 3 agent must treat every promoted rule as
a **candidate to validate on independent data** before the rule lands. The
current promotion label is best read as "this family is probably worth
spending an hour of cleaner-engineer time on", not "this rule is ready to
merge."

---

## 7. Top recommendations for Stage 1/2 (ordered by impact)

These are focused on what we can do *without* waiting for Stage 3. Each is
independently actionable.

### P1 — Inject the existing cleaner regex inventory into cleaning-mode prompts

The largest measurable improvement: make the case file include
`EXISTING_CLEANER_PATTERNS` (regex list for the current category). Model can
then answer `existing_regex_extension` vs `new_regex` with grounding.
Implementation: extend the bundler's case-file template and the review
driver's prompt; the library's regex inventory is small enough to list
inline.

### P2 — Materialise before/after for normalization-mode cases

Apply the chosen canonical target to the matched span, emit
`CONTEXT_AFTER_NORMALIZATION` alongside `CONTEXT_AFTER`. Same prompt, new
field. This turns "is it interchangeable?" from an imagined question into a
visible comparison.

### P3 — Make `_regex_hits_case` honest, and add negatives

Split into `_regex_hits_on_source_case` (current behaviour, rename to
reflect what it actually measures) and `_regex_hits_over_reviewed_negatives`
(test against reviewed cases that are NOT noise). Without the second, we
cannot claim "validated". Requires a negative set, which can bootstrap from
`control_private_use_replacement` "is_noise=no" cases from wave10.

### P4 — Fix or filter non-Python regex proposals loudly

When `_compile_regex` returns None, log it as a data-quality flag in
`family_summary.jsonl` (`uncompilable_regex_count`). Gemini produces
Java-syntax regexes often enough to matter.

### P5 — Promote on Wilson lower bound, not point rate

Replace flat ≥ 0.85 / 0.90 with lower-bound thresholds (e.g. Wilson 95%
lower bound ≥ 0.75 / 0.85). Small samples need stronger point agreement;
large samples can pass with current rates. Statistically honest.

### P6 — Stratify sampling by pattern_family

Trivial code change in the bundler. Important once specs grow beyond the
current 1-family-per-category shape.

### P7 — Expand the spec beyond four categories and beyond fresh-GlossAPI-only

Add the remaining plan categories (separator families, markdown table
separator cells, table_border_ascii_art, latin_* families). Pull literal
inventories from the continuous-BPE and GlossAPI+HPLT tokenizers as well,
so Stage 1 stops being biased by a single tokenizer's vocabulary.

### P8 — Pin thinking-level and temperature in wave manifests

Add them to `requests.jsonl` alongside `model`. Drift between waves is
otherwise invisible.

---

## 8. Status vs PIPELINE_REVIEW_PENDING.md

This document complements `PIPELINE_REVIEW_PENDING.md` — where that file
lists *design-level* findings on the plan, this one lists *implementation-
level* findings from reading the code and running outputs. Items here that
are also in that file (and where to find them):

- P3 (validation on negatives) → `PIPELINE_REVIEW_PENDING.md` §W4
  (re-audit over-cleaning), §M1 (regression-test set).
- P5 (Wilson bound) → §W3 (statistical weight).
- P6 (stratified sampling) → §W5.
- P7 (corpus-source coverage) → §M7a.
- The **three new findings** that were not in the earlier file:
  **P1, P2, P4** — existing-rule context injection, shadow-applied
  before/after, uncompilable-regex reporting — all surfaced by reading
  actual wave10 output.

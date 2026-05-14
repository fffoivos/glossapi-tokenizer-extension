# DRAFT — noise-page detection prompt

**Status**: proposal for review. Not wired into
`build_token_category_review_bundle.py` / `_build_prompt` yet.

**Task class**: per-page threshold-validation review.
**Review question**: are pages above the deterministic bigram-density
threshold genuinely unsalvageable? Are they cleanable, or should they go?

---

## Deterministic step first (no model)

Before any page reaches Gemini, compute a page-level noise score
deterministically in the Rust matcher:

```
suspect_bigram_char_ratio(page) =
    sum(len(b) for b in page_bigrams if b in SUSPECT_BIGRAMS) /
    total_chars_on_page
```

Where:
- **Page** = synthetic page (already built: header → paragraph →
  400-line fallback).
- **SUSPECT_BIGRAMS** = bigrams inventoried from the fresh-GlossAPI-only
  discovery tokenizer's non-Greek / non-ASCII vocab, minus a whitelist
  of common Greek digraphs (αι, ει, οι, ου, αυ, ευ, ηυ, γγ, γκ, γχ,
  μπ, ντ, τσ, τζ, σσ, ττ, λλ, ρρ, κκ, νν).
- Starting threshold: flag the page as suspicious when
  `suspect_bigram_char_ratio > 0.05` AND
  `suspect_bigram_match_count > 40`.

These numbers are placeholders. Gemini's job is to help set them, not
to compute them.

---

## Layer 1 — Wave preamble (cached)

```
[PROJECT_CONTEXT]
You are helping identify broadly-noisy pages in a Greek-language corpus
used to train a subword tokenizer extension for Apertus-8B-2509. A
noisy page inflates the tokenizer's vocabulary with one-off garbage
sequences and degrades BPE compression on legitimate Greek text.

[TASK_CONTEXT: page-level noise detection]
We flag a synthetic page as suspicious when the ratio of "suspect
bigrams" (rare non-Greek bigrams that are not common Greek digraphs)
exceeds a threshold. Above that threshold, we ask you three questions:

1. Is this page genuinely noisy (most of its content serves no
   semantic purpose)?
2. If noisy, is any meaningful Greek text salvageable by removing the
   noise in place, or should the whole page be discarded?
3. If it's noisy, what kind(s) of noise dominate?

Noise means spans that serve no semantic or structural purpose — not
meaningful language, not meaningful mathematics, not valid table or
formatting structure. Legitimate foreign-language content, legitimate
technical content, or legitimate mathematical notation is NOT noise
for this task.

[PROPOSED_USE]
We will use your judgments to tune the suspect-bigram threshold and
decide whether a flagged page should be (a) kept as-is, (b) cleaned by
a character-class or regex rule, or (c) dropped entirely. You do NOT
need to write the cleaning rule yourself — just classify the dominant
noise kind and we will map it to a rule from a fixed catalog.
```

## Layer 2 — Per-case case file

```
[PAGE_META]
review_case_id:                  noise_page::<match_id>
source:                          <source_corpus>/<source_path>
synthetic_page_index:            <n>
page_char_count:                 <int>
suspect_bigram_match_count:      <int>
suspect_bigram_char_ratio:       <float>
suspect_bigram_top_5:            [(bigram, count), ...]
threshold_bucket:                "below" | "just_above" | "far_above"

[PAGE_CONTENT]
(Full synthetic page, verbatim — typically 400 lines or less. No
truncation; if the page exceeds 6000 chars we send only the first 6000
plus a clear `[TRUNCATED: N chars omitted]` marker.)
...
...

[REVIEW_QUESTIONS]
Answer ONLY from the evidence in this case file. Be conservative. If
unsure, answer `uncertain`.

1. Is this page genuinely noisy — i.e., does most of its character
   budget serve no semantic or structural purpose?
2. If noisy, is any meaningful Greek (or legitimately foreign-language)
   content salvageable by removing noise in place, or should the whole
   page be discarded?
3. If noisy, which noise kinds dominate? Pick one or more from the
   closed set below.
4. If you answer `uncertain` to question 1 or 2, give one short
   sentence explaining the blocker.
```

## Layer 3 — Output schema (enforced via `responseJsonSchema`)

```json
{
  "is_noisy_page": "yes" | "no" | "uncertain",
  "page_disposition": "salvageable_clean" | "discard" | "flag_for_ocr" | "keep_as_is" | "uncertain",
  "dominant_noise_kinds": [
    "pua_replacement_chars"       // U+FFFD, U+E000-F8FF
    | "glyph_font_tags"           // GLYPH<c=...,font=...> extractor residue
    | "mojibake_latin_extended"   // broken glyph decoding → Latin Extended-A/B
    | "ocr_char_salad"            // dense unparseable garbage
    | "broken_layout"             // table/column residue rather than noise
    | "foreign_script_dominant"   // page is another language, not noise
    | "mathematical_symbols"      // legitimate math, not noise
    | "other"
  ],
  "blocker": "<short sentence or empty string>"
}
```

All fields required. `dominant_noise_kinds` must be empty list if
`is_noisy_page == no`.

---

## Mapping: Gemini output → cleaning rule (done deterministically, not by Gemini)

| `dominant_noise_kinds` value  | Rule we apply                                 |
|------------------------------|-----------------------------------------------|
| `pua_replacement_chars`      | strip Unicode class U+FFFD + U+E000–F8FF      |
| `glyph_font_tags`            | apply existing `GLYPH_FONT_TAG_REGEX` (`cleaning_module.rs:27-28`) |
| `mojibake_latin_extended`    | line-level density filter + strip U+0100–U+024F |
| `ocr_char_salad`             | drop page (`page_disposition = discard`)      |
| `broken_layout`              | `page_disposition = flag_for_ocr`             |
| `foreign_script_dominant`    | keep; out of scope for this cleaner           |
| `mathematical_symbols`       | keep; out of scope for this cleaner           |
| `other`                      | manual review queue                           |

The model never sees this mapping. It only classifies.

---

## Sampling policy for threshold validation

First wave — calibration:
- 100 pages `just_above` the starting threshold
  (0.05 < ratio ≤ 0.10 OR 40 < matches ≤ 80)
- 50 pages `far_above` (ratio > 0.10 OR matches > 80)
- 50 pages `below` (ratio ≤ 0.05 AND matches ≤ 40) — sanity check the
  negative side

What the aggregator does with the result:
- If `is_noisy_page == yes` rate on `just_above` is ≥ 85%, the current
  threshold is safe; consider lowering it for broader coverage next
  wave.
- If `is_noisy_page == yes` rate on `just_above` is < 50%, raise the
  threshold.
- `is_noisy_page == yes` rate on `below` should be ≤ 5% — if higher,
  the bigram inventory is missing something.

## Design notes

- **Gemini never does density math.** The deterministic stage produces
  the metric and the threshold bucket. Gemini only validates whether
  the bucket label matches what a reader would say.
- **Rule synthesis is ours, not the model's.** The model classifies
  noise kind; we map kind → rule. This keeps us out of the
  executing-model-written-code trap and keeps the cleaner composed of
  rules we wrote and tested.
- **Closed-set `dominant_noise_kinds`** — seven enumerated labels,
  `other` as escape valve. The labels correspond to real extractor /
  OCR failure modes we already see in wave10 output.
- **`page_disposition` separates `discard` from `flag_for_ocr`.** Some
  pages are noise because extraction failed (re-run OCR might fix
  them); others are irrecoverable. The disposition field captures this.
- **Page-content truncation is explicit** rather than silent; the model
  should know when it's seeing a partial page.

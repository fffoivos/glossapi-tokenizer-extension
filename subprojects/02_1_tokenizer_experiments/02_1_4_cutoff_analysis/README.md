# 02_1_4 Cutoff analysis

Sub-subproject of `02_1_tokenizer_experiments`. **Stage 4 (terminal):**

```
[02_1_1 tokenizer training]
[02_1_2 cutoff variant builder]
[02_1_3 fertility evaluation]
       │
       ▼
[02_1_4 cutoff analysis] → recommends a cutoff
```

## Goal

Combine three independent evidence streams to recommend a cutoff for
the trained tokenizer arm:

1. **Comparable-language vocab sizes** — empirical per-language token
   footprints in Apertus base (from `02_2_2_vocab_lang_attribution`).
   Anchors "what size of Greek vocab is roughly equivalent to X for
   another major language".
2. **Fertility on held-out** — from `02_1_3`. Per-(cutoff, slice) the
   primary Greek-quality metric is `greek_word_space_fertility`. The
   marginal gain per +1024 step is the elbow signal.
3. **Token-content composition** — combines:
   - **glossary categorization** (from
     `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl`
     — Gemini-pass per-token category + morphological structure +
     lexical role)
   - **char-language membership** (from
     `subprojects/02_2_tokenizer_implementation/02_2_1_char_language_membership/`
     — strict rejection-mask bits at script / family / language layers)

   Cross-product gives a "function" label per token:
     - GREEK (real Greek payload)
     - USEFUL_STRUCTURAL (MD tables / dot leaders / escape runs / math)
     - NOISE (mojibake / encoding artifacts / orphan diacritics)
     - AMBIGUOUS (mixed-script punct+Greek patterns; usually useful in
       practice but conservative tag)

## Outputs

- `REPORT.md` — first-draft cutoff-decision report (this directory's
  primary deliverable)
- `artifacts/` — local CSVs / JSON of the per-cutoff breakdowns the
  REPORT.md cites

## Scripts

- `scripts/apply_cutoff_grid.py` — slices the corrected C3 glossary
  into per-cutoff distributions. For each cutoff `N`, takes the first
  `N` rows of the glossary (sorted by id) and aggregates by category /
  greek_structure / greek_lexical / confidence bucket. Output:
  `distribution_at_<N>.json` + `cutoff_grid_summary.{md,json}`.
- `scripts/classify_added_tokens.py` — applies the
  `02_2_1_char_language_membership` char-mask table to each added
  token, classifies into the four-bucket function label, and tabulates
  per-cutoff progression. Output:
  `classified_added_tokens.jsonl` + `per_cutoff_report.json`.
- `scripts/build_cutoff_report.py` — generates the matplotlib PNG plots
  + the prose table cells used in `docs/C3_CUTOFF_REPORT.md`.

## Required inputs (provenance)

- **Fertility metrics**: `02_1_3` output
  (`fertility_<arm>_<scope>/metrics_by_slice.json`).
- **Glossary**: `~/runs/c2_c3_analysis_20260506/c3_added_tokens_20260507/data/glossary/tokens_glossary.jsonl`
  (corrected post-Gemini glossary). For future arms, the equivalent
  glossary should be (re-)produced before this stage runs.
- **Per-language attribution counts**: PMI promotion outputs from
  `subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv`.
- **Char-language masks**: `subprojects/02_2_tokenizer_implementation/02_2_1_char_language_membership/artifacts/char_language_bitmask.parquet`.

## Current C3 recommendation

See [`REPORT.md`](REPORT.md). Headline: **11,264 added units** (total
vocab 142,336) as the C3 baseline pick, subject to constraints
(Greek total ≤ English-unique ~13k; total vocab divisible by 128/256;
fertility + language-% optimized within the cap).

## Future Decision Boundary

C3 is the current combined-Greek baseline. A future modern/polytonic
split may be evaluated after source selection and dedup planning, but
that arm, its training recipe, and its cutoff grid are not fixed here.
The C3 analysis remains baseline evidence for whatever tokenizer path is
chosen next.

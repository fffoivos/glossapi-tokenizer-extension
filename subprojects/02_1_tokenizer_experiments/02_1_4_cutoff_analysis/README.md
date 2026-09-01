# 02.1.4 — Cutoff Analysis

> **In one line:** the first, analytic answer to "how many Greek units?" — **11,264**, derived from a self-imposed cap rather than from the data, and superseded a day later by measurement.
> **Period:** report dated 2026-05-17; committed 2026-05-18 (`7deea009`). **Status:** superseded by [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md), which chose 17,408. Its composition findings survive.
> **Came from / led to:** [`../02_1_3_fertility_evaluation/`](../02_1_3_fertility_evaluation/README.md) → this → [`../02_1_5_added_token_curation/`](../02_1_5_added_token_curation/README.md) and [`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/README.md)

## Why this existed

The cutoff had to be argued, not just measured. This stage joined three independent evidence streams — what vocab footprint comparable languages actually have inside Apertus, what fertility each cutoff buys on clean held-outs, and what the added tokens *are* — into one recommendation with a stated constraint set.

## History

**2026-05-17 — [`REPORT.md`](REPORT.md), marked "Draft".** Its three inputs:

1. **Comparable-language footprints** from `02_2_2_vocab_lang_attribution`'s PMI-promoted counts. English ~19,009 PMI tokens (~13k uniquely English); French 9,694; German 7,329; Korean 4,438 (largest script-isolated peer); **Greek 1,479** — 86.9 % of Greek mass captured, every token exclusively Greek. This retired the "Greek should match English's 88k" framing: 88k is the shared Latin-script *admissibility* ceiling, not anyone's allocation.
2. **Fertility** on the three verified-clean slices from `02_1_3`.
3. **Composition** — the Gemini-pass glossary crossed with `02_2_1`'s char-language masks, labelling each added token `GREEK` / `USEFUL_STRUCTURAL` / `NOISE` / `AMBIGUOUS`.

**The recommendation: 11,264 added units, vocab 142,336, fertility 1.47, 256-aligned.** The binding argument was constraint (1) — "Greek payload ≤ the ~13k uniquely-English anchor" — which caps the added budget at ~11,521. The report was candid that this was a *conservative design choice*, not a fact: the union of the two English slices gives ~16k, and "any cutoff up to ~17k can be defended". Its own anchor table already listed "match English total PMI (~19k) → **17,408**".

**The elbow was outside the cap.** Marginal fertility gain decayed smoothly from −0.155 (1k→2k) to −0.024 (10k→11k), and every step inside the cap still bought more than any step past 16k. The report's own conclusion: "under the cap, push as high as the constraint allows."

**2026-05-18 — superseded.** `02_1_7` dropped the cap, replaced it with a knee criterion measured on Apertus's own metric suite, and chose 17,408.

## Outcome

- **Recommendation (superseded)**: 11,264 / vocab 142,336. Cumulative gain over base at that point: fertility 2.41 → 1.47 (−39 %), chars/token 2.59 → 3.93 (+52 %), single-token Greek-word share 0.44 → 0.92.
- **Findings that survived and were reused**:
  - *Noise is flat.* True noise stays at **~0.13 %** of added units across the entire 1k–11k range; each +1,024 step adds ~1,011–1,020 real Greek tokens and at most ~13 non-Greek ones, and that cost does not rise with cutoff. Adding vocabulary does not dilute quality.
  - *Composition trajectory.* Early cutoffs are fragment-heavy; whole inflected `greek_word` forms dominate from ~5k (×20.5 growth 1k→11k); proper nouns and acronyms only accumulate past 11k. At 11,264: 4,182 whole-word Greek tokens, 2,479 morphemes, 637 function words.
  - *The exact non-Greek inventory* at 11,264 — 15 NOISE, 49 USEFUL_STRUCTURAL (TOC dot-leaders, MD table separators, math symbols, `.gr`), 33 AMBIGUOUS (mostly punctuation+Greek like `.Ε`, `,τι`, `/και`). This enumeration is what `02_1_5` turned into a removal policy.
- **Left open**: the report explicitly declined to decide whether a future arm should extend an existing tokenizer or freshly tokenize modern Greek and then bolt on a polytonic lane. The polytonic arm answered it by extending.

## Where things are

| What | Where |
|---|---|
| The report | [`REPORT.md`](REPORT.md) |
| Glossary slicing per cutoff | `scripts/apply_cutoff_grid.py` |
| Char-mask × glossary classification | `scripts/classify_added_tokens.py` |
| Plot + prose generator for the 1k–25k sweep report | `scripts/build_cutoff_report.py` → [`../../../docs/C3_CUTOFF_REPORT.md`](../../../docs/C3_CUTOFF_REPORT.md) |
| PMI footprint source | `../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/summary.tsv` |
| Char masks | `../../02_2_tokenizer_implementation/02_2_1_char_language_membership/artifacts/char_language_bitmask.parquet` |

`artifacts/` (per-cutoff distributions, `classified_added_tokens.jsonl`, the fertility metrics snapshot) is gitignored and regenerable; the glossary it depends on is ~25 MB and lives only on the worker.

## Working documents

- [`REPORT.md`](REPORT.md) is both the deliverable and the historical record. Read §2's full-sweep table and §4's noise-rate finding; treat §6's recommendation as superseded.

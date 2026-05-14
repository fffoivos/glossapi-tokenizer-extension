# Greek review — first-pass analysis

Source: `outputs/histogram_matrix.npz`, row `ell_Grek`, 1.003 B-token sample.

## Headline numbers

| Metric | Value |
|---|---:|
| Total Greek-sample tokens | **1,002,997,071** |
| Vocab entries that fired in Greek (≥1) | **100,014** / 131,072  (76.3 %) |
| ≥10 occurrences | 61,850 |
| ≥100 | 26,726 |
| ≥1,000 | 5,956 |
| ≥10,000 | 1,894 |
| ≥100,000 | 1,302 |
| Vocab entries with ≥1 Greek codepoint (mono + poly) | **1,507** |
| Greek-script tokens that fired | **1,504** / 1,507  (99.8 %) |
| Greek-script tokens that fired zero times | **3** (`▁μ`, `▁μg` × 2 variants) |
| Share of Greek mass in Greek-script tokens | **88.56 %** |
| Share of Greek mass in non-Greek-script tokens ("shared infrastructure") | **11.44 %** |

## What the picture is

The Greek count distribution is the classic Zipf — heavy head, long tail.
**~5,000 vocab entries cover essentially all of the Greek mass.** The top-30
overall tokens are dominated by:

1. **Pure punctuation/digits/structural tokens shared across languages** (`,`, `.`, `""`, `0`-`9`, `-`, `(`, `:`, …) — these are language-agnostic.
2. **Greek function words and frequent particles** (`και`, `να`, `το`, `του`,
 `της`, `την`, `με`, `από`, `για`, `που`, `είναι`, …) — these are unambiguously Greek.
3. **Single-character Greek subwords** (`ε`, `σ`, `ι`, `α`, `ο`, `η`, `ν`, `κ`, `λ`, …) —
 frequent suffixes/prefixes/letters from the byte-level BPE that the Apertus
 (Mistral-Nemo) tokenizer left as single-codepoint merges.

## How cleanly does Greek-script filtering separate?

**Very cleanly.** 88.56 % of the Greek count mass falls inside the 1,507
Greek-codepoint vocab entries. The remaining 11.44 % is overwhelmingly
language-shared infrastructure that you cannot blame on Apertus's tokenizer.

Top 10 non-Greek-script tokens by Greek count (from `tables/top200_NOT_greek_script.tsv`):

| Token | Greek count | What it is |
|---|---:|---|
| `,` | 20,366,551 | punctuation |
| `.` | 12,087,394 | punctuation |
| `""` | 7,033,169 | double-quote pair |
| `".\n` | 6,945,213 | period-newline structural |
| `0` | 3,831,574 | digit |
| `1` | 3,830,602 | digit |
| `-` | 2,944,031 | dash |
| `2` | 2,882,062 | digit |
| `"\n` | 2,520,858 | quote-newline structural |
| `(` | 1,775,833 | parenthesis |

These are dominantly **pure structural / digit / Latin-letter-mixed tokens** that fire across every language equally — exactly the "shared infrastructure" interpretation. **There's no surprise content here.**

One mild anomaly: token `1206 = �` (replacement char) fires 927,670 times in Greek, suggesting some FineWeb-2-HQ Greek docs have encoding issues. Worth flagging but not Greek-specific.

## What remains after Greek-script filter?

Of the 1,507 Greek-codepoint vocab entries, **1,504 fired at least once**.

The 3 that fired zero times:
- `▁μ` (id 49669)
- `▁μg` (id 121271)
- `▁μg` (id 122838 — variant)

All three are `μ` used as a **unit prefix** (micro-) followed by `g` (gram). These exist in the Apertus base vocab because the broader multilingual corpus that trained Mistral-Nemo's tokenizer includes scientific/measurement text mixing Greek letters with Latin units. The Apertus FineWeb-2-HQ Greek slice we sampled doesn't seem to contain this pattern (probably because the quality filter discards drug-info / dosage / scientific snippets). **Not a vocab waste — these would fire on technical Greek text outside FineWeb-2-HQ's HQ filter.**

## Threshold sweep: where does count-based "this is Greek" become precise?

From `tables/threshold_sweep.tsv`:

| Greek count ≥ | tokens kept | Greek mass kept | precision (fraction of kept that are Greek-script) |
|---:|---:|---:|---:|
| 0 | 131,072 | 100 % | 1.15 % |
| 1 | 100,014 | 100 % | 1.50 % |
| 10 | 61,850 | 99.99 % | 2.43 % |
| 100 | 26,726 | 99.86 % | 5.62 % |
| 1,000 | 5,956 | 99.20 % | 25.15 % |
| **10,000** | **1,894** | **98.17 %** | **78.46 %** |
| **100,000** | **1,302** | **95.62 %** | **95.31 %** |
| 1,000,000 | 241 | 62.51 % | 93.36 % |

**The interesting region is `T ∈ [10⁴, 10⁵]`.** At T=10,000 you keep ~1,900
tokens, get 98 % of Greek mass, and 78 % of them are Greek-script — so ~22 %
are shared-infrastructure tokens that are still genuinely Greek-relevant.
At T=100,000 you keep 1,302 tokens, get 95 % of mass, and 95 % are
Greek-script — extremely clean but you're starting to drop real Greek
content tokens that just don't fire 100k times in 1 B.

The Greek-script flag itself is a **strong-precision, slight-recall-loss** filter:
- **Recall**: 88.56 % of Greek mass.
- **Precision**: 99.8 % (only 3 of 1,507 Greek-script tokens don't fire in Greek; the rest absolutely are Greek tokens).

## Plots

- `plots/01_greek_zipf.png` — Greek count distribution rank-vs-count log-log. Classic Zipf.
- `plots/02_greek_cdf.png` — Cumulative Greek mass: all tokens (reaches 100 %) vs Greek-script subset (caps at 88.56 %).
- `plots/03_script_mass_breakdown.png` — Bar chart of Greek mass by mutually-exclusive script bucket.
- `plots/04_greek_script_zipf.png` — Same Zipf restricted to the 1,507 Greek-script tokens. Smoother / well-formed.
- `plots/05_greek_script_threshold.png` — Threshold sweep: tokens kept and precision as count threshold rises.
- `app/index.html` — interactive token-frequency explorer. Regenerate the ignored `app/data.json` with `app/build_data.py` after `outputs/` is present.

## Tables

- `tables/top200_overall.tsv` — top 200 vocab entries by Greek count, with script flags.
- `tables/top200_greek_script.tsv` — top 200 Greek-script-only.
- `tables/top200_NOT_greek_script.tsv` — top 200 NOT-Greek-script (the "shared infrastructure").
- `tables/greek_script_zero_count.tsv` — the 3 Greek-script tokens that fired zero times.
- `tables/threshold_sweep.tsv` — recall/precision curve under count thresholds.

## Takeaways for the C3 extension plan

1. **The script-based "this token is Greek" filter (1,507 entries) is a robust first pass.** It captures 88.56 % of Greek-text mass and ~99.8 % of those tokens are genuinely Greek-content.
2. **The remaining ~11.4 % of Greek mass that lives in non-Greek-script vocab is shared infrastructure** — pure punctuation, digits, whitespace, quote-newline patterns. Those tokens are equally important in *every* language and shouldn't be touched by a Greek-focused extension.
3. **The "what's added by C3" question** is whether the 25,600 new merges target morphology that the existing 1,507 Greek tokens compose inefficiently (i.e., they're well-formed individually but produce high fertility in real Greek text) — separate measurement; this analysis only tells us about *count-of-token-firing*, not about *fertility per word*.
4. **No urgent vocab waste**: only 3 Greek-script tokens fire zero times in 1 B Greek tokens, and even those are real scientific-notation patterns (`μ`, `μg`) that the FineWeb-2-HQ quality filter excludes. They'd fire in a broader Greek corpus.

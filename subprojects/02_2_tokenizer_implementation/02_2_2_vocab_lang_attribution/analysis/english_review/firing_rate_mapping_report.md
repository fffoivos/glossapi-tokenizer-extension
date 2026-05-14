# English: firing-rate mapping by script family

Reframed from the earlier `membership_report.md`. The
`02_2_1_char_language_membership` bitmask is used here only to **exclude** —
to partition fired tokens by script family. The **mapping** ("this
token is an English token") comes from the firing-rate distribution
within the Latin family of the English dataset.

## How fired tokens are partitioned

For each fired token in eng_Latn we look at `bitmask_and` and assign
a script family:

- `substrate` — `popcount == 55` (every locale admits — punctuation,
  digits, whitespace, common ASCII symbols).
- `unknown_standalone` — status in `{partial_utf8, byte_unmapped,
  special}` (byte fragments — char-membership tool can't decide).
- `family:<S>` — all set bits belong to a single script family `S`
  (Latn, Cyrl, Grek, Han = Hans∪Hant∪Jpan, Hang, Arab, Deva, Hebr,
  Thai, Armn, …).
- `mixed_scripts` — set bits straddle two or more script families.
- `no_script_admits` — `popcount == 0` (token has decoded text but no
  in-scope locale admits any of its chars).

## Family partition of the English firing histogram

| family               | vocab  | fired  | count        | mass %    |
| ---                  | ---:   | ---:   | ---:         | ---:      |
| **family:Latn**      | 88,240 | 83,252 | 758,044,015  | **75.40** |
| **substrate**        |  4,017 |  2,706 | 245,756,611  | **24.45** |
| unknown_standalone   |  2,435 |  1,291 |     620,773  |  0.062    |
| family:Han (CJK)     |  5,102 |  3,778 |     303,442  |  0.030    |
| family:Cyrl          |  7,548 |  6,062 |     165,802  |  0.016    |
| no_script_admits     |    232 |    171 |     120,630  |  0.012    |
| family:Grek          |  1,500 |  1,301 |     118,472  |  0.012    |
| family:Hang (Korean) |  4,439 |  2,324 |      43,144  |  0.004    |
| family:Arab          |  9,341 |  4,474 |      35,729  |  0.004    |
| family:Deva (Hindi)  |  1,529 |  1,225 |      30,400  |  0.003    |
| family:Hebr          |    961 |    851 |      28,943  |  0.003    |
| family:Thai          |    560 |    504 |      25,446  |  0.003    |
| 12 other Indic/Caucasian | …  | …      | < 9,000 each  | < 0.001 each |

Reading: English text in these 8 sources is **75.4 % Latin-letter
content + 24.5 % substrate**. Everything else (every Cyrillic letter,
every Han ideogram, every Greek letter that fired) sums to under
**0.10 %** of English-sample mass. That's the upper bound on
"foreign-language fragment" leakage in the English split of the run.

## What is in each foreign-family slice?

These are *quotations within English text*:

- **family:Han** — 学校 (school), 中 (middle), 大 (big), 王 (king),
  山 (mountain), 田 (field), 村 (village), 子 (child), の (no), 道 (way).
  Wikipedia-style place names and CJK proper nouns inside English
  articles, plus one hiragana token.
- **family:Cyrl** — single Cyrillic letters (С, а, и, е, на, к, я)
  and short bigrams. Russian/Bulgarian/Ukrainian alphabet showing
  through where English text quotes a Slavic name or word.
- **family:Grek** — single Greek letters: α, β, μ, σ, ε, ν, λ, ς, γ,
  δ, Δ. These are not Greek text — they are the **mathematical /
  physics symbols** that English STEM text uses (Wikipedia math
  formulas, scientific papers). The fact that the top Greek-family
  tokens are all single letters and the mass is concentrated on
  `α, β, μ, σ` confirms this.

This is the "yes, other languages are quoted in English" picture you
described. The bitmask cleanly separates them.

## Within the Latin family: firing-rate distribution

83,252 Latin-family tokens fire in English, carrying 758 M counts
(75.40 % of English mass). The distribution is heavy-tailed (Zipfian).

### Rank-frequency cutoffs

| top-N rank | last count | cum mass    | % of Latin | % of English |
| ---:       | ---:       | ---:        | ---:       | ---:          |
| 10         | 5,160,589  | 144,537,773 |  19.07     | 14.38         |
| 100        |   510,692  | 253,248,465 |  33.41     | 25.19         |
| 500        |   159,083  | 359,798,711 |  47.46     | 35.79         |
| 1,000      |    89,938  | 418,362,137 |  55.19     | 41.61         |
| 2,500      |    40,455  | 505,947,773 |  66.74     | 50.33         |
| **5,000**  |    20,722  | 577,444,863 |  76.18     | 57.44         |
| 10,000     |     9,558  | 646,704,074 |  85.31     | 64.33         |
| 20,000     |     3,981  | 708,104,597 |  93.41     | 70.43         |
| 50,000     |       290  | 756,221,249 |  99.76     | 75.22         |
| 83,252     |         1  | 758,044,015 | 100.00     | 75.40         |

### Mass quantiles (Latin family only)

| quantile | rank reached | last count |
| ---:     | ---:         | ---:       |
| 50 %     | 635          | 130,694    |
| 80 %     | 6,647        | 15,191     |
| 90 %     | 14,717       | 5,991      |
| 95 %     | 23,397       | 3,132      |
| 99 %     | 39,781       | 904        |
| 99.9 %   | 55,357       | 126        |

### Count-threshold cutoffs

| count ≥  | tokens   | cum mass    | % of Latin | % of English |
| ---:     | ---:     | ---:        | ---:       | ---:         |
| 1        | 83,252   | 758,044,015 | 100.000    | 75.40        |
| 5        | 75,666   | 758,027,280 |  99.998    | 75.40        |
| 10       | 71,406   | 757,998,471 |  99.994    | 75.40        |
| 100      | 56,776   | 757,445,347 |  99.921    | 75.34        |
| 1,000    | 38,643   | 749,382,520 |  98.857    | 74.54        |
| 10,000   |  9,614   | 642,927,565 |  84.814    | 63.95        |
| 100,000  |    890   | 407,939,164 |  53.815    | 40.58        |
| 1,000,000|     45   | 218,064,955 |  28.767    | 21.69        |

### Reading the cutoff

The argument for "the solid group of English tokens":

- **Top 6,647 Latin tokens** (count ≥ 15,191) already account for
  **80 % of Latin mass**. Same set covers 60 % of total English bytes.
- Going down to **count ≥ 100** gathers 56,776 tokens with **99.92 %**
  of Latin mass — the long tail below count=100 contributes 0.08 %.
- The bottom **27,500 tokens** (counts 1–125) together carry only
  **0.1 %** of Latin mass — these are exactly where the loanwords,
  proper names, OCR artifacts, and code identifiers concentrate.

So the count distribution gives a graceful menu of cutoffs rather than
a single sharp knee — the log-log curve is approximately linear over
most of its range. Where to set the cutoff depends on what we want
the "English token set" for downstream (embedding-norm analysis vs.
training-coverage estimation, etc.).

## How the bitmask is used here (exclusion, not mapping)

- **Excluded by family**: 1,301 Greek-script tokens, 6,062 Cyrillic
  tokens, 3,778 Han tokens, 2,324 Hangul tokens, 4,474 Arabic tokens,
  etc. that fired in English. The bitmask correctly rules them out
  as English content — they are quotations.
- **Excluded by no-script-admits**: 171 decoded tokens fired in
  English whose chars are not admitted by any in-scope locale
  (orphan diacritics like `ō`, exotic punctuation). 0.012 % mass.
- **Kept as substrate**: 2,706 universal tokens. They co-fire in every
  language; the bitmask doesn't decide their identity, the firing rate
  does — and in English they carry 24.45 % of mass, in Greek 8.69 %.

The remaining 83,252 Latin-family tokens are the candidate pool. The
firing-rate distribution decides which subset of them counts as the
"solid English vocabulary."

## Outputs

- `tables/family_firing_summary.tsv` — the family partition.
- `tables/latin_rank_frequency.tsv` — full rank-frequency curve for the
  Latin family (83,252 rows, one per fired token).
- `tables/latin_cumulative_mass.tsv` — cumulative mass at rank /
  count-threshold cutoffs.
- `tables/family_top20_*.tsv` — top 20 tokens per family (one file per
  family; useful for inspecting what each family slice actually contains).
- `firing_rate_summary.json` — single-glance numbers.

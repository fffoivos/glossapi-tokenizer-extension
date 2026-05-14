# German: firing-rate mapping by script family + overlap with English

Same framing as english_review: bitmask for exclusion, firing rate
for mapping. German = bit 3 (`de`); English = bit 0 (`en`). German
sample: 1,005,777,069 tokens (1 B cap).

## Family partition of the German firing histogram

| family               | vocab  | fired  | count        | mass %    |
| ---                  | ---:   | ---:   | ---:         | ---:      |
| **family:Latn**      | 88,240 | 85,332 | 851,108,565  | **84.62** |
| **substrate**        |  4,017 |  2,843 | 153,498,172  | **15.26** |
| unknown_standalone   |  2,435 |  1,026 |     510,829  |  0.051    |
| family:Han           |  5,102 |  4,261 |     249,744  |  0.025    |
| family:Cyrl          |  7,548 |  6,381 |     173,064  |  0.017    |
| family:Arab          |  9,341 |  5,673 |     100,840  |  0.010    |
| family:Grek          |  1,500 |  1,320 |      58,490  |  0.006    |
| family:Hang          |  4,439 |  3,088 |      44,187  |  0.004    |
| family:Hebr          |    961 |    835 |      18,076  |  0.002    |
| every other family   | …      | …      | < 10,000 each | < 0.001 each |

So German is also **Latin + substrate** (99.88% combined). Two
differences from English's split:

1. **German has more Latin and less substrate** (84.6 / 15.3 vs
   English's 75.4 / 24.4). Same number of total tokens, but each German
   text-byte spends proportionally more on letters and less on
   punctuation/digits/whitespace. This is the German-compound-noun
   imprint: longer words on average → fewer punctuation tokens per byte.
2. **Foreign-script quotation is similar in scale** (0.13 % vs
   English's 0.10 %). Slightly more Han, Cyrillic, Arabic content in
   the German feeds.

## Latin family — rank-frequency (German)

85,332 Latin-family tokens fired, carrying 851 M counts (84.62 % of
the German sample).

| top-N | last count | cum mass    | % of Latin | % of German |
| ---:  | ---:       | ---:        | ---:       | ---:        |
| 10    | 4,901,281  |  90,185,060 |  10.60     |  8.97       |
| 100   |   901,888  | 253,200,594 |  29.75     | 25.17       |
| 500   |   232,466  | 416,998,740 |  49.00     | 41.46       |
| 1,000 |   128,601  | 500,857,720 |  58.85     | 49.80       |
| 2,500 |    56,298  | 625,362,856 |  73.48     | 62.18       |
| **3,701** | 38,285 |       — 80.00 % of Latin       | (80 % quantile rank) |
| 5,000 |    27,296  | 722,712,207 |  84.91     | 71.86       |
| 10,000|     8,047  | 801,147,433 |  94.13     | 79.65       |
| 20,000|     1,493  | 837,129,436 |  98.36     | 83.23       |
| 50,000|        69  | 850,367,675 |  99.91     | 84.55       |

**Mass quantiles** (Latin family):

| quantile | rank      | last count |
| ---:     | ---:      | ---:       |
| 50 %     |     539   | 213,457    |
| 80 %     |   3,701   |  38,285    |
| 90 %     |   7,015   |  16,611    |
| 95 %     |  11,030   |   6,399    |
| 99 %     |  24,793   |     867    |
| 99.9 %   |  48,502   |      79    |

German is **more head-concentrated than English**: 80 % of Latin mass
sits in the top 3,701 tokens (vs English 6,647); 99 % in 24,793 (vs
English 39,781). Subword reuse in compounds piles repeated chunks like
`er`, `ung`, `en`, `lich` into a few high-mass types instead of
spreading across many word forms.

Top 15 Latin tokens (all `pc=28`, pure-pan-Latin ASCII):
`der · die · und · in · zu · von · den · ist · en · mit · das · auf
· des · eine · ein`.

## Overlap with English in the Latin family

Set overlap on token IDs (Latin family, fired ≥ 1 in each language):

| metric                       | value      |
| ---                          | ---:       |
| fired in English (Latin)     | 83,252     |
| fired in German  (Latin)     | 85,332     |
| fired in BOTH                | **81,750** |
| fired in English ONLY        |  1,502     |
| fired in German  ONLY        |  3,582     |
| Jaccard                      | **0.9415** |

Mass overlap:

| direction                                         | mass         | share of that lang's Latin mass |
| ---                                               | ---:         | ---:                            |
| German Latin mass on shared tokens                | 844,050,681  | **99.17 %**                     |
| German Latin mass on German-only tokens           |   7,057,884  |  0.83 %                         |
| English Latin mass on shared tokens               | 757,981,193  | **99.99 %**                     |
| English Latin mass on English-only tokens         |      62,822  |  0.008 %                        |

**The English Latin token set is essentially a subset of the German
Latin token set** in mass terms — only 0.008 % of English Latin mass
falls on tokens that didn't fire at all in 1 B German.

But this set overlap masks a huge **frequency-distribution gap**.
Looking at the joint top-50 (`tables/joint_top50_scatter.tsv`):

| token  | count_de    | count_en    | ratio de / en |
| ---    | ---:        | ---:        | ---:          |
| `,`    | 35,709,518  | 38,511,347  | 0.93          |
| `.`    | 23,588,152  | 20,821,847  | 1.13          |
| `the`  |    400,191  | 38,763,713  | **0.010**     |
| `and`  |    271,816  | 18,681,310  | **0.015**     |
| `of`   |    360,809  | 22,416,047  | **0.016**     |
| `to`   |    203,243  | 12,950,594  | 0.016         |
| `a`    |    261,483  | 12,352,702  | 0.021         |
| `der`  | 15,832,409  |     44,919  | **352**       |
| `die`  | 15,528,985  |     22,965  | **676**       |
| `und`  | 15,232,146  |     48,660  | **313**       |
| `der`-family vs `the`-family | | | ratio swings ~30,000× either way |

Substrate fires at roughly comparable rates in both (the comma is
the comma); content tokens swing by **300×–700×** in the language
where they belong. So Jaccard 0.94 is misleading on its own —
"shared" here means "fires ≥ 1 in both", which at 1 B samples is
satisfied by every English function word appearing once in a quoted
German name and vice versa. The **firing-rate ratio** is what carries
the language identity.

### Top-K rank intersection — almost disjoint at the head

| K        | `top-K(de) ∩ top-K(en)` | % of K  |
| ---:     | ---:                    | ---:    |
| 10       | **1**                   | 10 %    |
| 50       | **2**                   | 4 %     |
| 100      | **6**                   | 6 %     |
| 500      | 109                     | 22 %    |
| 1,000    | 210                     | 21 %    |
| 5,000    | 992                     | 20 %    |
| 10,000   | 2,732                   | 27 %    |
| 20,000   | 8,816                   | 44 %    |
| 50,000   | 39,885                  | 80 %    |

Only **2 of the top 50 Latin tokens** are shared between the two
languages. The heads of the two distributions are almost completely
disjoint; convergence only happens in the long tail (50,000+).

### German Latin mass partitioned by rate class

For each Latin token with `count_de > 0` or `count_en > 0`, take
`log10((count_de + 0.5) / (count_en + 0.5))` and bin:

| rate class                                | tokens   | German mass | German % | English mass | English % |
| ---                                       | ---:     | ---:        | ---:     | ---:         | ---:      |
| **german-dominant ≥10×**                  |  8,148   | 566,607,045 | **66.57** |  6,278,502  |  0.83     |
| german-leaning 3–10×                      |  6,029   |  92,843,200 |  10.91   | 18,223,453  |  2.40     |
| **comparable (within 3×)**                | 30,125   | 156,980,332 |  18.44   | 160,292,574 | **21.15** |
| english-leaning 3–10×                     | 15,799   |  18,183,969 |   2.14   | 98,131,173  | 12.95     |
| **english-dominant ≥10×**                 | 21,649   |   9,436,135 |   1.11   | 475,055,491 | **62.67** |
| near-singleton-german (en=0)              |  3,582   |   7,057,884 |   0.83   |          0  |  0        |
| near-singleton-english (de=0)             |  1,502   |          0  |   0      |     62,822  |  0.01     |

This is the **real** picture:

- **67 %** of German Latin mass sits on tokens that fire ≥10× more
  often in German than in English. These are the de-dominant tokens
  (`der`, `die`, `und`, `zu`, `von`, …, plus most of the German Latin
  pool by mass).
- **63 %** of English Latin mass sits on tokens that fire ≥10× more
  often in English than German (`the`, `and`, `of`, `to`, …).
- Only ~**18–21 %** of each language's Latin mass is "comparable"
  (rate within 3×). That comparable bucket is mostly shared
  international vocabulary, person/place names, and scientific terms.
- The strict singletons (fired only in the other language) are tiny:
  0.83 % German-only, 0.008 % English-only.

So the distribution shift across the Latin family is:

- Set overlap: ~94 % (almost everything fires at least once in both).
- Top-50 rank overlap: ~4 %.
- Mass with ≥10× rate skew **per language**: ~67 % de side, ~63 % en
  side. Mass with ≥3× skew: ~80 % each side.

The bitmask says "could this character be German"; this rate
partition says "does this token actually behave like German." The
gap between the two is exactly what you flagged.

## German-distinctive tokens (de-cap AND NOT en-cap)

Tokens whose `bitmask_and` has the `de` bit set but **not** the `en`
bit — i.e., they contain at least one character outside strict ASCII
(`ä`, `ö`, `ü`, `ß`, etc. that English's CLDR exemplar rejects).

| metric                                | value      |
| ---                                   | ---:       |
| German-distinctive tokens fired       | 1,431      |
| mass                                  | 52,169,951 |
| % of German total                     |  5.19 %    |
| % of German Latin mass                |  6.13 %    |

Top examples (count_de / count_en):

| token       | count_de  | count_en | popcount |
| ---         | ---:      | ---:     | ---:     |
| `für`       | 3,764,749 |    8,412 | 9        |
| `über`      | 1,837,367 |    1,580 | 9        |
| `können`    | 1,279,661 |       39 | 9        |
| `ä`         |   802,903 |   18,160 | 6        |
| `ü`         |   610,234 |   18,370 | 9        |
| `ität`      |   484,284 |      650 | 6        |
| `ö`         |   410,839 |   22,054 | 9        |
| `während`   |   386,964 |       52 | 6        |
| `müssen`    |   386,928 |       21 | 9        |
| `zurück`    |   369,907 |       72 | 9        |
| `Über`      |   332,122 |      768 | 9        |
| `ß`         |   310,319 |    4,398 | 1        |
| `würde`     |   276,257 |        8 | 9        |
| `daß`       |   260,773 |       65 | 1        |
| `könnte`    |   223,160 |        4 | 9        |

These are unambiguously German function words, modal verbs (`können`,
`müssen`, `würde`, `könnte`), morphological suffixes (`ität`, `är`,
`ür`, `ät`, `üt`), and ß-bearing tokens. Their English firings (1–22k)
are quotations and proper names; the German firings are 100–10,000×
higher.

## German-only Latin tokens (fired in German, NOT in English)

3,582 Latin-family tokens fired in German but zero times in 1 B
English. Notably **most of them are pc=28 — pure-pan-Latin ASCII**.
They're admissible in English by character set; they just don't appear
in 1 B English text:

| token            | count_de | popcount | character note |
| ---              | ---:     | ---:     | ---            |
| `verwendet`      | 222,777  | 27       | "used" (de past participle) |
| `handelt`        | 108,289  | 28       | "deals/acts"               |
| `Fällen`         |  86,779  | 6        | "cases" (umlaut)           |
| `mindestens`     |  86,699  | 28       | "at least"                 |
| `möglicherweise` |  86,102  | 9        | "possibly" (umlaut)        |
| `beispielsweise` |  85,323  | 27       | "for example"              |
| `durchgeführt`   |  83,256  | 9        | "carried out" (umlaut)     |
| `obwohl`         |  76,192  | 27       | "although"                 |
| `Zeitpunkt`      |  71,319  | 28       | "point in time"            |
| `gleichzeitig`   |  68,744  | 28       | "simultaneously"           |
| `ziemlich`       |  62,714  | 28       | "quite"                    |
| `regelmäßig`     |  56,803  | 1        | "regularly" (umlaut + ß)   |

This is the **bitmask-vs-firing distinction in action**: bitmask says
"these tokens *could* be English" (ASCII chars only, pc=28), firing
rate says "but they aren't." The compound-noun BPE merges that German
training surfaces produce long whole-word tokens (`gleichzeitig`,
`Zeitpunkt`, `verwendet`) that simply don't appear in English text.

## English-only Latin tokens (fired in English, NOT in German)

1,502 tokens. Three clusters:

| cluster                       | examples                                                  |
| ---                           | ---                                                       |
| English-specific scientific   | `longitudinally`, `disulfide`, `magnetization`, `bilayer`, `householder`, `passageway`, `recessed`, `announcers` |
| Tabular-data delimiters       | `\tL`, `\tC`, `\tT`, `\tJ`, `\tD` (tab + capital — table-cell markers from English HTML/CSV) |
| Unit-attached numbers         | `\xa0cm`, `\xa0mg`, `\xa0nm`, `\xa0years`, `\xa0days` (NBSP+unit — Wikipedia infobox style)  |

Same logic as the German-only set: pc=28 by character, but the BPE
merges that surface during English training don't appear in German
training.

## What the bitmask buys, what the firing rate buys

- **Bitmask (exclusion).** Cleanly removes ~0.13 % foreign-script
  contamination + ~0.01 % no-script-admits orphans. Substrate is left
  in. This is what the family partition gives us.
- **Bitmask (rule-shape).** The 1,431 ä/ö/ü/ß-bearing tokens
  (de-cap AND NOT en-cap) define a tight 6.13 % "German-distinctive"
  shell. By construction they cannot be misattributed to English.
- **Firing rate (mapping).** Within the Latin family, 94 % of token IDs
  fire in both languages. The mapping is in the *ratio*, not the set.
  Function words `der/die/und` are ~300–700× more frequent in
  German; `the/and/of` ~50–100× more frequent in English. A
  log-ratio-based or count-threshold-based rule on top of the
  Latin-family partition is what separates the two languages.

## Outputs

- `tables/family_firing_summary.tsv`
- `tables/latin_rank_frequency.tsv`
- `tables/latin_cumulative_mass.tsv`
- `tables/family_top20_*.tsv`
- `tables/en_de_overlap_summary.tsv`
- `tables/en_de_overlap_by_count.tsv`
- `tables/german_distinctive_top200.tsv` — ä/ö/ü/ß-bearing top fired
- `tables/german_only_top200.tsv` — fired in German, zero in English
- `tables/english_only_top200.tsv` — fired in English, zero in German
- `tables/joint_top50_scatter.tsv` — top tokens of either lang with
  both counts side by side
- `firing_rate_summary.json` — single-glance numbers

# English-membership: bitmask × empirical firing

Companion to `greek_review/membership_report.md`, applied to
`eng_Latn`. The dataset slice is the merged English firing histogram
across all 8 documented Apertus English sources (FineWeb-Edu, FineWeb-HQ,
DCLM-Edu, Clean-Wikipedia, EuroParl, ParaDocs, FineWeb-2, FineWeb-2-HQ).

The bitmask side is `02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet`:
55 (language, script, encoding) triples × 131,072 vocab tokens.
**English = bit 0** (`en`). The bit is set when every character of a
token's decoded text is in the CLDR-`en` exemplar set, which is strict
ASCII `[A-Za-z]`.

## Sample size

- **English sample tokens**: 1,005,327,142 (capped at 1 B during the run).
- **Vocab fired**: 111,086 / 131,072 = 84.8 %. English fires more of the
  vocab than Greek (76.3 %).

## Confusion vs "fired in English" (decoded / evaluable tokens only)

| classifier                          | TP     | FP    | FN     | TN     | P     | R     | mass % |
| ---                                 | ---:   | ---:  | ---:   | ---:   | ---:  | ---:  | ---:   |
| **en-capable (bit 0)**              | 77,014 | 5,229 | 32,781 | 13,613 | 0.936 | 0.701 | 99.51  |
| en-capable AND NOT substrate        | 74,308 | 3,918 | 35,487 | 14,924 | 0.950 | 0.677 | 75.07  |
| en-capable AND popcount == 28       | 68,665 | 3,523 | 41,130 | 15,319 | 0.951 | 0.625 | 69.07  |
| Latin-script flag (has_latin)       | 83,289 | 5,001 | 26,506 | 13,841 | 0.943 | 0.759 | 75.42  |
| Latin-script AND NOT structural     | 83,287 | 5,001 | 26,508 | 13,841 | 0.943 | 0.759 | 75.41  |

`en-capable` covers **99.51 %** of English-sample mass — the highest
single-rule coverage we see in this project. The 0.49 % gap is the
0.43 % hard rejection plus 0.06 % unknown-standalone.

## Popcount distribution of fired English tokens

The bitmask-popcount distribution is *trimodal* for English. Almost all
mass lives in three buckets:

| popcount | tokens  | mass         | mass % | meaning                                                  |
| ---:     | ---:    | ---:         | ---:   | ---                                                      |
| 0        | 1,462   | 741,403      | 0.074  | no in-scope language admits any char (orphan diacritics) |
| 1        | 12,312  | 732,562      | 0.073  | exactly one locale admits — language-distinctive         |
| 2–13     | 19,837  | 3,425,204    | 0.341  | small-cluster Latin diacritics + a few Cyrillic / Greek  |
| **27**   | 5,643   | 60,312,545   | **6.00** | Latin minus one locale (often Turkish ı or Polish-ish)  |
| **28**   | 68,665  | 694,358,757  | **69.07** | **all 28 Latin-script locales admit — pan-Latin**     |
| **55**   | 2,706   | 245,756,611  | **24.45** | every locale admits — substrate (punct/digit/ws)      |

Two consequences:

1. **`pc=1` is empty for English** in mass terms (0.07 %). English's
   CLDR exemplar is the strict ASCII alphabet, which is a subset of
   every other Latin-script locale's exemplar. So nothing is "uniquely
   English" in the language-membership sense — by contrast Greek's
   alphabet is exclusive to Greek so the `pc=2` ({el, el-polyton})
   bucket carries 88.55 % of Greek mass.
2. **`pc=28` is the natural English signature** — a token whose chars are
   admitted by every Latin-script locale. 69 % of English mass lives in
   this bucket.

## Hard leakage: decoded text fired but NOT en-capable

**32,781 tokens** fired in English without the `en` bit set. Mass =
**0.426 %**, much smaller than Greek's 2.38 %. Breakdown:

| class (token-metadata script)         | tokens | mass      | mass % |
| ---                                   | ---:   | ---:      | ---:   |
| Latin diacritic (loanwords / names)   | 8,974  | 3,488,474 | 0.347  |
| Other / multi-script                  | 16,338 |   502,890 | 0.050  |
| Cyrillic (foreign excerpts)           | 6,163  |   166,877 | 0.017  |
| Greek (foreign excerpts)              | 1,306  |   120,215 | 0.012  |
| Structural / pure digits              |      0 |         0 | 0      |

Top hard-rejected tokens are diacritic-bearing Latin letters and the
words built from them:

| token   | count  | popcount | locales admitting                                |
| ---     | ---:   | ---:     | ---                                              |
| `ō`     | 90,044 | 0        | (none — orphan; not in any in-scope exemplar)    |
| `é`     | 80,443 | 13       | cs, es, fr, hu, it, nl, pt, sv, vi, nb, sk, ca, is |
| `ā`     | 55,334 | 1        | lv                                               |
| `á`     | 40,206 | 8        | cs, es, hu, nl, pt, vi, sk, is                   |
| `í`     | 36,698 | 9        | cs, es, hu, nl, pt, vi, sk, ca, is               |
| `ū`     | 32,872 | 2        | lt, lv                                           |
| `ć`     | 28,521 | 4        | pl, sr-Latn, sl, hr                              |
| `ović`  | 26,062 | 4        | pl, sr-Latn, sl, hr                              |
| `José`  | 25,126 | 13       | cs, es, fr, hu, it, nl, pt, sv, vi, nb, sk, ca, is |
| `š`     | 23,334 | 9        | cs, sr-Latn, fi, sl, hr, sk, et, lt, lv          |

These are exactly the "foreign-language fragments in English" you'd
expect at the long tail of a 1 B-token English sample: place names,
person names, loanwords (`café` → `c` + `afé`-like sequences), and
quotations.

## Unknown-standalone tokens

**1,291 tokens** fired in English with status in
`{partial_utf8, byte_unmapped, special}`. Mass = **0.062 %**.
These are byte-fragment tokens that only have a defined Unicode meaning
in context with neighbours. Not a hard rejection — the char-membership
tool can't evaluate them standalone.

## Three rule definitions for "English tokens"

| group                       | predicate                                | vocab size | fired   | mass % |
| ---                         | ---                                      | ---:       | ---:    | ---:   |
| **English MAXIMAL**         | `bit 0 set in bitmask_and`               | 82,243     | 77,014  | 99.51  |
| **English DISTINCTIVE**     | `bit 0 set AND popcount < 55`            | 78,226     | 74,308  | 75.07  |
| **English pure-pan-Latin**  | `bit 0 set AND popcount == 28`           | 72,188     | 68,665  | 69.07  |
| script-flag-only (legacy)   | any Latin codepoint in decoded           | 89,313     | 83,317  | 75.42  |

Notes:

- **MAXIMAL** is the right choice when you want every token participating
  in English text, including the punctuation / digit / whitespace
  substrate. Use this for analyses where substrate co-fires with the
  language (embedding-norm / LM-head cluster studies).
- **DISTINCTIVE** strips the 2,706 substrate tokens (24.45 % mass) and
  keeps the Latin-letter content (75.07 %). Use this for "English-letter
  vocabulary" analyses.
- **Pure-pan-Latin** is `pc == 28`, which intentionally drops the
  Latin-minus-one bucket (5,643 tokens, 6.00 % mass; mostly tokens that
  drop Turkish or one other locale because of a single odd letter).
  Coverage gap vs DISTINCTIVE is exactly that 6 %.
- The legacy **script-flag-only** rule captures more vocab (89,313) but
  *less* mass than MAXIMAL (75.42 % vs 99.51 %) because it excludes
  substrate. It also lets through Latin-script foreign tokens (diacritic
  loanwords) that the bitmask correctly excludes.

## Comparison to Greek

| dimension                                  | Greek (`ell_Grek`)  | English (`eng_Latn`)  |
| ---                                        | ---:                | ---:                  |
| Vocab fired                                | 100,014 / 131,072   | 111,086 / 131,072     |
| Language-capable in vocab                  | 5,518 (el)          | 82,243 (en)           |
| Language-capable fired                     | 3,659               | 77,014                |
| Language-capable mass coverage             | 97.24 %             | 99.51 %               |
| Script-flag mass coverage                  | 88.56 %             | 75.42 %               |
| Substrate share of dataset (pc=55 mass)    | 8.69 %              | 24.45 %               |
| Hard rejection (decoded leakage) mass      | 2.38 %              | 0.43 %                |
| Unknown standalone mass                    | 0.38 %              | 0.06 %                |
| Pan-script-family bucket (pc=28 / pc=N)    | 2.17 %              | 69.07 %               |
| Script-distinctive bucket mass             | 88.55 % (pc=2)      | 0.07 % (pc=1)         |

The numbers tell three stories:

1. **Greek is script-distinctive, English is not.** Greek tokens at
   `pc=2` — admitted only by `el` and `el-polyton` — hold **88.55 %** of
   Greek-sample mass. That popcount-2 bucket is the Greek alphabet
   speaking for itself. English's analogous bucket is `pc=1` (only `en`
   admits) and it holds **0.07 %** of mass: English-only tokens are
   essentially absent because anything that admits ASCII admits
   `de`, `fr`, `it`, ... too. Instead, English mass concentrates at
   `pc=28` (every Latin locale admits), 69.07 %.
2. **English has 2.8× more substrate share than Greek** (24.45 % vs
   8.69 %). This is the BPE-fertility imprint: an English token wraps
   ~4 chars on average, so per English byte you emit fewer letter
   tokens and proportionally more punctuation / digit tokens. Greek's
   fertility is closer to 2 chars / token, so substrate dilutes less.
3. **English leaks less than Greek** (0.43 % vs 2.38 %). The leakage in
   English is mostly Latin diacritic loanwords (`ō`, `é`, `á`, `š`,
   `ć`, …) plus small amounts of Cyrillic and Greek excerpts. Greek's
   leakage is overwhelmingly Latin code/loanwords that fire in Greek
   text. Both directions are real but English is cleaner per byte
   because English text in these corpora is more monolingual than
   Greek text.

## Outputs

- `tables/confusion.tsv` — 5-row confusion matrix.
- `tables/popcount_distribution.tsv` — per-popcount counts and mass.
- `tables/leakage_top200.tsv` — top decoded-text hard rejections (the
  loanwords / foreign names / foreign-script excerpts).
- `tables/unknown_standalone_top200.tsv` — top partial-UTF8 / non-text
  fired tokens.
- `tables/dormant_en_capable.tsv` — all 5,229 en-capable tokens that did
  not fire in 1 B English.
- `tables/rule_summary.tsv` — the three rule definitions side by side.
- `summary.json` — numbers cited in this report.

# Greek-membership: bitmask × empirical firing

Joins `02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet`
(per-token 55-bit "(language, script, encoding) triples this token's chars
admit") with the Apertus-Greek firing histogram from `outputs/`. Greek =
bit 16 (`el`) and bit 22 (`el-polyton`).

Important correction: `partial_utf8`, `byte_unmapped`, and `special`
tokens are **not** hard language-membership rejections. They have zero
bitmasks because the char-level tool cannot evaluate them as standalone
decoded text. The leakage numbers below count only decoded/evaluable text.

## Confusion: bitmask vs script-flag, against "fired in Greek"

| classifier (predicate)                    | TP    | FP    | FN     | TN     | P     | R     | mass % |
| ---                                       | ---:  | ---:  | ---:   | ---:   | ---:  | ---:  | ---:   |
| script-flag `has_greek_mono ∨ poly`       | 1,504 | 3     | 97,709 | 29,421 | 0.998 | 0.015 | 88.56  |
| **el-capable (bit 16)**                   | 3,659 | 1,859 | 95,554 | 27,565 | 0.663 | 0.037 | 97.24  |
| el-poly-capable (bit 22)                  | 3,659 | 1,859 | 95,554 | 27,565 | 0.663 | 0.037 | 97.24  |
| either el or el-poly                      | 3,659 | 1,859 | 95,554 | 27,565 | 0.663 | 0.037 | 97.24  |
| el-capable AND NOT substrate              | 1,500 | 1     | 97,713 | 29,423 | 0.999 | 0.015 | 88.55  |

P/R are computed against the predicate "this decoded/evaluable token fired
>= 1 time in 1 B Greek". The bitmask is not trying to predict empirical
firing; it says "this token's decoded chars are admissible in Greek text".
So the load-bearing column is **mass %**: the share of Greek-sample tokens
explained by tokens the classifier admits.

`el-capable` mass coverage = **97.24 %**, vs **88.56 %** for the
script-flag. The 8.7 % gap is substrate-shared infrastructure
(punctuation, digits, whitespace): the bitmask correctly admits substrate
into Greek's membership because it is admissible in every language.

## What the bitmask buys us

### Maximal Greek membership

5,518 tokens in the Apertus vocab are Greek-compatible by chars. 3,659 of
those fired at least once in the 1 B-token Greek sample; 1,859 are dormant
(Greek-capable but unused in this particular Greek corpus). The dormant set
is in `tables/dormant_el_capable.tsv`.

### Hard leakage: decoded text fired but not Greek-compatible

95,554 decoded/evaluable tokens fired in Greek but their chars are not all
admissible in Greek. These tokens carry **2.38 %** of Greek-sample mass.

| class                                              | tokens | mass       | mass % |
| ---                                                | ---:   | ---:       | ---:   |
| Latin-letter tokens (loanwords, names, code, URLs) | 79,740 | 23,721,092 | 2.365  |
| Other decoded text / unmodeled letters             | 15,814 |    179,117 | 0.018  |
| Structural / pure digits                           | 0      |          0 | 0      |

Full per-token list (top 200 by Greek count) in
`tables/leakage_top200.tsv`. The top entries are English/Latin fragments
such as `the`, `of`, `and`, and single Latin capitals.

### Unknown standalone tokens

801 tokens fired in Greek but are `partial_utf8`/non-text as standalone
vocab entries. They carry **0.38 %** of Greek-sample mass. These are not
"not Greek"; they are byte-level fragments whose Unicode character only
exists in context with neighboring tokens. The top list is in
`tables/unknown_standalone_top200.tsv`.

## Three token-group definitions

| group name                 | membership rule                               | size (vocab) | size (fired) | mass captured |
| ---                        | ---                                           | ---:         | ---:         | ---:          |
| **Greek MAXIMAL**          | bit 16 OR bit 22 set in `bitmask_and`         | 5,518        | 3,659        | 97.24 %       |
| **Greek DISTINCTIVE**      | bit 16 set AND popcount(`bitmask_and`) < 55   | ~1,510       | 1,500        | 88.55 %       |
| **Greek script-flag-only** | token decoded contains >= 1 Greek codepoint   | 1,507        | 1,504        | 88.56 %       |

- **MAXIMAL** is best for embedding-norm/cluster analyses where substrate
  tokens should be included.
- **DISTINCTIVE** is best for "Greek-specific vocab" analyses. It strips
  substrate and keeps Greek-letter tokens.
- **script-flag-only** is the old baseline and is now mostly redundant.

## Outputs

- `tables/confusion.tsv` — the 5-row confusion matrix.
- `tables/leakage_top200.tsv` — top decoded text fired in Greek but not
  Greek-compatible.
- `tables/unknown_standalone_top200.tsv` — top partial-UTF8/non-text
  tokens that fired in Greek but cannot be evaluated standalone.
- `tables/dormant_el_capable.tsv` — all 1,859 Greek-capable-but-unused
  tokens.

## Interactive app

`http://greek-tokens.localhost:8080/` — Plotly chart + Tabulator table
with per-token bitmask fields. The filter row has tri-state chips for
`el-capable`, `el-poly-capable`, `Greek-only` (popcount <= 2), `substrate`
(popcount = 55), and `no-language` (popcount = 0). Use the status column
to keep `partial_utf8` separate from true decoded-text rejection.

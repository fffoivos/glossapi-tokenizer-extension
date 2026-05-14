# German: tiered dataset-anchored attribution

Companion to `firing_rate_mapping_report.md`. Implements the tier
policy from
`02_2_tokenizer_implementation/02_2_3_token_classification/PLAN.md` for the
German dataset. The point of the tiering is to separate **char-
evidenced German** from **could-be-German-under-premise**, so a
reader can see at a glance how much of the German firing mass is
char-certified vs how much is defaulted-to-German on the working
assumption that this is a German corpus.

## The premise (explicit and falsifiable)

> "In the German dataset, a token whose chars are all de-admissible
> (`bitmask_and` has the `de` bit set) is **provisionally** attributed
> to German. This is a working assumption that may be false for any
> individual token — common false attributions include English
> loanwords, pure-ASCII code identifiers, and proper names from sister
> Germanic locales. The premise is *not* invoked for tiers T0, T1, T3,
> T4, or T5."

T0 / T1 are char-evidenced — no premise. T2 is where the premise lives.

## Tier results (1 B German sample)

| tier                              | tokens | mass         | mass %  | meaning |
| ---                               | ---:   | ---:         | ---:    | ---     |
| **T0 definitely-de**              | **103**| **4,963,019**| **0.49**| `bitmask_and == {de} only` — char-evidenced, all ß-bearing |
| T1 definitely-germanic-latn-only  | **0**  | 0            | 0.00    | `family_and == {Germanic-Latn} only` AND has de-bit. **Empty — see note below.** |
| **T2 premise-de**                 | 77,645 | 845,472,197  | **84.06**| `bitmask_and` has de-bit; chars also admit other locales — defaulted to de under premise |
| **T3 substrate**                  | 2,843  | 153,498,172  | **15.26**| every in-scope locale admits all chars — punctuation, digits, whitespace |
| T4 excluded (non-de char)         | 31,824 | 1,342,567    | 0.13    | char evidence rules de out — token has at least one non-de char |
| T5 unknown standalone             | 1,026  | 510,829      | 0.05    | char tool cannot evaluate (partial_utf8 / byte_unmapped / special) |

Sum: 99.99 % of the German sample is accounted for.

**T4 subdivides:**
- `non_de_latin_char` — 7,585 tokens, 0.07 % (Latin tokens with chars
  like Polish `ć`, Spanish `ñ`, Czech `š`, French `ç` — fire in German
  text as quoted names/loanwords, but de's exemplar rejects them).
- `foreign_script` — 24,128 tokens, 0.07 % (Cyrillic / Greek / Han /
  Arabic / etc. quoted in German text).
- `no_in_scope_locale_admits` — 111 tokens, ~0 % (orphan glyphs).

## What "definitely German by chars" actually looks like

The 103 T0 tokens are all `ß`-bearing — ß is the only character in the
55-locale exemplar union that is exclusively German. Top 15 by firing
count:

| rank | decoded       | count    |
| ---: | ---           | ---:     |
| 1    | `ß`           | 310,319  |
| 2    | ` daß`        | 260,773  |
| 3    | `ßen`         | 193,069  |
| 4    | `ße`          | 186,455  |
| 5    | ` großen`     | 176,276  |
| 6    | ` große`      | 174,054  |
| 7    | ` weiß`       | 140,782  |
| 8    | ` groß`       | 128,954  |
| 9    | `iß`          | 122,562  |
| 10   | `uß`          | 121,074  |
| 11   | `äß`          | 100,727  |
| 12   | ` heißt`      |  95,210  |
| 13   | ` äuß`        |  88,148  |
| 14   | ` schließlich`|  84,870  |
| 15   | `ießen`       |  82,821  |

Collectively the 103 T0 tokens carry 0.49 % of total German mass —
char-certified, no premise needed.

## Why T1 is empty (a finding, not a bug)

In the abstract, T1 would catch tokens whose chars narrow to the
Germanic-Latn family (en/de/nl/da/nb/sv/is) but no further. Of the
171 vocab tokens with `family_and == {Germanic-Latn} only`, **none
fire in the German sample AND have de-bit set in `bitmask_and`**.

Empirically, when the family layer narrows to Germanic-Latn only,
the language layer typically narrows further to a single specific
locale (e.g. Icelandic `þ` → `bitmask_and = {is}` → T0 for Icelandic,
T4 for German). So the family tier doesn't gain us a useful
intermediate for German — the chars either pin to `de` exclusively
(T0) or admit `de` along with several non-Germanic locales (T2).

This is a structural finding about how CLDR exemplars stack: per-
language exemplars are typically either character-specific (yielding
language-popcount-1 narrowing) or share broadly across script
families (yielding many-locale popcount). The Germanic-Latn family
doesn't carve out its own popcount-equals-family-size band the way
one might naïvely expect.

## What the tiering changes about the earlier analysis

The earlier `firing_rate_mapping_report.md` reported "84.62 % Latin
family" as a single bucket. The tiered view decomposes that 84.62 %
as:

- 0.49 % T0 — char-evidenced de.
- 84.06 % T2 — could-be-de under premise.
- 0.07 % within T4 — non-de Latin (rejected from de-pool).

The 6.13 % "German-distinctive (de-cap AND NOT en-cap)" finding from
the earlier report is a **subset of T0 ∪ T2**: it's the umlaut /
ß-bearing tokens whose chars admit de but not en. It contains all of
T0 (the ß-only-de set, 0.49 %) plus the 5.6 % of T2 that has ä/ö/ü.

## Comparison to English (for the same framework)

Running the same tiering on the English dataset (using L = en,
family = Germanic-Latn, premise = "in this dataset, en-admissible
tokens default to en"):

| tier                   | English | German  |
| ---                    | ---:    | ---:    |
| T0 definitely-L        | **0**   | 103     |
| T1 definitely-family   | 0       | 0       |
| T2 premise-L           | high    | 77,645  |
| T3 substrate           | 2,706   | 2,843   |
| T4 excluded            | 32,781  | 31,824  |
| T5 unknown standalone  | 1,291   | 1,026   |

**T0 for English is structurally empty** — there is no character in
the 55-locale exemplar union that is exclusively English. English's
CLDR exemplar `[A-Za-z]` is a subset of every other Latin locale's
exemplar (since every Latin locale's exemplar includes basic Latin
letters). So 100 % of English's "could-be-English" mass lives at T2
under the premise. The English equivalent of "we know it's German
because of the ß" simply doesn't exist.

This is one of the headline asymmetries the framework surfaces:
**German has a tiny but inviolable char-distinctive signature; English
has none.**

## Outputs

- `tables/tier_summary.tsv` — six-row summary (tokens, mass, mass %).
- `tables/tier_top200_T0_definitely_de.tsv` — all 103 char-evidenced
  German tokens, ranked by firing count.
- `tables/tier_top200_T2_premise_de.tsv` — top 200 premise-defaulted
  German tokens (this is where `der`, `die`, `und`, etc. live).
- `tables/tier_top200_T3_substrate.tsv` — top 200 substrate tokens.
- `tables/tier_top200_T4_excluded.tsv` — top 200 char-excluded tokens
  (the loanwords / foreign-script / orphan-glyph set).
- `tables/tier_top200_T5_unknown.tsv` — top 200 unknown-standalone.
- `tier_summary.json` — single-glance numbers + premise text +
  schema-version pin.

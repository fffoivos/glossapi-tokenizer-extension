# Plan — char_language_membership (v2)

## Purpose

For each codepoint that the Apertus-8B-2509 vocab might decode to, emit
a bitmask whose set bits enumerate **(language, script, encoding)**
triples that could plausibly produce that codepoint in normal text.
Aggregated across a token's codepoints via AND, the bitmask answers a
single question: **which (language, script, encoding) triples can we
rule out for this token?**

The artifact is built for **rejection**, not classification. It does
not assign a token to a single language; it tells the downstream
consumer "this token's chars are admissible in this set of triples;
everything else is excluded."

## Scope unit: (language, script, encoding)

A bit corresponds to one CLDR locale that pins all three of:

- **Language** — e.g. `el` (Greek), `sr` (Serbian).
- **Script** — ISO 15924 code (e.g. `Latn`, `Cyrl`, `Grek`, `Hans`,
  `Hant`, `Hang`, `Deva`).
- **Encoding (variant)** — when one (language, script) has two
  meaningfully different orthographies, we give each its own bit:
  `el` vs `el-polyton` (monotonic vs polytonic Greek); `sr-Latn` vs
  `sr-Cyrl` (the two Serbian scripts are encoded as separate locales
  in CLDR); `zh-Hans` vs `zh-Hant`.

The 55 triples currently in `languages.yaml` (bits 0–53) are listed
under § Scope below.

## How we approach the missing-tokenizer-training-data problem

The Apertus-8B-2509 tokenizer was **not retrained on Apertus's pretrain
data**. Apertus inherited Mistral-Nemo-Base-2407's `tekken` tokenizer
wholesale (paper §2.2; verified in
`docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`). So the
merges in the vocab were learned on **Mistral-Nemo's pretrain corpus**,
not on Apertus's.

Mistral-Nemo's tokenizer was, by Mistral's own writeup, trained on
"over 100 languages". **Mistral has not published the actual list.**
Only 11 principal languages are publicly named:

> English, French, German, Spanish, Italian, Portuguese, Chinese,
> Japanese, Korean, Arabic, Hindi.

The remaining ~90+ languages are part of Mistral's proprietary
training mix. Primary-source attempts (Mistral blog, HuggingFace
model card, Mistral docs, NVIDIA technical blog) all stop at the
same 11-language list. We cannot enumerate Mistral's coverage from
primary sources.

**Our approach to this gap**: use **Apertus's documented pretrain
mix** as a proxy for Mistral-Nemo's coverage. The proxy rests on
two assumptions, neither provable but both defensible:

1. Mistral-Nemo and Apertus draw from the same family of
   CommonCrawl-derived multilingual web crawls — FineWeb-2-style.
2. No major language is in Mistral's training that's deliberately
   excluded from Apertus's. (Apertus's general FineWeb-2 random-33%
   sweep covers all 1,811 languages FineWeb-2 has tags for; the
   chance that Mistral covered a major language that FineWeb-2
   doesn't is low.)

This proxy assumption is documented in `manifest.json` under
`scope_proxy_assumption`. The **falsifiable** part of the assumption
is the audit: every script appearing with non-trivial token presence
in the actual Apertus vocab should be covered by a bit in
`languages.yaml`. `scripts/validate.py` runs the empirical audit
after every build.

## Strict rule

A bit gets set on a codepoint **only via positive CLDR evidence**,
plus four deterministic closures defined below. Never speculative.
A letter codepoint that no rule places into any language's set
gets 0 bits — and under AND-aggregation that 0 rejects every
in-scope language for tokens containing it. This is the desired
behaviour: a Cherokee letter `Ꭰ` in a token rules out every
(language, script, encoding) we model, because none of them admits
Cherokee.

The four deterministic closures applied on top of raw CLDR data:

1. **Script-compatibility filter** — a codepoint contributes to a
   language only if its Unicode script (derived from character-name
   prefix: `LATIN`, `GREEK`, `CYRILLIC`, `ARABIC`, `CJK`, `HIRAGANA`,
   `HANGUL`, …) is admissible by the locale's declared script.
   Suppresses CLDR cross-script bleed (zh-Hans's `index` set being
   `[A B C … Z]` would otherwise give Latin A the `zh-Hans` bit).
2. **Case closure** — for every codepoint in a language's letter
   set, also add `str.upper()` and `str.lower()`. CLDR's exemplar
   for Latin languages is lowercase only; case closure brings
   uppercase. Greek `Ά` `Έ` `Ώ`, Vietnamese `Ờ`, Czech `Č` etc.
   come from this.
3. **NFD closure** — for every codepoint, also add the codepoints
   of its `unicodedata.normalize("NFD", ch)`. Apertus has
   `normalizer: null`, so decomposed text (`α + U+0301`) appears
   alongside precomposed (`ά`). NFD closure puts the combining
   marks into the right locale bits.
4. **Script-range fallback** — for scripts where the (script ↔
   covered-locale set) mapping is essentially one-to-one (Han,
   Hangul, Hiragana/Katakana, Greek monotonic + polytonic block,
   Hebrew, Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam,
   Gujarati, Gurmukhi, Thai, Myanmar, Armenian, Georgian), any
   letter/mark codepoint in that script's Unicode range gets the
   bits of the locales using that script even if CLDR's curated
   exemplar doesn't list it. Closes the ~80k Han codepoints CLDR
   doesn't enumerate (`摇 呃 噢 删`). **Cyrillic and Arabic are
   deliberately excluded** from the fallback — their script blocks
   contain language-specific extensions (Kazakh `Қ`, Bashkir `Ҡ`,
   Pashto `ښ`, Sindhi `ڤ`, …) for languages outside our scope, so
   falling back would falsely admit them to in-scope Cyrillic/Arabic
   locales. Those codepoints intentionally fall through to 0 bits.
5. **Post-fallback NFD closure** — after the fallback step adds
   codepoints (notably polytonic Greek precomposed forms and the
   wide Han set), one more bitmask-level NFD pass propagates each
   codepoint's bits to its NFD components. Catches combining marks
   like U+0345 ypogegrammeni that are only reachable through
   precomposed polytonic chars added by the fallback.

## CLDR subsets we use, and why

For each locale, we union the following subsets of CLDR's
`characters.json`:

- `exemplarCharacters` (main) — canonical alphabet.
- `index` — uppercase / index characters (Latin exemplars are
  lowercase-only; index is the only CLDR source for uppercase before
  case closure).
- `numbers` — locale digit set.
- `punctuation` — locale punctuation.

We then **drop everything except `L*`/`M*`-category codepoints** from
the union before applying any closures. The non-letter codepoints
from CLDR's subsets are *not* used as language evidence — they're
handled uniformly by the substrate rule (next section).

We **exclude** `auxiliary`, `numbers-auxiliary`, `punctuation-auxiliary`,
`punctuation-person`. Rationale:

- `auxiliary` is "characters used in foreign words / loans / archaic
  forms". Under encoding-precise framing, including them collapses
  meaningful encoding distinctions — most importantly, CLDR `el`'s
  `auxiliary` contains the polytonic forms, which would make
  `el` and `el-polyton` indistinguishable as bits. Dropping
  `auxiliary` gives the polytonic encoding its own discrimination
  power.
- The other auxiliary-style subsets contain script-foreign or
  position-specific characters that would inflate cross-script bits.

## Substrate = all bits

Punctuation, digits, symbols, whitespace, format/control codepoints,
and **modifier letters (Lm)** — Unicode general categories `N*`,
`P*`, `S*`, `Z*`, `Cc`, `Cf`, `Lm` — get **every bit set (ALL_BITS)**.
Lm is letter-categorised in Unicode but functionally acts as
cross-language punctuation: `ʻ` (Uzbek Latin / Hawaiian / Polynesian
glottal-stop indicator), `ʼ`, `ˇ` etc. Treating them as language
evidence would falsely narrow membership the same way ASCII `=` did.
We additionally maintain an `EXTRA_SUBSTRATE_CODEPOINTS` list for
Ll/Lo codepoints that Unicode categorises as letters but function
as language-neutral typography: U+00AA (ª), U+00B5 (µ MICRO SIGN —
note this is distinct from Greek mu U+03BC), U+00BA (º). Substrate codepoints
contribute zero exclusion power under the rejection framing: every
language's text uses periods, digits, dashes, parens, and so on
(empirically verified by querying CLDR per-locale punctuation across
all 23 v1 locales — `.`, `,`, `;`, `:`, `(`, `)`, `-`, `*`, `…` are
in every locale's CLDR set including CJK).

Without this rule:
- ASCII space, having no CLDR exemplar membership, would AND-reject
  every language for space-prefixed tokens like ` the`.
- ASCII `=` (CLDR-listed in zh-Hans only by quirk) would rule out
  every non-Chinese language for any token containing `=`.
- ASCII `_` (CLDR-listed in zh-Hant + ja only) would similarly
  produce false rejections for code tokens.

The substrate override eliminates that whole class of error in one
move.

## Application semantics

The token-level artifact `token_language_bitmask.parquet` exposes:

- `bitmask_and` — AND across the token's codepoints. Answers "which
  triples does **every** char admit?" — the canonical "possible
  triples" set. Everything not in this set is rejected.
- `bitmask_or` — OR across the same. Answers "which triples does
  **at least one** char support?". Less stringent; useful for
  diagnosing tokens with mixed scripts.

For codepoints not in the bitmask table, the apply script falls back:
- If Unicode category is substrate (`N`/`P`/`S`/`Z`/`Cc`/`Cf`):
  treat as ALL_BITS (same as the build-time substrate rule).
- Else (a letter/mark in a script we don't model): treat as 0 bits.
  AND will collapse to 0, which correctly rejects every in-scope
  triple.

This duplicates the build-time rule at apply time, so the artifact
is robust to codepoints that didn't make it into the stored table
(e.g. exotic emoji, supplementary-plane symbols).

## Token-level status

- `text` — all decoded chars are in scope (either letters from CLDR
  / closures, or substrate). AND/OR are meaningful.
- `text_with_unmodeled_letters` — token decoded fine but contains at
  least one letter in a script outside our 55 triples. The AND will
  be 0 (because that char contributes 0). The OR captures which
  triples *some* in-scope chars admit; useful when these tokens
  contain mixed scripts.
- `partial_utf8` — token's bytes are not a complete UTF-8 sequence
  (BPE inner-token fragment).
- `byte_unmapped` — token contains a char not in the GPT-2
  ByteLevel alphabet (shouldn't happen for Apertus, but handled
  defensively).
- `special` — Apertus special token (`<s>`, padding, etc.).
- `no_in_scope_chars` — token contains nothing recognised as either
  letter-in-scope or substrate (e.g. surrogate chars only). Rare.

## Validation requirement

`scripts/validate.py` enforces strict-rule invariants after every
build. Categories of check:

- **v1 spot checks** still pass: 'a' has all-Latin bits, 'ñ' is
  Spanish-only, 'α' is Greek (mono + poly), Russian 'а' is Russian.
- **Case closure**: `Ά`, `Έ`, `Ώ`, `Ç`, `Ñ`, `Ờ`, `Ł`, Cyrillic
  capitals all inherit bits from their lowercase counterparts.
- **NFD closure**: combining acute U+0301 is in every locale whose
  alphabet contains an acute-accented precomposed letter (Czech,
  Spanish, French, Hungarian, Italian, Dutch, Portuguese, …).
- **Script-range fallback**: `摇`, `删` (Han codepoints outside
  CLDR's curated 3K exemplar) carry the CJK bits.
- **Substrate ALL_BITS**: digits, period, hyphen, space, newline,
  `=`, `_`, `{`, `}`, `$` all have every bit set.
- **New v2 locales**: ko 다, hi क, he ש, th พ, hy ա, ka ა, the
  Indic family, my က — each is in exactly the expected locale.
- **Sister-language attribution**: ø is both Danish and Norwegian
  Bokmål; å is in Danish + Norwegian + Swedish + Finnish; Romanian
  `ș` `ț` are in `ro` (no longer 0-bit); Ukrainian `і` `ї` are in
  `uk` (no longer 0-bit).

An audit pass also rescans the built token table for status counts
and script distribution of fall-through tokens. Strict-rule pass
condition: every script not in our 55 triples appears with <50
tokens in `text_with_unmodeled_letters` + `no_in_scope_chars`.

## Scope (current)

55 (language, script, encoding) triples = 55 bits, stored as uint64.
Icelandic (`is`) was added at bit 54 as a post-build audit follow-up:
without it, ð/þ pushed the Latin-fall-through count above the 50-token
audit threshold.

| script(s) | bits | locales |
|---|---|---|
| Latn | 0–14, 38, 42, 44–54 | en, cs, da, de, es, fr, hu, id, it, nl, pl, pt, sv, tr, vi, ro, sr-Latn, az, fi, nb, sl, hr, sk, et, lt, lv, ca, is |
| Cyrl | 15, 39, 40, 41, 43 | ru, uk, bg, mk, sr-Cyrl |
| Grek (monotonic) | 16 | el |
| Grek (polytonic) | 22 | el-polyton |
| Arab | 17, 18, 37 | ar, fa, ur |
| Hans / Hant / Jpan | 19, 20, 21 | zh-Hans, zh-Hant, ja |
| Hang | 23 | ko |
| Deva | 24 | hi |
| Hebr | 25 | he |
| Thai | 26 | th |
| Armn | 27 | hy |
| Geor | 28 | ka |
| Beng | 29 | bn |
| Taml | 30 | ta |
| Telu | 31 | te |
| Knda | 32 | kn |
| Mlym | 33 | ml |
| Gujr | 34 | gu |
| Guru | 35 | pa |
| Mymr | 36 | my |

Bit assignments are stable wire-format identifiers — never reused.
Adding a triple appends at the next free bit. uint64 leaves 10 bits
free for future extensions.

## Output schema

### `char_language_bitmask.parquet`

One row per codepoint that received at least one bit (either via
CLDR, a closure, or the substrate rule).

| column | type | meaning |
|---|---|---|
| `codepoint` | `uint32` | Unicode scalar value |
| `bitmask` | `uint64` | triple-membership bits (positions per `languages.yaml`) |
| `char` | `string` | the character itself, for inspection |
| `num_langs` | `uint8` | popcount of bitmask |
| `category` | `string` | Unicode general category (Lu, Ll, Nd, …) |

### `token_language_bitmask.parquet`

One row per Apertus vocab token (131,072 rows).

| column | type | meaning |
|---|---|---|
| `token_id` | `uint32` | Apertus token id |
| `token_bytes` | `binary` | decoded raw bytes (post ByteLevel inversion) |
| `decoded_text` | `string` | UTF-8 decode of the bytes, or `null` if invalid |
| `bitmask_and` | `uint64` | AND across codepoint bits |
| `bitmask_or` | `uint64` | OR across codepoint bits |
| `num_chars` | `uint16` | decoded codepoint count |
| `status` | `string` | see § Token-level status |

### `manifest.json`

Build metadata: timestamp, CLDR release, included/excluded subsets,
closures applied, language list with bit assignments, per-language
codepoint counts, and the proxy-assumption note.

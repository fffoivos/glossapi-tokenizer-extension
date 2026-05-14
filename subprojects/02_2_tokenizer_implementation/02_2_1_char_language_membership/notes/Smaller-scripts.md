# Smaller scripts (Thai / Armenian / Georgian / Burmese)

> Four single-locale scripts in scope: `th` (Thai, bit 26),
> `hy` (Armenian, bit 27), `ka` (Georgian, bit 28), `my` (Burmese /
> Myanmar, bit 36). Combined into one notes file because each has
> modest per-script complexity and very similar empirical
> structure (one locale = one script = near-perfect token-level
> attribution). Status: coverage exhaustive for in-text content;
> the small missing-from-table sets are all substrate-category
> (script-specific punctuation + native digits) handled correctly
> by the apply-time substrate fallback.

## Thai (th)

### Sources

- CLDR cldr-misc-full 48.2.0 — `th/characters.json`.
- Unicode 16.0 Thai block (U+0E00–0E7F).
- Royal Institute of Thailand — Thai orthography authority.
- Wikipedia "Thai script".

### Coverage

- CLDR `th` exemplar: 44 consonants + 28 vowel signs + 4 tone marks
  + auxiliary marks (yamakkan, nikhahit, etc.). Total ~72 codepoints
  in CLDR.
- Thai block U+0E00–0E7F: **72 of 128 codepoints in table** (rest
  are unassigned or rarely-used).
- Apertus vocab: **563 tokens** contain Thai codepoints, 63 unique
  codepoints used.
- Token-level attribution: 560 of 563 (99.5 %) AND-narrow to `th`
  alone.

### Missing in vocab

Only 2 codepoints used in vocab not in table:

| codepoint | char | category | name | tokens |
|---|---|---|---|---|
| U+0E46 | `ๆ` | Lm | THAI CHARACTER MAIYAMOK (repetition mark) | 2 |
| U+0E50 | `๐` | Nd | THAI DIGIT ZERO | 1 |

Both substrate by Unicode category, ALL_BITS via apply-time
fallback. Correct.

### Decisions

No changes. Coverage works; the structural conventions (vowels
positioned before / after / above / below consonants, all encoded
as distinct codepoints in a linear sequence) are entirely in scope.

## Armenian (hy)

### Sources

- CLDR cldr-misc-full 48.2.0 — `hy/characters.json`.
- Unicode 16.0 Armenian block (U+0530–058F) + Armenian Presentation
  Forms (U+FB13–FB17, ligatures).
- Wikipedia "Armenian alphabet".
- Mesropian alphabet (5th-century CE original) + two letters added
  in the 20th century (`օ` and `ֆ`).

### Coverage

- CLDR `hy` exemplar: 39 lowercase letters (the 37 original
  Mesropian + 2 added).
- Armenian block U+0530–058F: **79 of 96 codepoints in table**;
  Armenian Presentation Forms U+FB13–FB17: **0 of 5** (rare
  ligatures).
- Apertus vocab: **1,098 tokens** with Armenian codepoints, 72
  unique codepoints used.
- Token-level: 1,089 of 1,098 (99.2 %) AND-narrow to `hy` alone.

### Missing in vocab

| codepoint | char | category | name | tokens |
|---|---|---|---|---|
| U+0589 | `։` | Po | ARMENIAN FULL STOP (verjaket — looks like a colon) | 7 |
| U+055D | `՝` | Po | ARMENIAN COMMA (bowt) | 2 |

Both Po category, substrate via fallback → ALL_BITS at apply time.
Correct.

Notable: Armenian uses its own end-of-sentence and punctuation
codepoints (`։ ՛ ՝ ՜ ՞ ֊`). All are Po category and handled
correctly.

### Decisions

No changes. The 17 missing in-table Armenian-block codepoints are
mostly unassigned or rarely-used historical letters; none appear
in Apertus vocab.

## Georgian (ka)

### Sources

- CLDR cldr-misc-full 48.2.0 — `ka/characters.json`.
- Unicode 16.0: Georgian (U+10A0–10FF, the modern Mkhedruli +
  archaic Asomtavruli), Georgian Supplement (U+2D00–2D2F,
  Nuskhuri — archaic ecclesiastical), Georgian Extended
  (U+1C90–1CBF, **Mtavruli** uppercase Mkhedruli, added in Unicode
  11.0 in 2018).
- Wikipedia "Georgian scripts".

### Coverage

- CLDR `ka` exemplar: 33 letters of the modern Mkhedruli alphabet.
- Georgian block U+10A0–10FF: **86 of 96 codepoints in table**.
- Mtavruli block U+1C90–1CBF: **33 of 48 codepoints in table** —
  thanks to case closure! CLDR `ka` has lowercase Mkhedruli; case
  closure adds the Mtavruli uppercase counterparts via `str.upper()`,
  which Python's Unicode-data tables handle correctly for the
  Mkhedruli → Mtavruli case pair (added to Python ≥ 3.7).
- Georgian Supplement U+2D00–2D2F (archaic Nuskhuri): **0 in
  table**.
- Apertus vocab: **480 tokens** with Georgian codepoints, 32
  unique codepoints used. 480 (100 %) AND-narrow to `ka` only.

### Missing in vocab

None. Every Georgian codepoint appearing in Apertus vocab is in
our table with the `ka` bit.

### Decisions

No changes. Mtavruli case-closure working correctly. Asomtavruli
(U+10A0–10C5) and Nuskhuri (U+2D00–2D25) are archaic ecclesiastical
scripts; not in Apertus vocab; out-of-scope by design.

## Burmese / Myanmar (my)

### Sources

- CLDR cldr-misc-full 48.2.0 — `my/characters.json`.
- Unicode 16.0: Myanmar (U+1000–109F), Myanmar Extended-A
  (U+AA60–AA7F, additional letters for Burmese-script minority
  languages), Myanmar Extended-B (U+A9E0–A9FF, more).
- Wikipedia "Burmese script", "Myanmar (Unicode block)".
- **Zawgyi vs Unicode encoding caveat** — pre-2019 Burmese web
  text often used a non-Unicode Zawgyi-1 encoding (Myanmar code
  points but with different semantics for combining marks).
  Modern Apertus training data should be Unicode-compliant
  Myanmar text.

### Coverage

- CLDR `my` exemplar: 33 consonants + medials/finals + vowel
  signs + tone marks (asat, virama). Largest exemplar of the four
  smaller scripts (~50–60 codepoints).
- Myanmar block U+1000–109F: **132 of 160 codepoints in table**.
- Myanmar Ext-A / Ext-B: **0 in table** (used by Burmese-script
  minority languages — Shan, Mon, Karen, etc., not in scope).
- Apertus vocab: **127 tokens** with Burmese codepoints, 56 unique
  codepoints used. 110 of 127 AND-narrow to `my` alone (~87 %; the
  remaining 17 have additional bits from substrate-fallback chars).

### Missing in vocab

| codepoint | char | category | name | tokens |
|---|---|---|---|---|
| U+104B | `။` | Po | MYANMAR SIGN SECTION (sentence terminator) | 3 |
| U+104F | `၏` | Po | MYANMAR SYMBOL GENITIVE | 2 |
| U+104A | `၊` | Po | MYANMAR SIGN LITTLE SECTION (comma) | 1 |
| U+1041 | `၁` | Nd | MYANMAR DIGIT ONE | 1 |
| U+1040 | `၀` | Nd | MYANMAR DIGIT ZERO | 1 |

All 15 missing codepoints are substrate-category (Po punctuation,
Nd digits) and handled correctly by the apply-time fallback.

### Zawgyi check

Audit observation: no obvious Zawgyi-style misencoding in the
Burmese tokens. The codepoints we see (က ခ ဂ ဃ ... န ပ etc.)
match standard Unicode Burmese. If Apertus's training data did
include any Zawgyi-encoded text, it would have used the same
Myanmar block codepoints but with different combining-mark
semantics — the codepoint-level audit can't tell the difference,
only rendering / linguistic inspection can. **Out of scope to
investigate further unless audit-driven need surfaces.**

### Decisions

No changes. Burmese coverage is adequate for the in-vocab content;
Burmese-script minority-language coverage is out of scope by
design.

## Cross-script decisions

1. **No `languages.yaml` changes for any of th / hy / ka / my.**
2. **No `EXTRA_SUBSTRATE_CODEPOINTS` additions.** Script-specific
   punctuation (`. ։ ՝ ။ ၏ ၊ ๆ`) handled by Unicode-category
   substrate fallback. Consistent with the rest of the project.
3. **Native-digit policy**: same as for Indic — Thai, Burmese
   native digits are `Nd` and fallback-handled. No per-script
   seeding.
4. **Mtavruli case-closure** working correctly. Reaffirms that
   the case-closure pipeline in `_common.py` handles Python ≥ 3.7
   Unicode data correctly for non-Latin case pairs.

## Followups

- **Burmese-script minority languages** (Shan, Mon, Karen,
  Tai Laing): not modeled. Their additional letters in Myanmar
  Ext-A/B would 0-bit if they appear; not in Apertus vocab today.
- **Lao (lo)**: Brahmic-descended; uses Lao block U+0E80–0EFF.
  Not in scope. Audit didn't flag any Lao tokens.
- **Khmer (km)**: U+1780–17FF. Not in scope; no Apertus tokens
  detected.
- **Tibetan (bo)**: U+0F00–0FFF. Not in scope.
- **Sinhala (si)**: U+0D80–0DFF. Not in scope.
- **Ethiopic (am)**: U+1200–137F. Not in scope.
- **Cherokee, NKo, Vai, Bamum, etc.** — not in scope; 0-bit
  strict-rejection is correct for any tokens containing them.

# Indic family — per-script research notes

> Eight in-scope locales sharing Brahmic-derived structure: hi
> (Devanagari, bit 24), bn (Bengali, bit 29), ta (Tamil, bit 30),
> te (Telugu, bit 31), kn (Kannada, bit 32), ml (Malayalam, bit 33),
> gu (Gujarati, bit 34), pa (Gurmukhi, bit 35). Each has its own
> single-locale family bit at the family layer. Status: coverage
> exhaustive; near-perfect token-level discrimination per script
> (each script is one-locale-per-script in our scope).

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — per-locale `characters.json` for
  the 8 in-scope Indic locales.
- Unicode 16.0 South Asian Scripts: Devanagari (U+0900–097F),
  Bengali (U+0980–09FF), Tamil (U+0B80–0BFF), Telugu
  (U+0C00–0C7F), Kannada (U+0C80–0CFF), Malayalam (U+0D00–0D7F),
  Gujarati (U+0A80–0AFF), Gurmukhi (U+0A00–0A7F).
- Unicode Standard chapter 12 "South Asian Scripts" — definitive
  reference for Indic orthographic structure (independent vowels,
  consonants, dependent vowel signs, virama, ZWJ/ZWNJ semantics).
- W3C Indic Layout Requirements documents — one per script,
  describing per-script rendering rules in implementation detail.
- Wikipedia: "Devanagari", "Bengali script", "Tamil script",
  "Telugu script", "Kannada script", "Malayalam script", "Gujarati
  script", "Gurmukhi script".

## Empirical Apertus baseline — script-level

Token-level discrimination (each script is one-locale-per-script
in our scope, so the AND narrowing is essentially perfect):

| locale | script | block | cps in table | cps in vocab | tokens | tokens narrowing to this locale only |
|---|---|---|---|---|---|---|
| hi | Devanagari | U+0900–097F | 114 | 79 | 1,555 | 1,529 (98.3 %) |
| bn | Bengali | U+0980–09FF | 75 | 63 | 820 | 810 (98.8 %) |
| ta | Tamil | U+0B80–0BFF | 51 | 43 | 525 | 525 (100 %) |
| te | Telugu | U+0C00–0C7F | 81 | 52 | 900 | 900 (100 %) |
| kn | Kannada | U+0C80–0CFF | 80 | 62 | 558 | 548 (98.2 %) |
| ml | Malayalam | U+0D00–0D7F | 90 | 60 | 377 | 377 (100 %) |
| gu | Gujarati | U+0A80–0AFF | 79 | 48 | 181 | 180 (99.4 %) |
| pa | Gurmukhi | U+0A00–0A7F | 69 | 43 | 148 | 148 (100 %) |

**Total: ~5,064 Indic-script tokens, with ≈99 % single-locale
attribution.** The Indic scripts deliver the strongest single-
locale discrimination in our entire scope, precisely because each
script is overwhelmingly used by one language.

## Coverage of in-vocab Indic codepoints

Each script's CLDR exemplar covers the standard alphabet plus
common diacritic / vowel-sign combinations. After case closure (a
no-op for Indic — these scripts don't have case) and NFD closure,
every Indic codepoint actually appearing in Apertus vocab tokens
is either in the table or handled by the substrate fallback.

Per-script missing codepoints in vocab:

| script | unique missing | examples |
|---|---|---|
| Devanagari | 13 | U+0964 `।` DANDA (14 tokens), U+0966–096F native digits ०–९ |
| Bengali | 10 | U+09E6–09EF native digits ০–৯ |
| Tamil | 0 | nothing missing |
| Telugu | 0 | nothing missing |
| Kannada | 10 | U+0CE6–0CEF native digits ೦–೯ |
| Malayalam | 0 | nothing missing |
| Gujarati | 1 | U+0AE7 native digit ૧ |
| Gurmukhi | 0 | nothing missing |

**Every missing codepoint is either native digits (Unicode category
`Nd`) or script-specific punctuation like the Devanagari danda
(`Po`).** Both are substrate-category and correctly handled by the
apply-time fallback → ALL_BITS. Consumers using `query_codepoint.py`
get the right answer. Direct-parquet readers without the fallback
would false-reject (the documented sparse-with-fallback contract).

## ZWJ / ZWNJ (zero-width joiner / non-joiner)

Critical for Indic conjunct-consonant rendering: ZWNJ U+200C
disables conjunct ligature formation; ZWJ U+200D forces it.

- **ZWNJ (U+200C)**: in **79 Apertus tokens**. NOT in our table.
  Category `Cf` (format) → substrate fallback handles it as
  ALL_BITS. Correct.
- **ZWJ (U+200D)**: in 1 Apertus token. NOT in our table.
  Category `Cf` → ALL_BITS via fallback. Correct.

Both are substrate-category and apply-time-correct. No action.

## Native digit policy

Indic digits (U+0966–096F Devanagari, U+09E6–09EF Bengali, etc.)
are NOT in any of the in-scope CLDR locales' `numbers` set — CLDR
`hi` and `bn` etc. ship Latin digits 0–9 as their default
numbering system. So:

- Native Indic digits aren't in our table.
- They appear in a handful of Apertus tokens (10 tokens for
  Devanagari, 10 for Bengali, 10 for Kannada, 1 for Gujarati).
- Category `Nd` → substrate fallback handles them as ALL_BITS.
- **Empirically harmless.**

If we ever wanted strict per-script native-digit attribution we'd
add them to a per-locale supplementary set in `languages.yaml`,
but no current consumer needs that.

## Orthographic structure (shared across the family)

All 8 Indic scripts share Brahmic structural conventions:

- **Independent vowels** (e.g. Devanagari `अ आ इ ई …`).
- **Consonants** (e.g. `क ख ग घ …`).
- **Dependent vowel signs** that attach to consonants (e.g.
  Devanagari `ा ि ी ु …`). Unicode category `Mc` (spacing mark)
  or `Mn` (non-spacing mark) — these are treated as letter-or-mark
  by `is_letter_or_mark` (`M*` categories pass).
- **Virama** (e.g. `्` U+094D for Devanagari) — suppresses the
  inherent vowel; combines with following consonant via ZWJ/ZWNJ
  rules for conjunct ligatures.
- **Special marks**: anusvara, candrabindu, visarga (e.g. Devanagari
  `ं ँ ः`).

All of these are in CLDR `hi` / `bn` / `ta` / etc. main + index
where applicable. Case closure is a no-op (Indic scripts are
caseless). NFD closure is a no-op for standalone marks but
relevant for precomposed-syllable forms in Bengali / Devanagari
(rare; the NFD closure picks them up).

## Decisions

1. **No `languages.yaml` changes for any Indic locale.** Coverage
   is exhaustive for in-text content.
2. **Native digit handling via substrate fallback is correct.** No
   need to seed Indic digits explicitly.
3. **ZWJ / ZWNJ handling via substrate fallback is correct.** They
   carry no language signal individually; an Indic token with a
   conjunct contains the base consonants + virama + ZWJ in
   sequence, and the base consonants carry the locale bit.
4. **Single-locale single-family alignment**: every Indic script
   has exactly one in-scope locale, so the `family_bits` at this
   layer is a deterministic echo of `bitmask` (the language bit)
   for these scripts. This is the documented "single-locale
   families for symmetry" decision from PLAN_v3.

## Followups

- **Marathi (mr)** — uses Devanagari with additional chars beyond
  CLDR `hi` (e.g. `ळ` `ऍ` `ऑ`). Apertus's FineWeb-2 mix has
  Marathi content. Not in scope. If audit ever surfaces high
  fall-through count for Marathi-specific chars, add `mr` at the
  next free bit.
- **Nepali (ne)** — Devanagari; similar story to Marathi. Most
  chars overlap with hi. Out of scope.
- **Sanskrit (sa)** — Devanagari + additional Vedic-text marks
  (U+1CD0–1CFF). Very specialized; out of scope.
- **Assamese (as)** — uses Bengali script + an extra char `ৰ`. Out
  of scope; would extend Bengali family.
- **Sinhala (si)**, **Lao (lo)**, **Khmer (km)**, **Tibetan (bo)**
  — Brahmic-descended scripts not in scope. Their codepoints fall
  through correctly (strict-rejection) if they appear in vocab.
- **Marathi candrabindu and other Vedic marks** in
  U+0951–0954 — handled by the script-range fallback over the
  Devanagari block (these positions are in U+0900–097F).

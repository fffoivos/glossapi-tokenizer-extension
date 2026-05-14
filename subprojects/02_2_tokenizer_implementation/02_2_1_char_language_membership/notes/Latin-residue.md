# Latin-residue (Indonesian + Vietnamese) — per-locale research notes

> Two non-European Latin locales in scope: `id` (Indonesian,
> bit 7, Austronesian-Latn family) and `vi` (Vietnamese, bit 14,
> Vietic-Latn family). Both ride the European-Latin v3 hierarchy
> infrastructure; this file documents what each adds (or doesn't)
> beyond it.

## Indonesian (id) — uses bare Latin

### Sources

- CLDR cldr-misc-full 48.2.0 — `id/characters.json`.
- Wikipedia "Indonesian alphabet".
- Indonesian uses the 26-letter modern Latin alphabet **without
  diacritics** (post-1972 spelling reform). Loanwords retain
  diacritics from source language but those are auxiliary, not
  core.

### Coverage

- CLDR `id` main: `a b c d e f g h i j k l m n o p q r s t u v w x
  y z` — bare ASCII Latin, no diacritics.
- CLDR `id` auxiliary: foreign-loan accented letters
  (`áàăâåäãā æ ç éèĕêëē ñ óòŏôöøō úùŭûüū ÿ`) — dropped under our
  project-wide auxiliary-exclusion rule.

### Empirical Apertus baseline

- **0 tokens** have `id` as the *only* Latin bit in their AND.
  This is structurally inevitable: `id`'s main set is a strict
  subset of every other Latin locale's main set (every Latin
  language admits a-z). So any token containing only a-z chars
  has all 28 Latin locale bits set in its AND, including `id`.
  Any token containing a diacritic outside CLDR `id` main would
  drop `id` from the AND.

### Decisions

1. **`id` is empirically un-discriminating at the token level.** This
   isn't a bug — it reflects that Indonesian shares 100 % of its
   core character set with every other Latin-script language.
   Discrimination would require auxiliary chars (which we
   uniformly drop) or n-gram / morphology signal (out of scope).
2. **No `languages.yaml` change**. Coverage is correct and
   complete.
3. **Consumers wanting to detect Indonesian text** at the
   token level: cannot do it from the bitmask alone. Use external
   language-detection (e.g., langdetect, FastText langid) or
   frequency statistics.

### Followups

- **Loanword discrimination**: `id` auxiliary contains foreign-
  loan accented forms. If a future consumer needs Indonesian-vs-
  Other-Latin attribution, the option exists to selectively re-
  admit `id` auxiliary. But this is a per-locale carve-out from
  the project-wide rule; defer until a concrete consumer needs
  it.
- **Mixed-script Indonesian** (Indonesian text borrowing Chinese
  / Arabic / Sanskrit terms) — handled correctly by the existing
  multi-script aggregation; the `id` bit stays set when only
  Latin chars are involved.

## Vietnamese (vi) — heavy tone-mark coverage

### Sources

- CLDR cldr-misc-full 48.2.0 — `vi/characters.json`.
- Wikipedia "Vietnamese alphabet" — the Quốc ngữ system.
- Unicode 16.0 Latin Extended Additional (U+1E00–1EFF) — most of
  Vietnamese's precomposed tone-marked vowels live here.

### Coverage

- CLDR `vi` main is the largest of any Latin locale in our scope
  by main-alphabet codepoint count: 89 codepoints spread across
  vowel + tone-mark combinations. Includes all 12 vowel-quality
  variants (`a ă â e ê i o ô ơ u ư y`) crossed with 6 tone marks
  (no tone, grave, hook-above, tilde, acute, dot-below).
- CLDR `vi` auxiliary is tiny: `f j w z` — letters not used in
  native Vietnamese but appearing in loanwords. Dropped.
- Vietnamese-specific block U+1EA0–1EF9: **90/90 in table** —
  full coverage of all precomposed Vietnamese vowel + tone-mark
  combinations.

### Empirical Apertus baseline

- **797 tokens** have `vi` as the *only* European-Latin locale
  bit in their AND. This is the highest single-locale narrowing
  count of any in-scope Latin locale (vs pl 371, cs 201, az 160,
  ro 134, pt 121, …).
- The Vietic-Latn family bit narrows similarly cleanly: **786
  tokens** AND to `{Vietic-Latn}` only at the family layer.
- The reason: Vietnamese's tone-marked vowel forms (`ờ`, `ấ`,
  `ặ`, `ụ`, `ự`, …) are largely unique to Vietnamese — they're in
  Latin Extended Additional and only Vietnamese in our scope uses
  them in normal text.

### Decisions

1. **No `languages.yaml` changes**. CLDR `vi` coverage is
   complete; case closure handled the precomposed-tone uppercase
   variants; NFD closure handled the combining-mark decompositions.
2. **Vietnamese is one of the strongest token-level
   discriminators in our scope** — better than any single
   European Latin locale. The tone marks do the work.
3. **NFD-decomposed Vietnamese**: Apertus has `normalizer: null`,
   so both precomposed (`ờ` U+1EDD) and decomposed (`o` + U+031B
   + U+0300) forms can appear. The post-fallback NFD closure
   (v3.0 implementation) propagates `vi` bits to the constituent
   combining marks. Verified at build time.

### Followups

- **Vietnamese-romaji collision**: there's a small overlap
  between Vietnamese tone-marked vowels and other languages' rare
  forms. Empirically irrelevant in Apertus's vocab — none of the
  797 vi-only-AND tokens has a competing-language bit.
- **Vietnamese punctuation**: standard Latin punctuation per
  CLDR; substrate-handled. No action.

## Combined decisions for both locales

1. **id is structurally un-discriminating** at the token level
   (its main set is a subset of every Latin locale's). Documented
   as expected behaviour, not a bug.
2. **vi is among the strongest single-locale discriminators** in
   our scope, thanks to its rich tone-mark inventory in Latin
   Extended Additional.
3. **Neither requires changes** to languages.yaml, families.yaml,
   scripts.yaml, or the build pipeline.

This completes the per-script research pass over all 22 scripts /
31 families / 55 locales in `languages.yaml` v4.

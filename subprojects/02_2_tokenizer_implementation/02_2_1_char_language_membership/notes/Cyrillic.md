# Cyrillic — per-script research notes

> Five in-scope locales (`ru` bit 15, `uk` bit 39, `bg` bit 40,
> `mk` bit 41, `sr-Cyrl` bit 43). Includes the Russian deep-dive
> from the special-set scope. Status: in-scope coverage exhaustive;
> 132-token coverage gap is real (out-of-scope Cyrillic-using
> languages — Kazakh, Belarusian, Mongolian, Tatar, Bashkir), and
> the strict-rejection behaviour for those is correct.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `ru.json`, `uk.json`, `bg.json`,
  `mk.json`, `sr-Cyrl.json`.
- Unicode 16.0: Cyrillic (U+0400–04FF), Cyrillic Supplement
  (U+0500–052F), Cyrillic Extended-A (U+2DE0–2DFF), Cyrillic
  Extended-B (U+A640–A69F), Cyrillic Extended-C (U+1C80–1C8F).
- Wikipedia: "Cyrillic alphabets" — comparative table across
  Russian, Ukrainian, Belarusian, Bulgarian, Macedonian, Serbian,
  Kazakh, Bashkir, Mongolian Cyrillic, Tatar, etc.
- ISO 8859-5, KOI8-R, CP1251 — legacy Cyrillic encodings.
- Russian Academy of Sciences (Институт русского языка) — Russian
  orthography authority. Standard alphabet is **33 letters** post-
  1918 reform.

## Per-locale CLDR exemplar (verified)

| locale | main (lowercase, 30–33 letters) |
|---|---|
| ru | `а б в г д ёе ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я` (33 incl. ё) |
| uk | adds `є ґ і ї`, lacks `ё ы э ъ` |
| bg | `а б в г д е ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ь ю я` (30; no ё ы э й-variant) |
| mk | adds Macedonian-specific `ѓ ѕ ј љ њ ќ џ` |
| sr-Cyrl | adds Serbian-specific `ђ ј љ њ ћ џ` |

All five in-scope locales' base alphabets are present in our table
with the correct per-locale bit, plus case closure for uppercase.

## Empirical Apertus baseline

- 7,685 Apertus vocab tokens contain Cyrillic codepoints.
- 94 unique Cyrillic codepoints used in the vocab.
- All in-scope Cyrillic codepoints (matching CLDR for our 5
  locales) are in our table.

Per-token AND result:

- **487 tokens** with `bitmask_and` = ru only (no uk / bg / mk /
  sr-Cyrl bits). These contain Russian-distinctive characters like
  `ё ы э ъ` that Ukrainian/Bulgarian/etc. don't admit.
- **6,707 tokens** with `bitmask_and` containing multiple Cyrillic
  locales — typically chars in the East-Slavic-Cyrl ∩ South-Slavic-
  Cyrl overlap (the bulk of basic Cyrillic letters).

The cross-locale split works as designed: where chars are shared,
the token AND keeps multiple Cyrl bits; where chars are locale-
specific, the AND narrows.

## The 132-token "coverage gap" — out-of-scope Cyrillic languages

The phase-2 validate audit flags 132 Apertus tokens with Cyrillic
chars not in any of our 5 locales' bits — reported as `[in-scope,
coverage-gap]` Cyrl (in-scope script, locale-gap within).

Audit confirms these are characters from **languages we deliberately
don't model**:

| codepoint | char | locale(s) | token count |
|---|---|---|---|
| U+045E | `ў` | Belarusian | 23 |
| U+04D9 | `ә` | Kazakh, Tatar, Bashkir | 22 |
| U+04AF | `ү` | Kazakh, Kyrgyz, Mongolian, Tatar | 22 |
| U+049B | `қ` | Kazakh, Tajik | 17 |
| U+04E9 | `ө` | Kazakh, Kyrgyz, Mongolian, Tatar | 16 |
| U+04A3 | `ң` | Kazakh, Kyrgyz, Tatar, Bashkir | 11 |
| U+0493 | `ғ` | Kazakh, Tajik | 7 |
| U+04B1 | `ұ` | Kazakh | 6 |
| U+049A | `Қ` | Kazakh (capital) | 4 |
| U+04BB | `һ` | Kazakh, Bashkir, Tatar | 3 |
| U+04C0 | `Ӏ` | Caucasian Cyrillic (palochka) | 2 |
| U+0497 | `җ` | Tatar, Bashkir | 2 |
| U+045D | `ѝ` | Bulgarian-archaic | 1 |
| U+04D8 | `Ә` | Kazakh (capital) | 1 |
| U+04E8 | `Ө` | Kazakh (capital) | 1 |

Distribution by language:

- **Kazakh (kk)**: 109 tokens — the dominant out-of-scope
  Cyrillic language.
- **Belarusian (be)**: 23 tokens.
- **Mongolian Cyrillic (mn)**: 39 tokens (overlaps with Kazakh on
  `ү ө`).
- Smaller: Tatar, Bashkir, Bulgarian-archaic `ѝ`, Caucasian
  palochka.

Under strict-rejection semantics this is **correct behaviour**:
these codepoints have 0 bits at every level, so tokens containing
them AND-reject every in-scope (script, family, language). The
"coverage gap" is real — we don't model these languages — and
strict-rejection handles it correctly.

## Adding Kazakh / Belarusian / Mongolian Cyrillic — when worth it?

PLAN_v3 `§ Open items` flagged these as candidate v3.1 bits.
Decision criteria:

- **Vocab token count**: 109 (kk) + 39 (mn) + 23 (be) = ~170
  tokens total. Each is well under 100 — adding any one of them
  brings us nowhere near the 50-token audit-fail threshold (which
  is for out-of-scope **scripts**, not locales).
- **CLDR coverage**: all three have CLDR exemplar data
  (`kk.json`, `be.json`, `mn.json` in cldr-misc-full).
- **Apertus training data**: FineWeb-2 includes Kazakh, Mongolian,
  Belarusian. Apertus saw them.

Recommendation: **add Kazakh first** (109 tokens, dominant). The
others are smaller and can wait.

If we add `kk` at bit 55 it'd pull Kazakh-specific chars (ә ғ қ ң ө ұ
ү һ) into the East-Slavic-Cyrl family? Or a new Turkic-Cyrl family?
Mongolian and Belarusian aren't Turkic, so:

- `kk` Kazakh: would join a new `Turkic-Cyrl` family (alongside
  Tatar, Bashkir if ever added).
- `mn` Mongolian: a new `Mongolic-Cyrl` family.
- `be` Belarusian: extends `East-Slavic-Cyrl` (joining ru, uk).

The families.yaml change would be small (2 new family bits + 3
locale bits). Deferred — flagging in TODO for if/when we want to
land it.

## Decisions

1. **No changes to in-scope locales' coverage.** The 5 in-scope
   Cyrillic locales' CLDR exemplars are exhaustive and verified.
2. **132-token coverage gap is correct strict-rejection** for
   languages we don't model. Documented here for transparency.
3. **Recommended v3.1 additions** (deferred until user calls them):
   `kk` (Kazakh, ~109 tokens), `be` (Belarusian, ~23 tokens), `mn`
   (Mongolian, ~39 tokens). Each would take a new bit at the
   language layer + family.yaml updates. Roughly 1 hour of work
   total.

## Followups

- **Add kk / be / mn** when user prioritizes. Cyrillic is the
  largest in-scope-script coverage gap; the audit flags it
  prominently.
- **Cyrillic Supplement / Extended-A / B / C blocks** (0 in
  table): mostly historical (Old Church Slavonic, Caucasian
  scripts). Out of scope unless an audit-driven need surfaces.
- **Re-examine Russian auxiliary set**: CLDR `ru` auxiliary lists
  stressed-vowel clusters `{а́} {е́} {и́} {о́} {у́} {ы́} {э́} {ю́} {я́}`.
  Our build drops auxiliary; these clusters contribute the
  combining acute U+0301 via the post-fallback NFD closure (which
  picks it up via NFD of any precomposed acute-accented char). So
  U+0301 ends up with broad Cyrillic + Romance bits, matching the
  empirical use of stress marks in Russian dictionaries / pedagogy.
  Verified working.

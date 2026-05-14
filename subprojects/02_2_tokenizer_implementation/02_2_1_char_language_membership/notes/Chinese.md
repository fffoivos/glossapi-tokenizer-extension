# Chinese — per-script research notes

> Two in-scope locales: `zh-Hans` (bit 19, Simplified) and `zh-Hant`
> (bit 20, Traditional). Plus the shared `Sinitic-Hans` / `Sinitic-
> Hant` family bits and the `Hans` / `Hant` script bits. Status:
> coverage complete; the Hans/Hant token-level split is empirically
> uniform, which is a documented consequence of the permissive
> script-range fallback.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `zh-Hans/characters.json`,
  `zh-Hant/characters.json`.
- Unicode 16.0: CJK Unified Ideographs (U+4E00–9FFF), Extensions A
  through G (U+3400–4DBF, U+20000–2A6DF, U+2A700–2B73F, U+2B740–2B81F,
  U+2B820–2CEAF, U+2CEB0–2EBEF), CJK Compatibility Ideographs
  (U+F900–FAFF, U+2F800–2FA1F).
- Unicode UAX #38 — Unicode Han Database. Per-codepoint kSimplifiedVariant
  / kTraditionalVariant / kBigFive / kGB / kJis source flags.
- GB18030 (PRC) and Big5 (Taiwan/HK) — legacy encoding standards.
- Wikipedia: "CJK Unified Ideographs", "Simplified Chinese characters",
  "Traditional Chinese characters".

## Empirical Apertus baseline

- 3,740 Apertus vocab tokens contain Han codepoints.
- 1,613 unique Han codepoints used.
- All Han codepoints appearing in Apertus vocab are in our table
  (zero missing).

Top tokens are dominantly **modern Simplified Chinese**: 的, 是, 不,
这 (simplified-only), 一, 年, 说, 日, 在, 人. These appear with
high frequency consistent with web Chinese.

Per-token AND result:

- **0 tokens** with `bitmask_and` containing `zh-Hans` only (no
  zh-Hant).
- **0 tokens** with `bitmask_and` containing `zh-Hant` only (no
  zh-Hans).
- **3,482 tokens** with both zh-Hans **and** zh-Hant in their AND.
- **3,739 tokens** (nearly all CJK-containing) with `ja` also in
  the AND.

The Hans/Hant distinction is **char-level present** (CLDR `zh-Hans`
exemplar has 国 U+56FD but not 國 U+570B; CLDR `zh-Hant` has 國 but
not 国) but **token-level uniform** — every Apertus Han token AND-
attributes to both encodings.

## Why Hans/Hant is token-level uniform (it's the script-range fallback)

`apply_script_fallback` in `build_char_language_bitmask.py` gives
every codepoint in the CJK Unified / Extension blocks the union
of `Hans + Hant + Jpan` bits at the language layer. So `国` (in
CLDR `zh-Hans` only) ends up with the Hant bit too via fallback;
`國` (in CLDR `zh-Hant` only) ends up with Hans too. Net effect:
every Han codepoint carries the three CJK language bits, regardless
of CLDR's Hans/Hant exemplar attribution.

This is the **permissive-rejection** position. Two arguments for
it (current behaviour) and two against (alternative):

**For permissive (current)**:

1. **Strict rejection would over-reject.** Modern Chinese readers
   in Hong Kong / Taiwan can read simplified chars without
   difficulty; mainland readers can read traditional. A token like
   `国` in a zh-Hant context (Hong Kong news, Taiwan academic
   citing simplified sources) is admissible. Strict CLDR-only
   attribution would AND-reject zh-Hant for any token containing
   `国`, even though it's plausibly present in zh-Hant text.
2. **CJK ⊆ shared Han** is the standard tokenizer assumption. The
   Unicode Han block was unified specifically because Chinese,
   Japanese, and Korean share most ideographs; we follow that
   assumption.

**Against permissive (i.e., for stricter Hans/Hant split)**:

1. **Vocab analysis use case**: a downstream consumer wants to
   ask "is this token Simplified-only?" — under current behaviour
   they can't. Under strict-CLDR they could.
2. **Symmetry with the Greek monotonic / polytonic split**, where
   we deliberately preserve the encoding distinction (we don't
   apply a "broad Greek fallback" that merges modern + polytonic).

We chose permissive in v2.2 (the fallback was the fix for "20 Han
codepoints not in CLDR fell through to no_charset_chars"). Under
v3 we kept it. The decision is consistent but the **token-level
uniformity is a real loss of discrimination** for the Hans vs Hant
question.

**Open question to resolve before any consumer relies on the
Hans/Hant split at the token level**: do we want to (a) keep
permissive (no change; consumers should not expect Hans-only or
Hant-only tokens to exist), or (b) restrict the Han script-range
fallback to give only `ja` and not Hans/Hant (so CLDR's per-locale
Hans/Hant attribution survives, with non-CLDR Han codepoints
attributed only to ja)? This is a real design choice. Deferred to
the user.

## Coverage state

| block | size | in table | coverage |
|---|---|---|---|
| CJK Unified Ideographs (U+4E00–9FFF) | 20,992 | 20,992 | 100 % |
| CJK Extension A (U+3400–4DBF) | 6,592 | 6,592 | 100 % |
| CJK Extension B (U+20000–2A6DF) | 42,720 | 42,720 | 100 % |
| CJK Extension C (U+2A700–2B73F) | 4,160 | 4,154 | 99.86 % |
| CJK Extension D (U+2B740–2B81F) | 224 | 222 | 99.11 % |
| CJK Extension E (U+2B820–2CEAF) | 5,776 | 5,762 | 99.76 % |
| CJK Compatibility (U+F900–FAFF) | 512 | 472 | 92.19 % |
| CJK Compatibility Suppl. (U+2F800–2FA1F) | 544 | 542 | 99.63 % |

The shortfall in CJK Compatibility (40 codepoints missing out of
512) is mostly unassigned codepoints in that block.

**No Apertus Han token contains a codepoint missing from our table.**
The script-range fallback fully covers what Apertus's tokenizer
saw, plus much more.

## Compatibility ideographs (U+F900–FAFF, U+2F800–2FA1F)

CJK Compatibility Ideographs are NFKD-decomposable to a CJK Unified
codepoint. For our purposes they're substrate-via-fallback at the
script level (the v2.2 fullwidth-substrate work also added these
ranges to the Han fallback). Verified: U+F900 (the first
compatibility ideograph) is in table with all 3 CJK bits.

NFKC-aware script detection in `char_script` handles them: a
compatibility ideograph normalizes to its base CJK Unified form,
which is named "CJK …" and matches the Hans/Hant/Jpan scripts.

## Decisions

1. **No `languages.yaml` or scripts.yaml changes.** Coverage is
   exhaustive.
2. **Keep the permissive Han script-range fallback for now.**
   Token-level Hans/Hant uniformity is the documented consequence;
   consumers wanting the split should look at codepoint-level
   `script_bits` / `family_bits` directly (where the bits **are**
   distinguishable for the CLDR-listed Hans-only / Hant-only
   codepoints).
3. **Document the permissive choice in PLAN_v3** as a known
   property of the Han handling, parallel to the polytonic
   monotonic/polytonic note.

## Followups

- **Hans/Hant token-level split**: open design question, see
  "Why Hans/Hant is token-level uniform" above. Deferred.
- **CJK Extensions F–G** (U+2CEB0–3134F): not yet in
  `SCRIPT_FALLBACK_RANGES`. The Apertus vocab doesn't contain any
  codepoints there, so impact is zero today. Add the ranges if
  ever audit-triggered.
- **Variant tags** (U+E0100–E01EF) — Unicode variation selectors
  for Han chars. Out of scope.
- **Japanese (ja) overlap**: nearly every Han token also gets the
  ja bit. This matches reality (Japanese kanji are largely Han);
  consumers wanting "Japanese-only" attribution rely on the
  presence of Hiragana / Katakana chars in the token (which DON'T
  give Hans/Hant bits — only Jpan via Kana → Jpan in script-range
  fallback).

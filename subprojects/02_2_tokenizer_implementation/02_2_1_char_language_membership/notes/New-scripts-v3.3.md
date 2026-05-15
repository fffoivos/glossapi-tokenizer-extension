# v3.3 new scripts (Ethi / Khmr / Sinh / Laoo / Tibt / Orya / Thaa)

> Seven single-locale scripts added to fulfil the PMI-consumer
> feedback (2026-05-15): Amharic (am, bit 78, Ethiopic),
> Khmer (km, bit 79), Sinhala (si, bit 80), Lao (lo, bit 81),
> Tibetan (bo, bit 82), Odia (or, bit 83), Dhivehi (dv, bit 84).
> Each gets its own script bit (22–28) and single-locale family bit
> (38–44).
>
> Status: added for **consumer canonical-key resolution**, not for
> Apertus vocab discrimination — none of these scripts appear in the
> inherited Mistral-Nemo tokenizer's vocab.

## Empirical Apertus baseline

| script | code | locale | bit | Apertus vocab tokens | unique cps in vocab |
|---|---|---|---|---|---|
| Ethiopic | Ethi | am | 78 | 0 | 0 |
| Khmer | Khmr | km | 79 | 0 | 0 |
| Sinhala | Sinh | si | 80 | 0 | 0 |
| Lao | Laoo | lo | 81 | 0 | 0 |
| Tibetan | Tibt | bo | 82 | 0 | 0 |
| Odia | Orya | or | 83 | 0 | 0 |
| Thaana | Thaa | dv | 84 | 0 | 0 |

**Every one of these scripts is absent from Apertus's vocab.**
Mistral-Nemo's tokenizer had no merges in any of them; any text in
these scripts gets byte-level fragmented when Apertus tokenises.

This means **adding these locales does not change the Apertus
token-level analysis at all** (status counts unchanged: 128,612 text /
21 unmodeled-letters / 4 no-in-scope-chars after v3.2; identical
after v3.3).

## Why ship them anyway

The consumer at `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`
runs PMI promotion against per-language samples from FineWeb-2. The
samples in these scripts (~14 B tokens collectively for the Tier-1
new-script keys) **exist as language samples**, but the consumer's
char-admissibility filter falls back to Variant B (rate-test only)
because the canonical keys `amh_Ethi`, `khm_Khmr`, `sin_Sinh`,
`lao_Laoo`, `bod_Tibt`, `ory_Orya`, `div_Thaa` had no char-tool
mapping.

After v3.3:

| consumer canonical key | char-tool code |
|---|---|
| `amh_Ethi` | `am` |
| `khm_Khmr` | `km` |
| `sin_Sinh` | `si` |
| `lao_Laoo` | `lo` |
| `bod_Tibt` | `bo` |
| `ory_Orya` | `or` |
| `div_Thaa` | `dv` |

The consumer can now apply the char-admissibility filter (Variant A
+ B) to these samples by reading `canonical_key_to_char_tool_code`
from `manifest.json`.

## Sources consulted (one per script)

- **Amharic / Ethiopic (am)** — CLDR `am/characters.json` covering
  the modern Amharic syllabary (consonant × vowel base + extension
  letters); Unicode 16.0 Ethiopic block U+1200–137F plus Supplement
  U+1380–139F plus Extended U+2D80–2DDF plus Extended-A
  U+AB00–AB2F. Used by Amharic (~32 M speakers), Tigrinya, Tigre,
  Ge'ez (liturgical), Oromo (some texts). Authority: Ethiopian
  Academy of Sciences.
- **Khmer (km)** — CLDR `km/characters.json`; Unicode Khmer block
  U+1780–17FF plus Khmer Symbols U+19E0–19FF. ~16 M speakers; sole
  in-scope locale. Authority: Cambodian Royal Academy.
- **Sinhala (si)** — CLDR `si/characters.json`; Unicode Sinhala
  block U+0D80–0DFF. ~16 M speakers. Brahmic-descended. Authority:
  Department of Official Languages (Sri Lanka).
- **Lao (lo)** — CLDR `lo/characters.json`; Unicode Lao block
  U+0E80–0EFF. ~30 M speakers (including Thai-Lao border regions).
  Brahmic-descended, very close to Thai structurally. Authority:
  Lao National Commission on Spelling.
- **Tibetan (bo)** — CLDR `bo/characters.json`; Unicode Tibetan
  block U+0F00–0FFF. ~6 M speakers across Tibet, Bhutan, parts of
  India/Nepal. Brahmic-descended. Used liturgically across Tibetan
  Buddhist communities far beyond native-speaker count.
- **Odia (or)** — CLDR `or/characters.json`; Unicode Oriya block
  U+0B00–0B7F (Unicode still uses "Oriya" naming). ~38 M speakers
  in Odisha, India. Brahmic-descended. Authority: Odia Bhasha
  Pratisthan.
- **Dhivehi (dv)** — CLDR `dv/characters.json`; Unicode Thaana
  block U+0780–07BF. ~340 K speakers in Maldives; the only RTL
  Brahmic-descended script in our scope (Thaana is a unique abugida
  that's written right-to-left like Arabic but derives from
  Arabic + Indic-numeral shapes). Authority: National Centre for
  Linguistic and Historical Research (Maldives).

## Coverage state

Per-script CLDR exemplar sizes (lowercase + index + numbers +
punctuation):

| locale | letters added to table | Unicode block coverage |
|---|---|---|
| am | 247 | Ethiopic + Suppl + Ext (full block-range fallback) |
| km | 99 | Khmer + Khmer Symbols |
| si | 74 | Sinhala |
| lo | 54 | Lao |
| bo | 87 | Tibetan |
| or | 64 | Oriya |
| dv | 35 | Thaana |

All script-range-fallback ranges are configured in
`SCRIPT_FALLBACK_RANGES` (see `build_char_language_bitmask.py`).
`SCRIPT_PREFIXES` and `LOCALE_SCRIPT_COMPAT` in `_common.py` have
the corresponding `ETHIOPIC` / `KHMER` / `SINHALA` / `LAO` /
`TIBETAN` / `ORIYA` / `THAANA` entries.

## Decisions

1. **Single-locale-per-script for all seven.** Each script gets one
   family bit echoing its language bit (the same pattern as
   `Hangul-family`, `Devanagari-family` in v3.0 — modified to
   multi-locale only when sister languages exist; none here).
2. **No per-locale auxiliary additions.** CLDR exemplars are the
   sole source, plus the standard four closures (case is a no-op
   for these scripts since none has uppercase/lowercase distinction
   except Ethiopic, which has none either; NFD; script-range
   fallback; post-fallback NFD).
3. **No special-case handling for sister-but-not-modeled
   languages**: Amharic's Ethiopic block also serves Tigrinya
   (Ethiopia / Eritrea), Tigre, Ge'ez (liturgical), Oromo (some
   texts); Khmer is occasionally used for Pali liturgical
   transcription; Tibetan serves Dzongkha (Bhutan), Ladakhi, Sherpa,
   etc.; Lao is essentially monolingual. **All Brahmic / Ethiopic
   chars in scope locales' CLDR sets get the host-language bit;
   chars outside CLDR get the script-range-fallback locale bit.**
   This is the same permissive-rejection rule used by Devanagari /
   Bengali / etc. in v3.0.

## Followups

- **If the C3 extension or a future Apertus snapshot adds any of
  these scripts to the vocab**, no code change needed — the build
  pipeline already covers them. Just re-run apply and the new
  tokens will get the correct script/family/language bits.
- **Per-script sister-language additions** (Tigrinya `ti` for
  Ethiopic, Dzongkha `dz` for Tibetan, Sherpa `xsr`, etc.):
  out of scope today, audit-driven if ever needed.
- **Khmer's special symbols** (U+19E0–19FF, lunar-date / day-of-
  week markers): handled by the Khmr fallback range. Empirically
  zero impact on Apertus vocab.
- **Thaana RTL handling**: codepoints don't need any special
  attribute; renderer handles directionality at display time.
  Bidi marks (LRM / RLM) are substrate-category and handled by the
  existing rule.
- **Mismatch with FineWeb-2 / ISO 639-3 individual / macro codes**:
  - `ory` (ISO 639-3 individual, Odia) — primary in our `or`
    entry. `ori` (macrolanguage) is in aliases.
  - `bod` (ISO 639-3 individual, Tibetan) — primary in `bo`.
    `tib` (the alpha-3 alternative) is in aliases.
  All consumer key forms resolve correctly.

# Char-tool feedback — from the PMI promotion consumer

> Author: the consumer at
> `02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/`.
> Date: 2026-05-15.
> Context: I built the multi-language PMI promotion pass (spec at
> `02_2_4_language_category_promotion/PMI_PROMOTION_SPEC.md`) against the
> v4 artifacts. This is a report on what worked, what hurt, and what
> I'd ask for in the next iteration.

The v4 hierarchical schema (script_and / family_and / bitmask_and) is
**working as advertised** for the languages it covers. T0 detection
falls out cleanly, the popcount-based substrate filter is correct,
and the cross-script overlap test shows **zero leakage** across the
87 cap-hit canonical keys — the char mask perfectly partitions Latin
/ Cyrillic / Han / Arabic / etc. The complaints below are about
**coverage and ergonomics**, not correctness.

## Hard pain points

### 1. 34 of the 87 cap-hit canonical keys have no char-tool mapping

(Originally 38 in our first run — 4 turned out to be consumer-side
mapping bugs from missing ISO 639-3 alternative codes. See finding #2.
After the fix: 34 keys genuinely fall outside the 55-locale list.)

We ran PMI promotion on 87 language samples ≥ 1 B tokens each. **39 %
of them** (34 keys) couldn't get the char-admissibility filter and
fall back to "Variant B only" (rate-test, no char mask).

Categorised by impact:

**Tier 1 — major-language additions worth prioritising** (10–200 M
speakers, clear linguistic identity, often a distinct script):

| canonical_key | language | speakers | gap reason |
| --- | --- | --- | --- |
| `swh_Latn` | Swahili | ~200 M | Bantu Latin |
| `mar_Deva` | Marathi | ~80 M | Devanagari, major Indic |
| `fil_Latn` | Filipino | ~80 M | Austronesian Latin |
| `amh_Ethi` | Amharic | ~32 M | Ethiopic script (not modelled) |
| `khm_Khmr` | Khmer | ~16 M | Khmer script (not modelled) |
| `npi_Deva` | Nepali | ~16 M | Devanagari |
| `sin_Sinh` | Sinhala | ~16 M | Sinhala script (not modelled) |
| `lao_Laoo` | Lao | ~30 M | Lao script (not modelled) |
| `kaz_Cyrl` | Kazakh | ~13 M | Cyrillic |
| `bel_Cyrl` | Belarusian | ~3 M | Cyrillic |
| `ceb_Latn` | Cebuano | ~20 M | Austronesian Latin |
| `ory_Orya` | Odia | ~38 M | Odia script (not modelled) |
| `bod_Tibt` | Tibetan | ~6 M | Tibetan script (not modelled) |
| `tgk_Cyrl` | Tajik | ~8 M | Cyrillic Persian |

**Tier 2 — also missing, smaller impact but cleanly modellable in
existing scripts:**

| canonical_key | language | gap reason |
| --- | --- | --- |
| `afr_Latn` | Afrikaans | Germanic Latin |
| `eus_Latn` | Basque | language isolate, Latin |
| `cym_Latn` | Welsh | Celtic Latin |
| `glg_Latn` | Galician | Romance Latin |
| `zsm_Latn` | Standard Malay | Austronesian Latin |
| `bos_Latn` | Bosnian | BCMS — may overlap with hr/sr-Latn |
| `kir_Cyrl` | Kyrgyz | Cyrillic |
| `khk_Cyrl` | Khalkha Mongolian | Cyrillic |
| `uzn_Cyrl` / `uzn_Latn` | Uzbek (both scripts) | digraphia |
| `ary_Arab` | Moroccan Arabic | Arabic dialect |
| `ckb_Arab` | Central Kurdish (Sorani) | Arabic-script Kurdish |

**Tier 3 — edge cases:**

| canonical_key | what it is |
| --- | --- |
| `als_Latn` | Alemannic German (Wikipedia code; ISO 639-3 `als` is Tosk Albanian — naming conflict) |
| `lat_Latn` | Latin (classical) — no living-language CLDR exemplar, but exists historically |
| `gmh_Latn` | Middle High German — historical |
| `div_Thaa` | Dhivehi (Maldivian) — Thaana script, not modelled |
| `und_Cyrl`, `und_Grek`, `und_Kana`, `und_Mong` | undetermined-language samples (FineWeb-2's "und" tag — language not identified by the source). Could potentially be handled by script-only attribution, no language bit needed. |

**Ask:** expand the in-scope language list. Prioritise Tier 1.
Tier 1 collectively fired ~14 B tokens that we can't currently
char-certify. The scripts not yet modelled (Khmer, Sinhala, Lao,
Odia, Tibetan, Ethiopic, Thaana) each need a small per-script
research note like the v3 plan template; the rest are mechanical
CLDR pulls.

### 2. The `iso639_3 ↔ char_tool_code` mapping isn't published — 4 silent bugs found and fixed

The consumer needs to take canonical keys like `deu_Latn` (ISO 639-3 +
ISO 15924) and look up the right char-tool language code (BCP 47:
`de`). I had to **hardcode a ~50-entry dict** in `build.py`:

```python
ISO_639_3_TO_BCP47 = {
    "eng": "en", "deu": "de", "fra": "fr", "ita": "it", "spa": "es",
    "por": "pt", "nld": "nl", "rus": "ru", ...
}
```

**This is fragile and caused 4 silent bugs in our first run.** When I
checked the unmapped list, I noticed the following keys had been
silently treated as unmapped:

| canonical_key | should have mapped to | bug cause |
| --- | --- | --- |
| `srp_Cyrl` | `sr-Cyrl` (in char tool) | I had `srp` not in the dict — code returned None before reaching the script disambiguation |
| `lvs_Latn` | `lv` (in char tool) | `lvs` is the ISO 639-3 *individual* code for Standard Latvian; `lav` is the *macrolanguage*. I had `lav` only. |
| `ekk_Latn` | `et` (in char tool) | same pattern — `ekk` (Standard Estonian, individual) vs `est` (macrolanguage) |
| `cmn_Hani` | `zh-Hans` (default) | FineWeb-2 uses `cmn` (individual) + `Hani` (generic Han script) — needed both `cmn` in dict and a `Hani→Hans` default |

These 4 keys collectively cover ~4 B tokens of (Serbian Cyrillic +
Standard Latvian + Standard Estonian + Mandarin Chinese). All four
should "obviously" be in the char tool's scope — they were just
hidden by the ISO 639-3 macrolanguage / individual-language
distinction and by the FineWeb-2 `Hani` script tag.

After fixing the consumer-side mapping the unmapped count dropped
from 38 → 34. But the bug existed for two days before I noticed it,
because nothing in the pipeline complained.

**Ask:** add an `iso_639_3` field — and explicitly its **alternative
ISO codes** — to each entry in the manifest's `languages` list:

```json
{ "bit": 0, "code": "en", "iso_639_3": ["eng"],
  "cldr_locale": "en", "name": "English", "script": "Latn" }

{ "bit": 50, "code": "et", "iso_639_3": ["est", "ekk"],
  "cldr_locale": "et", "name": "Estonian", "script": "Latn" }

{ "bit": 19, "code": "zh-Hans", "iso_639_3": ["zho", "cmn"],
  "iso_15924": ["Hans"],
  "iso_15924_aliases": ["Hani"],
  "cldr_locale": "zh-Hans", "name": "Chinese (Simplified)", "script": "Hans" }
```

Then a consumer can build the lookup directly from `manifest.json` —
both the primary `iso_639_3` and the alternative individual /
macrolanguage codes — without hand-curation. Missing-key bugs become
loud: the consumer can detect "canonical key references an iso639_3
not in the manifest" at build-time.

A second `aliases` list per script (e.g., `Hani → Hans` as a stated
default) makes generic-script tags resolvable.

### 3. Multi-script-language disambiguation is undocumented

The char tool has separate codes `zh-Hans`, `zh-Hant`, `sr-Latn`,
`sr-Cyrl`, `el`, `el-polyton`. The consumer's canonical key uses
`iso639_3 + ISO_15924_script` (`zho_Hans`, `srp_Latn`, `ell_Grek`).

The consumer has to *know* that `zho_Hans → zh-Hans`,
`srp_Latn → sr-Latn`, and that `ell_Grek` could mean either `el` (the
sample is monotonic Greek) or `el-polyton` (we picked `el`, but the
choice is implicit).

**Ask:** publish a helper at the char-tool level, either:

- a function `code_from_key(canonical_key) → char_tool_code` shipped
  alongside `_common.py`, OR
- the iso639_3 field above plus a documented rule
  ("`{iso_639_3}_{script}` → `char_tool_code` after disambiguation
   by script suffix").

### 4. `ell_Grek` ↔ `el` / `el-polyton` ambiguity

We have one canonical key for Greek (`ell_Grek`, the FineWeb2-HQ
sample). The char tool models `el` and `el-polyton` as separate
languages, with different exemplars.

Our PMI sample is mostly modern Greek but **may contain occasional
polytonic forms** in historical quotes, classical references, place
names, etc. Under the strict char-mask, polytonic chars fail the
`el`-admissibility check (because polytonic letters like `ἀ` are in
`el-polyton`'s exemplar, not `el`'s).

Empirically, this isn't dramatic — Greek's masked set has 87 % mass
coverage, suggesting polytonic leakage is small in our sample. But
two cleanups would help:

- **Document** in the manifest which locales' exemplars are subsets
  of which (e.g., monotonic Greek chars are a strict subset of
  polytonic Greek chars, so a token admissible in `el` is always
  admissible in `el-polyton`).
- Consider an **`el-tolerant` variant** in the manifest that admits
  common-but-rare polytonic forms (those that appear in modern Greek
  texts even after the 1982 orthography decree).

### 5. The substrate "self-cancellation under PMI" assumption fails under domain shift

This isn't a char-tool bug per se, but worth documenting **as part of
the consumer guidance** in the char-tool README. We expected substrate
tokens (popcount = 55) to have PMI ≈ 0 in every language by symmetry.
In practice, when language samples come from **different corpus
sources** (Wikipedia for `eng_Latn`, FineWeb2-HQ for `deu_Latn`,
small-Wikipedia for `als_Latn`), substrate firings vary noticeably by
domain. The validation check showed 73 % of substrate tokens have
|PMI| > 0.30 in `als_Latn`.

The masked variant of our PMI promotion explicitly excludes substrate
via `popcount < N_LANG_BITS`, so this doesn't affect the canonical
output. But the unmasked variant promotes some substrate tokens, and
that's a behavioural surprise.

**Ask:** in the char-tool documentation (or in `PLAN.md`'s "downstream
consumer notes"), add a paragraph: *"Substrate tokens may have
non-zero PMI under sample-domain mismatch — consumers using PMI for
language attribution should rely on the popcount-based substrate
filter rather than expecting PMI self-cancellation."*

### 6. `cmn_Hani` not in the in-scope list

Looking at the FineWeb-2 keys we have: there's `cmn_Hani` (Mandarin
Chinese, Han script — the FineWeb-2 standard tag for written Chinese)
but the char tool uses `zh-Hans` / `zh-Hant` as separate codes. Our
sample's `cmn_Hani` falls back to Variant B only.

**Ask:** either alias `cmn_Hani` to one of `zh-Hans`/`zh-Hant` in
the manifest (with a note about which), or document the resolution
the consumer should use (pick the one with more glyph overlap?
Promote against the union of the two?).

## Smaller asks

- **Norwegian Nynorsk (`nno_Latn`)** isn't in the char tool. nb is. nb
  is Norwegian Bokmål. Nynorsk is the other written standard for
  Norwegian; it's distinct in vocabulary.
- The `families.yaml` taxonomy is great but **family bits per
  cross-script linguistic family** would be useful — e.g., Slavic is
  split into `Slavic-Latn` (Polish/Czech/…) and `East-Slavic-Cyrl`
  (Russian/Ukrainian/…) and `South-Slavic-Cyrl` (Bulgarian). A
  "Slavic" bit cutting across scripts would let consumers ask "is this
  token admissible in any Slavic language?" with one bit lookup. Same
  for Indo-Iranian (Persian + Urdu + Hindi) etc. (Lower priority —
  derivable in the consumer.)
- The `category` field in `char_language_bitmask.parquet` exists per
  codepoint. **A per-token category aggregate** (e.g., "this token's
  chars are all in Unicode letter categories L/M plus optional Z") in
  `token_language_bitmask.parquet` would let consumers do the
  natural-text vs code-mixed classification (separating ` der` from
  `{H` or `.Tests`) without reimplementing Unicode-category logic.

## What's working well — keep it

- v4 hierarchical layout is the right abstraction.
- The validation phase (substrate-has-all-bits, family derivation
  consistency) is the kind of rigour I'd want everywhere.
- The `cldr_subsets_included` / `cldr_subsets_excluded` provenance in
  the manifest is clear and reproducible.
- T0 detection (popcount-1 ∩ has-L-bit) gives clean char-evidenced
  language anchors for every distinctive-script language. **ß-bearing
  for German, Greek-letter for el, ñ for Spanish, ą/ę/ć/ń/ś/ź/ł for
  Polish, etc. all just work.**

## Summary

The char tool's **correctness** is excellent — zero cross-script
leakage in the PMI overlap matrix proves it. The **coverage** is
the gap. Closing the 38-unmapped-key hole, publishing the
`iso639_3 → char_tool_code` map, and clarifying multi-script and
polytonic-Greek handling would make the next consumer's life
substantially easier.

Estimated work for the consumer-side fixes:

| ask | effort |
| --- | --- |
| publish `iso_639_3` field in manifest | 10 min (add one field to `languages.yaml`) |
| document multi-script aliasing | 20 min |
| add `cmn_Hani` and `nno_Latn` aliases | 30 min |
| substrate-non-cancellation note in README | 10 min |
| Add 5–10 high-impact missing languages (mar, bel, mlt, gle, lao, …) | 4 h per language (CLDR pull + per-script note) |
| Per-token Unicode-category aggregate column | 1 h |

The first four are quick. The language additions are the main
investment, and they're the only piece that requires substantive new
research (per-script CLDR cross-referencing per the v3 plan template).

# Latin — per-script research notes

> European-and-adjacent locales in scope: 26 (en, cs, da, de, es,
> fr, hu, it, nl, pl, pt, sv, ro, sr-Latn, fi, nb, sl, hr, sk, et,
> lt, lv, ca, is, tr, az). The two non-European Latin locales (id,
> vi) are also in `languages.yaml` and covered identically by CLDR;
> we count them with the others when reporting Latin coverage.
>
> Status: coverage exhaustive; the **token-level discrimination
> within Latin is intrinsically limited** because bare-ASCII tokens
> can come from any Latin language. The family layer (added in v3)
> is the main discrimination win.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `characters.json` for each of the
  28 in-scope Latin locales.
- Unicode 16.0: Basic Latin (U+0000–007F), Latin-1 Supplement
  (U+0080–00FF), Latin Extended-A (U+0100–017F), Latin Extended-B
  (U+0180–024F), Latin Extended Additional (U+1E00–1EFF), Latin
  Extended-D (U+A720–A7FF), Latin Extended-E (U+AB30–AB6F), IPA
  Extensions (U+0250–02AF), Spacing Modifier Letters (U+02B0–02FF).
- Wikipedia "X alphabet" pages for each in-scope locale — strong
  reference for what's "in the alphabet" per language.
- ISO 8859-1 / -2 / -3 / -4 / -9 / -10 / -13 / -14 / -15 / -16 —
  legacy charsets defining each European region's required IT
  characters.
- DIN 91379 (Germany, 2022) — authoritative for de including
  minority-language names within Germany.
- Real Academia Española (es), Académie française (fr), Türk Dil
  Kurumu (tr), Svenska Akademien (sv), Norwegian Language Council
  (nb), Hungarian Academy of Sciences (hu).

## Per-locale CLDR exemplar size

CLDR `main` exemplar lengths (rough char count of the spec string,
not parsed):

| locale | main_len | aux_len | notes |
|---|---|---|---|
| en | 53 | 49 | 26 lowercase + index uppercase; auxiliary contains French loans |
| cs | 77 | 49 | 41 letters incl. á č ď é ě í ň ó ř š ť ú ů ý ž |
| da | 59 | 35 | 28 letters incl. å æ ø |
| de | 58 | 53 | 30 letters incl. ä ö ü ß |
| es | 61 | 44 | 27 letters incl. ñ |
| fr | 71 | 35 | 26 + à â ç é è ê ë î ï ô œ ù û ü ÿ |
| hu | 157 | 49 | longest main: 44 letters incl. á é í ó ö ő ú ü ű plus digraphs cs sz zs etc. |
| it | 59 | 38 | 21 + à è é ì ò ù |
| nl | 73 | 28 | 26 + ij; auxiliary loanwords |
| pl | 65 | 37 | 32 letters incl. ą ć ę ł ń ó ś ź ż |
| pt | 66 | 38 | 26 + á à â ã ç é ê í ó ô õ ú |
| sv | 61 | 29 | 29 letters incl. å ä ö |
| ro | 63 | 23 | 31 letters incl. ă â î ș ț |
| sr-Latn | 70 | 11 | 30 letters incl. č ć đ š ž; minimal aux |
| fi | 61 | 109 | core 28; very large auxiliary set including all Sami chars |
| nb | 64 | 34 | 29 letters incl. å æ ø |
| sl | 51 | 59 | 25 + č š ž |
| hr | 70 | 9 | 30 letters incl. č ć đ š ž; nearly empty aux |
| sk | 90 | 45 | 46 letters — Slovak has the largest Latin alphabet by codepoint count incl. á ä č ď é í ĺ ľ ň ó ô ŕ š ť ú ý ž |
| et | 65 | 40 | 32 letters incl. ä õ ö ü š ž |
| lt | 58 | 142 | 32 letters; very large aux (includes accent-marked vowels for stress) |
| lv | 67 | 13 | 33 letters incl. ā č ē ģ ī ķ ļ ņ š ū ž |
| ca | 65 | 41 | 26 + à ç é è í ï ó ò ú ü |
| is | 65 | 9 | 32 letters incl. á ð é í ó ú ý þ æ ö |
| tr | 60 | 53 | 29 letters incl. ç ğ ı i İ ö ş ü (dotted/dotless i distinction!) |
| az | 66 | 3 | 32 letters incl. ə ç ğ ı İ ö ş ü (similar to tr + schwa) |

## Coverage state — Unicode block-by-block

| block | size | in table |
|---|---|---|
| Basic Latin (U+0000–007F) | 128 | 98 (78 % — non-printable control range excluded by category) |
| Latin-1 Supplement (U+0080–00FF) | 128 | 66 (50 % — includes all accented vowels + ñ ç ß æ ø etc.) |
| Latin Extended-A (U+0100–017F) | 128 | 83 (65 % — Czech / Polish / Slavic-Latn / Hungarian / Baltic letters) |
| Latin Extended-B (U+0180–024F) | 208 | 9 (4 % — most rare; sr-Latn / az / tr extras are here) |
| Latin Extended Additional (U+1E00–1EFF) | 256 | 90 (35 % — Vietnamese tone marks dominate) |
| IPA / Modifier blocks (U+0250–02FF) | 176 | 1 | mostly out of scope (linguistic transcription) |

The shortfall in Latin Extended-B and IPA / Modifier blocks is by
design — those blocks contain mostly characters for languages we
don't model, IPA transcription, and rare modifications. Apertus's
Latin vocab tokens don't depend on them.

## Empirical Apertus baseline

- **110,315 Apertus vocab tokens contain Latin codepoints** — by
  far the largest script in vocab (vs ~7K for Cyrillic, ~3.7K for
  Han, ~4.4K Hangul, ~1.5K Greek, ~1K Hebrew).
- **323 unique Latin codepoints** used in the vocab.

Single-family token discrimination (token AND narrows to exactly
one Latin family):

| family | tokens narrowing to it alone | discriminating chars |
|---|---|---|
| Vietic-Latn | 786 | Vietnamese tone-mark combinations from Latin Extended Additional |
| Slavic-Latn | 624 | ą ć č ď ę ł ń ř ś š ť ž ż etc. |
| Turkic-Latn | 568 | ç ğ ı İ ö ş ü ə + dotted/dotless i (shared tr + az) |
| Romance-Latn | 323 | ñ ç à â ê ë etc. — note many Romance accents are shared with Slavic-Latn |
| Germanic-Latn | 171 | ß ø æ ð þ — distinctive only because Slavic / Romance don't admit them |
| Baltic-Latn | 99 | ā ē ī ū plus macron family — shared with lv ē ī but distinctive to Baltic |
| Uralic-Latn | 85 | ő ű (specifically Hungarian) plus some Finnish/Estonian-specific |

**Total: 2,656 of ~110,315 Latin tokens (≈ 2.4 %) AND-narrow to a
single Latin family.** The remaining 97.6 % AND-attribute to
multiple families (bare ASCII covers all 8; common shared diacritics
like é cover Romance + Germanic + others).

Single-language token discrimination (token AND narrows to exactly
one Latin **language**):

| language | tokens narrowing to it alone |
|---|---|
| pl | 371 (Polish ł ą ę ń ś ź ż) |
| cs | 201 (Czech ř ů — Slavic-distinctive within Slavic) |
| az | 160 (Azerbaijani schwa ə) |
| ro | 134 (Romanian ș ț ă ) |
| pt | 121 (Portuguese ã õ — distinctive vs other Romance) |
| de | 103 (German ß) |
| es | 88 (Spanish ñ) |
| hu | 85 (Hungarian ő ű) |
| lv | 64 (Latvian ā ē ī ū with macron) |
| fr | 44 (œ + specific combos) |
| lt | 23 (Lithuanian ą ę ė į ų — partial overlap with pl) |
| it | 22 (Italian-specific; mostly à è ò ù in particular combos) |
| is | 12 (Icelandic ð þ) |
| sk | 6 (Slovak ä ĺ ŕ ô — narrow because most overlaps with cs) |

Total: ~1,434 tokens that AND-resolve to a single European Latin
language. That's ≈ 1.3 % of Latin tokens. The remaining 98.7 % are
either bare-ASCII (28 Latin languages all admit) or shared-diacritic
(several languages admit).

**This is the design intent.** Within Latin script, codepoint-level
discrimination is intrinsically limited because most letters are
shared. The family layer adds real discrimination (2.4 % single-
family vs 1.3 % single-language); the script layer is the strongest
discriminator at the broadest level (every Latin token is `script_and
= {Latn}`).

## Discrimination expectations (per the v3 design)

Reading the three layers for a Latin token:

- `script_and = {Latn}` always for any Latin token: tells you it's
  Latin-script.
- `family_and = {Romance-Latn}` (or similar single-family): tells
  you it's distinctively in that family.
- `bitmask_and = {de}` (or similar single-locale): tells you it's
  distinctively in that locale.

For bare-ASCII tokens, only the script bit is informative; family
and language are saturated. For tokens with distinctive diacritics
or letters, all three layers narrow simultaneously.

Consumers who want to attribute bare-ASCII Latin tokens to a single
language need to look beyond the bitmask — n-gram statistics,
training-data co-occurrence, etc. The bitmask cannot do this
discrimination *by construction* because the chars admit it.

## Decisions

1. **No `languages.yaml` changes.** All 28 in-scope Latin locales
   have correct CLDR-derived coverage; closures (case, NFD,
   script-range fallback for Latin Extended Additional via the v2
   work) are working.
2. **The ~98 % family-non-discrimination rate within Latin is
   expected**, not a bug. Document in PLAN.md.
3. **No new locales to add right now** — the residual Latin
   fall-through (42 tokens at audit time, mostly Esperanto + Japanese
   romaji `ō` + Icelandic-residual) is well under the 50-token
   audit threshold for in-scope scripts. If we ever want to close
   it: add `eo` (Esperanto, ~30 tokens) at bit 55+.

## Followups

- **Esperanto (`eo`)** at bit 55: would close most of the residual
  Latin fall-through (`ĉ ĝ ĥ ĵ ŝ ŭ`). Cheap addition. Whether to
  do it depends on whether the user wants Esperanto in scope at
  all — it's a constructed language; Apertus saw very little of it.
- **Japanese romaji compatibility-form `Ō` `ō` `ū`** (with macron):
  these appear in Apertus tokens for Japanese romanization (`Tōkyō`,
  `kōbe`). Currently 0-bit because they're Latin chars but not in
  any in-scope locale (Latvian uses macron but `ō` isn't in Latvian
  alphabet either). Closing this would require adding a "Japanese-
  romaji" or "Hawaiian" locale — out of scope.
- **Faroese (`fo`)** uses ð like Icelandic; if a future scope
  expansion adds it, the Icelandic-only narrowing for `ð` would
  spread to {is, fo}. Defer.
- **Catalan (`ca`)** narrowing — added in v2 but appears in only
  a handful of tokens (`l·l`, `ç` shared with fr/pt/tr). Working
  as designed; CLDR coverage verified.

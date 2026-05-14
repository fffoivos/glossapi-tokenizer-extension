# Plan v3 — Hierarchical (script / family / language) char categorization

> Status: **active**. Shipped in the v4 artifact schema. PLAN.md
> (the v2.2 design) is preserved as the prior baseline and remains
> useful for understanding the language-layer mechanics that this
> plan extends.

## Scope — and what this is *not*

This plan covers **char identification** only. For each Unicode
codepoint that the Apertus vocab might decode to, we record which
**(scripts, language-families, languages)** admit it.

What that means in practice:

- Three parallel layers of categorization at the codepoint level.
- The bits *are* the categorization. We don't pick a single answer
  per char; we record the (multi-membership) facts.
- Specificity at the char level means **how narrow the bit set is**.
  `ñ` is more specific (1 language, 1 family) than bare `a` (28
  languages, 8 families) — the artifact records both honestly.

Explicitly **not** in scope:

- **Token classification** — picking *the* language a token comes
  from. Different decision procedure; needs frequency / morphology /
  n-gram signal that CLDR membership can't provide. The hierarchical
  char masks are designed to be *inputs* to such a classifier, but
  the classifier itself lives elsewhere.
- **Anomaly / outlier detection** — flagging "surprising" chars,
  cross-family overlaps, etc. Those queries are downstream: read
  the bit columns, popcount or filter as the consumer needs. We
  emit no curated audit artifact at build time.
- **Confidence / probability scoring** — bits remain 0/1.

The strict separation: char categorization at this layer answers
*"which categories admit this char?"*. Anything else is a different
problem.

## The problem with v2.2

v2.2 has 55 language-level bits. A token's `bitmask_and` over its
codepoints gives "languages compatible with every char". For a bare-
ASCII Latin token like ` the` or `tion`, the AND is all 28 Latin-
script bits — every Latin language admits every ASCII letter. The
artifact reports "compatible with 28 specific languages" when it
*could* report "Latin-script: yes, narrower than Latin: no signal".

What we want is the same rejection model at **three levels of
granularity** stored in parallel:

| level | unit | example for bare-ASCII `the` |
|---|---|---|
| **Script** | ISO 15924 (Latn, Cyrl, Grek, …) | `{Latn}` |
| **Family** | language family within a script | `{Romance-Latn, Germanic-Latn, Slavic-Latn, Uralic-Latn, Baltic-Latn, Turkic-Latn, Austronesian-Latn, Vietic-Latn}` |
| **Language** | individual locale (current 55 bits) | `{en, cs, da, de, …, ca, is}` |

A token containing `ß` would narrow at *every* level — script
`{Latn}`, family `{Germanic-Latn}`, language `{de}` — which is the
real value of the multi-level encoding: the more specific a char is
intrinsically, the more it narrows at every level.

## Level 1 — Script (~22 bits)

A bit per Unicode script we model. Derived from `char_script(cp)`
(NFKC-aware) plus the locale-script declarations in `languages.yaml`.

| bit | script | locales |
|---|---|---|
| 0 | Latn | en, cs, da, de, es, fr, hu, id, it, nl, pl, pt, sv, tr, vi, ro, sr-Latn, az, fi, nb, sl, hr, sk, et, lt, lv, ca, is |
| 1 | Cyrl | ru, uk, bg, mk, sr-Cyrl |
| 2 | Grek (modern) | el |
| 3 | Grek-polyton | el-polyton |
| 4 | Arab | ar, fa, ur |
| 5 | Hans | zh-Hans |
| 6 | Hant | zh-Hant |
| 7 | Jpan (Han + Kana) | ja |
| 8 | Hang | ko |
| 9 | Deva | hi |
| 10 | Hebr | he |
| 11 | Thai | th |
| 12 | Armn | hy |
| 13 | Geor | ka |
| 14 | Beng | bn |
| 15 | Taml | ta |
| 16 | Telu | te |
| 17 | Knda | kn |
| 18 | Mlym | ml |
| 19 | Gujr | gu |
| 20 | Guru | pa |
| 21 | Mymr | my |

Stored as `binary(16)` — uniform width across script / family /
language masks. See § Storage / schema for the rationale.

## Level 2 — Family (within script)

A "family" = a cluster of in-scope languages that share a script and
have meaningful structural overlap. Practical taxonomy:

**Latin script** (8 families):
- `Romance-Latn`: es, fr, it, pt, ro, ca
- `Germanic-Latn`: en, de, nl, da, nb, sv, is
- `Slavic-Latn`: pl, cs, sk, sl, hr, sr-Latn
- `Uralic-Latn`: hu, fi, et
- `Baltic-Latn`: lt, lv
- `Turkic-Latn`: tr, az
- `Austronesian-Latn`: id
- `Vietic-Latn`: vi

**Cyrillic script** (2 families):
- `East-Slavic-Cyrl`: ru, uk
- `South-Slavic-Cyrl`: bg, mk, sr-Cyrl

**Arabic script** (2 families):
- `Semitic-Arab`: ar
- `Iranian-Arab`: fa, ur

**CJK** (3 families):
- `Sinitic-Hans`: zh-Hans
- `Sinitic-Hant`: zh-Hant
- `Japonic`: ja (Han + Kana)

**Greek** (2 families, matching the encoding split):
- `Grek-modern`: el
- `Grek-polyton`: el-polyton

**Single-locale scripts** — each gets its own family bit, for
symmetry. Every codepoint then has an answer at every level even if
the family bit is a deterministic echo of the script bit. Consumers
never have to special-case "no family bit at this position".

- `Hangul-family`: ko
- `Devanagari-family`: hi
- `Hebrew-family`: he
- `Thai-family`: th
- `Armenian-family`: hy
- `Georgian-family`: ka
- `Bengali-family`: bn
- `Tamil-family`: ta
- `Telugu-family`: te
- `Kannada-family`: kn
- `Malayalam-family`: ml
- `Gujarati-family`: gu
- `Gurmukhi-family`: pa
- `Myanmar-family`: my

Total: 8 (Latn) + 2 (Cyrl) + 2 (Arab) + 3 (CJK) + 2 (Grek) + 14
(single-locale) = **31 families**, stored in `binary(16)` for
uniformity with the other mask columns.

## Level 3 — Language (55 bits, unchanged)

The existing per-language bits and their wire positions stay stable.

## How chars get bits at each level

**Strict rule unchanged.** Language-level bits come from positive
CLDR evidence plus the four documented closures (case, per-locale
NFD, script-range fallback, post-fallback NFD) plus the substrate
override. Script and family bits are **derived deterministically by
projection from language bits**:

- `family_bits[f] = 1 iff` `lang_bits & (OR of language_bits[L]
  for every locale L in family f) != 0`.
- `script_bits[s] = 1 iff` `lang_bits & (OR of language_bits[L]
  for every locale L in every family assigned to script s) != 0`.

Substrate codepoints get ALL bits set at every level — same rule as
today, applied uniformly across the three masks.

### Why projection-only, not the "OR Unicode script" rule

An earlier draft of this plan said `script_bits[s] = 1 iff the
codepoint's NFKC-aware Unicode script is s OR any locale … has the
language bit set`. We deliberately do **not** ship the Unicode-script-
fallback half of that rule. Three reasons:

1. **The whole artifact is "what we model and can reject."** Adding
   a free Unicode-script-detection axis on top of the language and
   family layers mixes semantics: the language and family bits mean
   "an in-scope locale admits this char"; a Unicode-script-OR rule
   for script bits would mean "this char's Unicode block is in our
   scripts.yaml regardless of whether any modeled locale uses it."
   Consumers reading `script_and` would get a different kind of
   answer than `family_and` and `bitmask_and`. Splitting that out
   is a separate artifact, not a hidden override on this one.

2. **Consumers wanting broad Unicode-script detection can compute it
   themselves.** `unicodedata.name(chr(cp))` plus a script-prefix
   lookup gives the broad answer in a few lines; we don't need to
   bake it into the bitmask. The bitmask's job is to express which
   in-scope categories admit each char.

3. **It is the conservative extension of v2.2.** v2.2 had no script
   layer; every bit was a language bit, set only on positive
   evidence. v3 adds script and family layers as derived projections
   of that same language evidence. No new evidence kinds, no new
   inclusion rules — just two coarser views of the same data.

Concrete consequence: a Latin codepoint outside every in-scope
locale's CLDR (e.g. `ō` U+014D — used in Japanese romaji, Latvian,
Hawaiian, Māori; none in our scope) gets **zero bits at every
level**. A token containing it AND-rejects every in-scope (script,
family, language). Same rejection semantics consumers already know
from v2.2 — now extended to the two new layers without contradiction.

There is one place where this question initially looked harder: the
Greek collision. `Grek-modern` and `Grek-polyton` both declare
`iso15924: Grek`, so a `name`-prefix match on "GREEK" would set both
script bits indiscriminately and collapse the encoding distinction.
Under the projection-only rule the collision is moot — α has both
`el` and `el-polyton` language bits and ends up with both Greek
script bits via the family→script mapping; ἀ has only `el-polyton`
and ends up with only `Grek-polyton`. The encoding split is
preserved exactly because the projection follows the language layer.

## Token-level — the *same* rejection model, three resolutions

`apply_to_apertus_vocab.py` emits parallel AND/OR aggregations at
each level:

| consumer question | column to AND |
|---|---|
| Which **scripts** can this token come from? | `script_and` |
| Which **families**? | `family_and` |
| Which **languages**? (current) | `bitmask_and` |

This is char categorization applied via AND — *not* a token
classifier. A consumer that wants single-language attribution still
needs additional signal beyond what we emit.

## Per-script research plan — questions and resources

CLDR is the baseline, but our scope claim ("this char is in
language X's normal text") needs verification per script. For each
script we need to answer a fixed set of questions and consult a
fixed set of authoritative resources. Workflow per script:

1. Audit Apertus's vocab for tokens in this script (we already have
   the v2.2 token table — filter by Unicode block).
2. Anchor on CLDR exemplar for each in-scope locale.
3. Cross-reference with the authoritative source(s) below to find
   chars CLDR omits or includes too broadly.
4. Decide per char whether to add/remove a bit. Most gaps are
   already handled structurally by case/NFD/script-fallback
   closures; the remaining ones are auxiliary loan letters or
   compatibility forms.
5. Document the per-char decisions in `notes/<script>.md` inside
   this sub-subproject so future-us knows *why*.

Priority order = vocab-token impact: Cyrillic (132), Latin (42),
Arabic (38), then the rest under 50.

### Generic questions (apply to every locale)

1. **Canonical alphabet** — official letter set, case-paired.
2. **Diacritics** — which combinations are normal text, in both
   NFC and NFD forms.
3. **Auxiliary chars in normal text** — letters from other languages
   that appear unchanged (loanwords, foreign names). CLDR's
   `auxiliary` covers some but we dropped it under v2.2; revisit
   per locale to decide which we want back.
4. **Orthographic reforms** — pre-reform spellings still encountered
   (Russian 1918, German 1996, Portuguese 1990, Greek 1982).
5. **Punctuation** — locale-specific vs Latin-convention shared.
6. **Numerals** — Latin digits, native digits, or both?
7. **Cross-locale overlap to expect** — sister locales with
   near-complete char overlap (es/ca, da/nb, ru/uk core).

### Latin (28 locales — largest scope)

**Specific questions**:
- Core alphabet vs only-loanwords-admitted per locale (Polish q/v/x;
  English natively avoids diacritics but admits *café*, *naïve*).
- Multi-script locales — sr-Latn vs sr-Cyrl, az's Latin / Cyrillic /
  Arabic history.
- Encoded ligatures as single codepoints (ß, æ, œ, compat forms
  ﬁ ﬀ).
- Orthographic reforms — German ß ↔ ss, Portuguese accents.

**Resources**:
- **CLDR exemplar per locale** — `main + index + numbers + punctuation`.
- **Wikipedia "X alphabet"** pages — usually maintained by native
  speakers; surprisingly authoritative. Vietnamese alphabet entry
  enumerates all 89 tone-mark combinations of Quốc ngữ.
- **ISO 8859 family** (Latin-1 Western, Latin-2 Central European,
  Latin-3 South European, etc.) — legacy charsets, informally
  enumerated each region's required IT chars.
- **DIN 91379 (German)** — government-mandated char set for names
  from minority languages within Germany; authoritative for `de`
  auxiliary scope.
- **Native-language official references** — Real Academia Española
  (es), Académie française (fr), Türk Dil Kurumu (tr), Norwegian
  Language Council (nb), Svenska Akademien (sv).
- **Unicode TR #36 "Unicode Security Considerations"** — confusable
  Latin spoofs (relevant for OCR / mixed-script text auditing).

### Cyrillic (5 in scope, 132-token coverage gap)

**Specific questions**:
- Post-reform standard alphabet per in-scope locale (Russian 33,
  Ukrainian 33, Bulgarian 30, Macedonian 31, Serbian-Cyrl 30).
- Letters unique to one locale: uk's `і ї є ґ`, mk's `ј љ њ ѓ ќ ѕ џ`,
  bg's `ъ` semantics differs from Russian.
- For the out-of-scope coverage gap audit: which of Kazakh,
  Belarusian, Mongolian, Tatar, Bashkir do we add at bits 55+?

**Resources**:
- CLDR per locale.
- **Wikipedia "Cyrillic alphabets"** — comparative table across all
  Cyrillic-using languages; very strong reference.
- **ISO 8859-5 / KOI8-R / CP1251** — legacy Cyrillic charsets.
- Per-language language academies — Russian Academy of Sciences
  Institute of Linguistics for ru, etc.

### Greek (el monotonic + el-polyton)

**Specific questions** (mostly settled; verify):
- Monotonic — exactly which acute/diaeresis combinations are normal
  post-1982?
- Polytonic — full set of breathing + accent + iota subscript
  combinations? (Already verified by reviewer for v2.2 build.)
- Greek numerals (α´ β´ γ´) — admitted in modern texts?

**Resources**:
- CLDR `el` and `el-polyton`.
- **Greek government 1982 decree on monotonic orthography** —
  defines exactly which polytonic marks were eliminated.
- **Wikipedia "Polytonic orthography"** + classical Greek refs.
- **TLG (Thesaurus Linguae Graecae)** — UC Irvine's classical Greek
  corpus, the standard for polytonic encoding.

### Arabic-script (ar, fa, ur — and Pashto/Sindhi/Uyghur gap)

Most per-locale divergence of any script in our scope.

**Specific questions**:
- Standard Arabic 28 letters + per-locale additions:
  - **fa (Persian)**: `پ چ ژ گ`, uses `ی` U+06CC (not U+064A) and
    `ک` U+06A9 (not U+0643).
  - **ur (Urdu)**: `ٹ ڈ ڑ ے ں` retroflex + extras.
  - **ps (Pashto, out of scope)**: `ښ ږ ځ څ ړ`.
  - **sd (Sindhi, out of scope)**: `ڪ ڳ ڙ ڄ ڃ ٿ ڌ`.
  - **ug (Uyghur, out of scope)**: `ۇ ۈ ۆ ۋ ې`.
- **Positional forms** — Arabic Presentation Forms-A (U+FB50-FDFF)
  and -B (U+FE70-FEFF). Modern text uses base letters and renderers
  pick forms, but presentation forms still appear in legacy PDFs.
  **Check Apertus's vocab** for them — they may be tokens we should
  treat differently (substrate vs base-letter).
- **Ligatures** as single codepoints — اللّٰه U+FDF2.
- ZWJ / ZWNJ usage in compound words.
- **Quranic punctuation** (U+06D6-06ED) — appears in religious text;
  decide whether to model.

**Resources**:
- CLDR per locale.
- **Wikipedia "Arabic alphabet", "Persian alphabet", "Urdu alphabet"**
  — per-locale letter inventories with codepoints.
- **Unicode Standard chapter 9 "Middle East-I"** — definitive Arabic-
  script reference.
- **Persian Language Academy (Farhangestan)** — definitive for fa.
- **Center for Research in Urdu Language Processing (CRULP)** —
  Urdu inventory.

### CJK (zh-Hans, zh-Hant, ja)

**Specific questions**:
- **Chinese (zh-Hans / zh-Hant)**:
  - CLDR covers only ~3K most-common Han chars. The script-range
    fallback (v2.2) covers all ~80K Han codepoints. Are there real
    locale-specific char distinctions we're missing?
  - Simplified ↔ Traditional pairing per codepoint (some chars exist
    in both forms as separate codepoints).
  - GB18030 (China mainland) vs Big5 (Taiwan / HK) coverage.
- **Japanese (ja)**:
  - Joyo kanji (常用漢字, 2,136 chars) — Japan's "regular use" list.
    Every educated speaker knows these.
  - Jinmeiyo kanji (人名用漢字) — additional names list.
  - Hiragana + Katakana — finite, fully enumerable.
  - Hentaigan (変体仮名) — historical kana variants; rare in modern
    text but might appear.

**Resources**:
- **Joyo kanji official list** (Japan Agency for Cultural Affairs)
  — single authoritative file.
- **GB18030 specification** (China mainland encoding standard) —
  defines the full Chinese char inventory including 70k+ Han.
- **Big5 specification** (Taiwan / Hong Kong traditional encoding).
- **Unicode UAX #38 "Unicode Han Database"** — per-codepoint Han
  metadata (Simplified ↔ Traditional variants, kBigFive / kGB /
  kJis source flags).
- **Wikipedia "List of jōyō kanji"**, "Hiragana", "Katakana".
- **CJKlib** (Python library) — programmatic access to Han variant
  relationships.

### Korean (ko, Hangul)

**Specific questions**:
- Modern Korean uses precomposed Hangul syllables (U+AC00-D7AF,
  11,172 codepoints) almost exclusively. Are decomposed Jamo
  (U+1100-11FF) ever in the corpus?
- **Hanja** (Han characters used in Korean) — appears in academic /
  legal text. Should a Hanja-containing token carry the `ko` bit, or
  only the Sinitic CJK bits, or both?
- Halfwidth Korean compatibility forms (handled by v2.2 NFKC-aware
  script detection — verify).

**Resources**:
- CLDR `ko`.
- **Wikipedia "Hangul Syllables", "Korean Hanja"** — clear
  inventories.
- **KS X 1001 / KS X 1002** — Korean industry standards.

### Indic family (hi, bn, ta, te, kn, ml, gu, pa)

Most complex orthographic rules of any group.

**Specific questions per Indic script**:
- Independent vowels + consonants + dependent vowel signs + virama
  + special marks.
- Conjunct consonant ligatures — encoded as `base + virama + base`
  sequences, sometimes precomposed.
- ZWJ / ZWNJ (U+200D / U+200C) — critical for conjunct formation.
  Are these in Apertus's vocab as bytes?
- Native digits (`०१२...` for Devanagari, etc.) — used in normal
  text or only ceremonial?
- Cross-script visual overlap — many Indic scripts share visual
  analogues with different codepoints.

**Resources**:
- CLDR per locale.
- **Unicode Standard chapter 12 "South Asian Scripts"** — definitive.
- **Per-script Wikipedia pages** — `Devanagari`, `Bengali script`,
  `Tamil script`, etc. — each has authoritative tables.
- **W3C Indic Layout Requirements** documents (one per script) —
  describes orthographic rules in implementation detail.

### Hebrew (he)

**Specific questions**:
- 22 standard letters + 5 final forms (`ך ם ן ף ץ`).
- Niqqud (vowel points, ~13 codepoints in U+05B0-05C7) — religious
  texts, dictionaries, children's books. In modern news / web text,
  mostly absent. Decide whether to include.
- Cantillation marks (te'amim) — appear in Torah text. Likely out
  of scope for the modern locale, but might appear in religious-
  text tokens.
- Hebrew presentation forms (U+FB1D-FB4F).

**Resources**:
- CLDR `he`.
- **Wikipedia "Hebrew alphabet", "Niqqud"**.
- **Academy of the Hebrew Language** — definitive authority.

### Thai (th)

**Specific questions**:
- 44 consonants + ~28 vowel signs / marks + 4 tone marks.
- Vowels appear before, after, above, or below consonants — each is
  a distinct codepoint.
- No spaces between words (tokenization is downstream).

**Resources**:
- CLDR `th`.
- **Wikipedia "Thai script", "Thai alphabet"**.
- **Royal Institute of Thailand** standards.

### Myanmar / Burmese (my)

**Specific questions**:
- Complex tone / medial / final / asat marks.
- **Zawgyi vs Unicode encodings** — older Burmese web text used a
  non-Unicode Zawgyi-1 encoding. Apertus might have either. Worth
  checking the actual vocab.

**Resources**:
- CLDR `my`.
- **Unicode Standard chapter 16** for Myanmar.
- **Wikipedia "Burmese script"**.

### Armenian (hy) and Georgian (ka)

**Specific questions**:
- **Armenian**: 39 letters modern (plus 2 added in 20th century);
  punctuation Armenian-specific (`։` full stop, `՝ ՛` emphasis).
  Historical letters?
- **Georgian**: 33 letters in modern Mkhedruli. **Mtavruli** is a
  recent (Unicode 11, 2018) uppercase variant — these are SEPARATE
  codepoints from Mkhedruli. Check whether they appear in Apertus
  vocab and whether they should carry the `ka` bit.

**Resources**:
- CLDR for both.
- **Wikipedia "Armenian alphabet", "Georgian scripts"**.
- **Unicode Standard chapters 7 / 11** for Armenian / Georgian.

## Storage / schema (v4)

`char_language_bitmask.parquet` adds two columns; existing schema
stays:

| column | type | meaning |
|---|---|---|
| `codepoint` | uint32 | unchanged |
| `char` | string | unchanged |
| `category` | string | unchanged |
| `script_bits` | binary(16) | NEW — 22 bits used |
| `family_bits` | binary(16) | NEW — 31 bits used (includes single-locale families for symmetry) |
| `bitmask` | binary(16) | unchanged — language-level, 55 bits used |
| `num_langs` | uint8 | unchanged |

`token_language_bitmask.parquet` adds AND/OR pairs at each level;
existing pair stays:

| column | type | meaning |
|---|---|---|
| `token_id`, `token_bytes`, `decoded_text`, `num_chars`, `status` | existing |
| `script_and`, `script_or` | binary(16) | NEW — 22 bits used |
| `family_and`, `family_or` | binary(16) | NEW — 31 bits used |
| `bitmask_and`, `bitmask_or` | binary(16) | unchanged — 55 bits used |

**Uniform `binary(16)` across every mask column.** Script fits in 4
bytes today and family in 4 bytes today, but we store both at the
same width as the language mask. Three reasons: (1) one decode rule
for every mask column — consumers write `int.from_bytes(b, "little")`
once and use it everywhere; (2) the storage cost is negligible
(~12 bytes per row × ~95K rows = ~1.1 MB total across both parquets,
irrelevant against compute cost); (3) wire-format headroom for
future audit-driven additions, especially at the family layer where
new languages routinely arrive.

Manifests bump `schema_version: 4`. Backwards compat: old consumers
reading `bitmask` / `bitmask_and` / `bitmask_or` continue to work.

## Build pipeline changes

1. New `families.yaml` — declarative family ↔ language mapping.
   Single source of truth; validated at build (each in-scope locale
   in exactly one family; family member lists match `languages.yaml`).
2. `build_char_language_bitmask.py`:
   - After computing the language-level `cp_bitmask`, derive
     `cp_script_bits` and `cp_family_bits` per codepoint.
   - Write the two new columns + bump manifest schema.
3. `apply_to_apertus_vocab.py`: aggregate AND/OR at all three levels;
   write six bitmask columns.
4. `validate.py`:
   - Phase 1 (char): new assertions per level. e.g. `ñ` family =
     `{Romance-Latn}`, `ß` family = `{Germanic-Latn}`, `ł` =
     `{Slavic-Latn}`, polytonic `ἀ` family = `{Grek-polyton}`,
     `中` script = `{Hans, Hant, Jpan}`. Substrate digits / punct
     have ALL script bits set.
   - Phase 1 (derivation consistency): for every codepoint,
     `family_bits` is exactly the projection of `bitmask` onto each
     family's locale set. Direct equality assertion — catches drift.
   - Phase 2 (token): mask-type checks at all three levels; the
     recomputation gate runs at all three.
5. `scripts/_common.py`: factor the family / script derivation
   helpers here so build / apply / validate share them.
6. `query_codepoint.py`: returns all three masks; CLI prints them.
7. PLAN.md → renamed PLAN_v2.md; this file becomes the canonical
   plan once approved.
8. `notes/<script>.md` per-script research notes file added as the
   research from § "Per-script research plan" is done. One file per
   script; checklists + decisions made.

## What we deliberately *don't* change

- Strict-rule guarantee. Bits at every level come from positive
  evidence (or its closures) + the documented substrate rule.
- The Apertus-as-proxy assumption for Mistral-Nemo's coverage.
- The set of 55 in-scope locales (`languages.yaml`).
- Existing `bitmask` column semantics.

## Resolved decisions

1. **Family granularity for single-locale scripts**: **include them
   for symmetry.** 31 family bits total. Every codepoint has an
   answer at every level; consumers never have to special-case
   missing family slots. The 14 single-locale family bits are
   deterministic echoes of their script + language bits, accepted
   as a small redundancy cost in exchange for layer symmetry.
2. **Per-script research scope** — the active research scope is
   **European** scripts + a **special deep-dive set**:
   - **European** (broad): every European-language locale in our
     scope. Concretely:
     - All 5 Cyrillic locales (ru, uk, bg, mk, sr-Cyrl).
     - Both Greek encodings (el, el-polyton).
     - The European-and-adjacent Latin locales: en, cs, da, de,
       es, fr, hu, it, nl, pl, pt, sv, ro, sr-Latn, fi, nb, sl, hr,
       sk, et, lt, lv, ca, is, tr, az (26 locales).
   - **Special deep-dive set**: Greek (both encodings, already in
     European), Russian (already in European Cyrillic), **Hebrew
     (he), Korean (ko), Chinese (zh-Hans + zh-Hant)**.
   - **Deferred** (not in active research scope, kept on bits but
     no per-script research notes yet): Latin non-European (id, vi),
     Arabic-script (ar, fa, ur), Japanese (ja), Hindi (hi), Indic
     family (bn, ta, te, kn, ml, gu, pa), Thai (th), Burmese (my),
     Armenian (hy), Georgian (ka).

## Remaining open questions

1. **Renames**: rename `bitmask` → `language_bits` for naming
   consistency across the three levels? Bigger downstream break;
   keep as-is is the safer call.
2. **Token-level best-guess classifier**: confirmed out of scope
   for this plan (separate decision procedure as discussed). The
   three-level mask gives the classifier rich inputs; the
   classifier itself lives elsewhere.

## Estimated work

- `families.yaml` + family taxonomy: 0.5 h
- Build script changes (derivation, new columns): 2 h
- Apply script (per-level aggregation): 1 h
- Validate (per-level char + token assertions): 2 h
- Doc sweep + per-script `notes/<script>.md` skeletons: 1 h
- Rebuild + audit: 0.5 h
- **Per-script research** (cross-reference each script's resources,
  decide on auxiliary chars to re-admit): variable — 1–3 h per
  script depending on depth.

Implementation work ~7 h; research work is open-ended and is the
main cost. Worth deciding research depth before committing.

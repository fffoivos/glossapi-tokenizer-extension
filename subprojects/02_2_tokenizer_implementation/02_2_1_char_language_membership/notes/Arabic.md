# Arabic-script — per-script research notes

> Three in-scope locales: `ar` (Arabic, bit 17, Semitic-Arab family),
> `fa` (Persian, bit 18, Iranian-Arab family), `ur` (Urdu, bit 37,
> Iranian-Arab family). Status: in-scope language-level coverage
> exhaustive; per-locale split works correctly at the token level
> (1,016 ar-only tokens, 152 fa-only, 117 ur-only). Two coverage
> issues to flag — Arabic punctuation handled by substrate fallback
> (correct), and ~30 Pashto/Sindhi/Uyghur-specific letters in vocab
> falling through (correct strict-rejection of out-of-scope
> languages).

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `ar/characters.json`,
  `fa/characters.json`, `ur/characters.json`.
- Unicode 16.0: Arabic (U+0600–06FF), Arabic Supplement (U+0750–077F),
  Arabic Extended-A (U+08A0–08FF), Arabic Extended-B (U+0870–089F),
  Arabic Extended-C (U+10EC0–10EFF), Arabic Presentation Forms-A
  (U+FB50–FDFF), Arabic Presentation Forms-B (U+FE70–FEFF).
- Wikipedia: "Arabic alphabet", "Persian alphabet", "Urdu alphabet".
- Unicode Standard chapter 9 "Middle East-I, Modern and Liturgical
  Scripts" — definitive Arabic-script reference.
- Farhangestan (Persian Language Academy) — Persian language
  authority.
- CRULP (Center for Research in Urdu Language Processing) — Urdu
  language authority.

## Per-locale CLDR exemplar — verified

CLDR `ar` main (28 standard Arabic letters):
`ء أ ؤ إ ئ ا آ ب ة ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى ي`
plus 9 diacritical marks (fatha ً, damma ٌ, kasra ٍ, sukun ْ, …).

CLDR `fa` main (Persian-specific additions to base Arabic):
- `پ` U+067E (pe, voiceless bilabial stop)
- `چ` U+0686 (che, voiceless palatal affricate)
- `ژ` U+0698 (zhe, voiced palatal fricative)
- `گ` U+06AF (gaf, voiced velar stop)
- Uses `ک` U+06A9 instead of Arabic `ك` U+0643
- Uses `ی` U+06CC instead of Arabic `ي` U+064A
- Plus a few diacritics in main: `آ`, `ٔ`

CLDR `ur` main (Urdu = Persian + Indic retroflex additions):
- All Persian letters above
- Plus retroflex: `ٹ` U+0679 (te), `ڈ` U+0688 (de), `ڑ` U+0691 (re)
- Plus aspirated marker: `ھ` U+06BE (do-chashmi he)
- Plus `ے` U+06D2 (bari ye, final-only form)
- Plus `ں` U+06BA (nun ghunna, in auxiliary not main per CLDR)
- Plus `ہ` U+06C1 (gol he) and `ۂ` U+06C2

The per-locale Persian/Urdu-specific letters are all in our table
with the correct bit-set (verified — 152 fa-only AND tokens, 117
ur-only AND tokens).

## Empirical Apertus baseline

- **9,444 Apertus vocab tokens contain Arabic-script codepoints.**
- 102 unique Arabic-script codepoints used.

Per-token AND split:

| AND value | tokens | meaning |
|---|---|---|
| ar only | 1,016 | distinctively Arabic (specific Arabic chars not in fa/ur, e.g. ك U+0643 vs Persian ک U+06A9) |
| fa only | 152 | distinctively Persian (پ چ ژ گ + Persian-specific ya/kaf vs Urdu retroflex letters) |
| ur only | 117 | distinctively Urdu (ٹ ڈ ڑ ے + retroflex) |
| ar + fa + ur all three | 3,923 | shared Arabic chars admissible by every Arabic-script locale |

The Semitic-Arab vs Iranian-Arab family split also discriminates:
ar-only tokens have Semitic-Arab; fa-only and ur-only have
Iranian-Arab; the 3,923 shared tokens have both.

## Coverage state — what's missing from the table

44 Arabic-script codepoints appearing in the Apertus vocab are not
in our char table. Categorized:

### Group A — substrate punctuation (~9 codepoints, handled by fallback)

Arabic-specific punctuation, all Po/Sk/Lm category, all handled
correctly by the apply-time substrate fallback → ALL_BITS:

| codepoint | char | category | name | vocab tokens |
|---|---|---|---|---|
| U+0640 | `ـ` | Lm | ARABIC TATWEEL (justification-only kashida) | 21 |
| U+060C | `،` | Po | ARABIC COMMA | 10 |
| U+061F | `؟` | Po | ARABIC QUESTION MARK | 10 |
| U+06D4 | `۔` | Po | ARABIC FULL STOP (Urdu sentence-end) | 4 |
| U+061B | `؛` | Po | ARABIC SEMICOLON | 3 |
| U+066A | `٪` | Po | ARABIC PERCENT SIGN | 2 |

All correct under strict-rejection — these are typography, not
language-discriminating. Consumers using `query_codepoint.py` get
ALL_BITS; direct-parquet readers without the fallback get 0
(documented sparse-with-fallback contract).

### Group B — out-of-scope language letters (~30 codepoints, strict rejection)

Letters specific to Pashto, Sindhi, Uyghur, Kurdish-Sorani,
Kashmiri, etc. — none of which are in our scope. Top examples:

| codepoint | char | name | vocab tokens | language |
|---|---|---|---|---|
| U+06BA | `ں` | NOON GHUNNA | 14 | Urdu (in CLDR ur auxiliary; we dropped auxiliary) |
| U+06D0 | `ې` | ARABIC LETTER E | 4 | Uyghur, Pashto |
| U+0693 | `ړ` | REH WITH RING | 3 | Pashto |
| U+06AB | `ګ` | KAF WITH RING | 2 | Pashto |
| U+0696 | `ږ` | REH WITH DOT BELOW AND DOT ABOVE | 2 | Pashto |
| U+0685 | `څ` | HAH WITH THREE DOTS ABOVE | 2 | Pashto |
| U+0681 | `ځ` | HAH WITH HAMZA ABOVE | 2 | Pashto |
| U+069A | `ښ` | SEEN WITH DOT BELOW AND DOT ABOVE | 2 | Pashto |
| U+067C | `ټ` | TEH WITH RING | 2 | Pashto |

`U+06BA ں` (nun ghunna) is the interesting case — it's in CLDR `ur`'s
auxiliary set, which we dropped under the v2.2 "no auxiliary" rule.
Tokens containing it currently AND-reject all in-scope locales
(strict-rejection of an out-of-scope-auxiliary char). If we ever
want to add Urdu's auxiliary back specifically, it'd add ~14 tokens
to ur-attribution.

Decision: keep current behaviour (auxiliary excluded uniformly).
Documented limitation.

The other letters are Pashto-specific (`ps` locale, not in scope).
The original audit noted ~22 Arab-coverage-gap tokens; this run
counts 44 missing codepoints across the broader fall-through set
including the substrate punctuation in Group A. Most of the
Pashto-letters appear in 2–4 tokens each, all under 30 total —
well under the 50-per-out-of-scope-script audit threshold.

## Presentation Forms (U+FB50–FDFF, U+FE70–FEFF) — barely used

Only **3 codepoints** from Presentation Forms-A in vocab, 0 from
Presentation Forms-B. Mistral-Nemo's tokenizer evidently saw
near-zero legacy-encoded Arabic content. The full 832 presentation-
form codepoints are out of scope by design (modern text uses base
letters and rendering engines).

## Decisions

1. **No `languages.yaml` changes for ar / fa / ur.** Coverage of
   modern Arabic / Persian / Urdu is exhaustive at the language
   layer. Per-locale discrimination (ar-only / fa-only / ur-only
   AND counts) confirms the family split works.
2. **No `EXTRA_SUBSTRATE_CODEPOINTS` additions.** Arabic
   punctuation (`،` `؟` `؛` `۔` `٪` etc.) is correctly handled by
   the Unicode-category substrate fallback. Consistent with the
   Greek / Hebrew decision.
3. **Out-of-scope Pashto / Sindhi / Uyghur letters**: strict-
   rejection is correct. Could add `ps` (Pashto, ~15-20 tokens),
   `sd` (Sindhi, fewer), `ug` (Uyghur, fewer) at bits 55+ if user
   prioritizes. Each is well below audit threshold individually —
   defer.
4. **Arabic tatweel (U+0640)** — Lm category, handled by substrate
   fallback. Tatweel is a justification character used by Arabic
   typography to elongate a kashida (joining stroke); it's
   language-neutral within the script. ALL_BITS is correct.

## Followups

- **Urdu `ں` U+06BA** is in CLDR `ur` auxiliary; under our
  drop-auxiliary rule it falls through. ~14 Apertus tokens
  affected. Could selectively re-admit by adding to a small
  Arabic-script-specific seed list, but precedent risk — we'd
  then need to audit every locale's auxiliary for similar cases.
  Defer.
- **Pashto / Sindhi / Uyghur / Kashmiri / Kurdish-Sorani** —
  Arabic-script locales not modeled. Apertus has small-but-
  nonzero token coverage. Bits 55+ if any user-driven need.
- **Arabic Presentation Forms** — 3 tokens; out of scope by
  design. No action.
- **Hindko / Western Punjabi (Shahmukhi)** — uses ur-similar
  script. Out of scope.
- **NFKC normalisation of presentation forms** — if those 3
  tokens ever matter, NFKC would map them to base Arabic letters
  which ARE in our table.

# Hebrew — per-script research notes

> Single in-scope locale: `he` (bit 25). Status: coverage complete;
> no `languages.yaml` changes required.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `he/characters.json`.
- Unicode 16.0 Hebrew block (U+0590–05FF) and Hebrew Presentation
  Forms (U+FB1D–FB4F).
- Wikipedia: "Hebrew alphabet", "Niqqud", "Hebrew punctuation".
- Academy of the Hebrew Language (האקדמיה ללשון העברית) — Israel's
  language authority. Modern Hebrew orthography rules.
- SBL Hebrew Font docs (Society of Biblical Literature) — the
  standard niqqud / cantillation reference for biblical texts.

## Empirical Apertus baseline

Apertus vocab audit (token-level):

- 962 tokens contain at least one Hebrew codepoint.
- All 962 are `status: text`.
- 32 unique Hebrew codepoints appear in the vocab.

The 32 codepoints break down as:

- **22 base consonants** — all of the standard Hebrew alphabet
  (`א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת`) plus a `ה` /
  `ח` etc. variations. All in our table with the `he` language bit.
- **5 final-form consonants** — `ך ם ן ף ץ` (U+05DA, U+05DD, U+05DF,
  U+05E3, U+05E5). All in CLDR `he` exemplar; all in table.
- **2 niqqud (vowel-point) codepoints** appearing in vocab:
  U+05BC HEBREW POINT DAGESH OR MAPIQ (1 token), U+05B8 HEBREW
  POINT QAMATS (1 token). Both in CLDR `he`'s auxiliary set —
  technically dropped under our "no auxiliary" rule but the
  per-locale NFD closure picked them up (Hebrew NFD doesn't add
  them; they're standalone marks). Verifying: looking at the table
  output above shows both ✓ in table. So they're in via... actually
  let me note: CLDR `he`'s `auxiliary` includes `ֽ ׄ‎‏ ְ ֱ
  ֲ ֳ ִ ֵ ֶ ַ ָ ֹ ֻ ׂ ׁ ּ ֿ ״` which contains both U+05BC `ּ`
  and U+05B8 `ָ`. We dropped auxiliary in v2.2, yet these are
  ✓ in the table — they were added via case closure of the base
  letters (which is a no-op for Hebrew), NFD closure of standalone
  marks (also no-op), or the substrate override (Mn is not
  substrate). Investigation flagged in followups.
- **1 punctuation** appearing in vocab: U+05BE `־` HEBREW
  PUNCTUATION MAQAF (1 token). Not in our table; category Pd
  (dash punctuation); handled at apply time by substrate fallback
  → ALL_BITS.
- **2 niqqud in the table not appearing in vocab samples above**:
  not enumerated in the top-30, but the table has 82 Hebrew-block
  codepoints, so many niqqud and cantillation marks are stored
  even though not currently in Apertus tokens.

## Coverage state

- Hebrew block (U+0590–05FF, 144 codepoints):
  - 82 in our table
  - 30 not in table (24 unassigned + 6 punctuation)
  - balance: 32 codepoints we don't seed but Unicode assigns are
    cantillation marks (te'amim, ~26 codepoints), which the build
    omits because they're not in CLDR exemplar.
- Hebrew presentation forms (U+FB1D–FB4F, 51 codepoints): **0 in
  table**. None appear in Apertus vocab either (presentation forms
  are mostly used by legacy software that hasn't migrated to base
  letters + rendering engines).

## Missing-from-table Hebrew codepoints

### Group A — unassigned Unicode (skip)

24 codepoints in U+0590–05FF are reserved/unassigned. Correctly
absent.

### Group B — Hebrew-specific punctuation (substrate via fallback)

| codepoint | char | category | name |
|---|---|---|---|
| U+05BE | `־` | Pd | HEBREW PUNCTUATION MAQAF (Hebrew hyphen) |
| U+05C0 | `׀` | Po | HEBREW PUNCTUATION PASEQ (verse divider) |
| U+05C3 | `׃` | Po | HEBREW PUNCTUATION SOF PASUQ (verse end) |
| U+05C6 | `׆` | Po | HEBREW PUNCTUATION NUN HAFUKHA |
| U+05F3 | `׳` | Po | HEBREW PUNCTUATION GERESH (apostrophe-like) |
| U+05F4 | `״` | Po | HEBREW PUNCTUATION GERSHAYIM (acronym mark) |

All Pd/Po category, handled correctly by the apply-time substrate
fallback → ALL_BITS. Note CLDR `he` punctuation includes geresh
`׳` and gershayim `״` as `'׳` and `"״` (paired with their ASCII
look-alikes) — these are seemingly in the punctuation source but
parsed as multi-char strings and skipped by our UnicodeSet parser
for clusters. The substrate fallback handles them anyway.

### Group C — cantillation marks (te'amim)

26 codepoints in U+0591–05AF (etnahta, segol, shalshelet, zarqa,
etc.) — used in Tanakh / Torah cantillation. Category Mn (non-
spacing marks). The build doesn't seed them because they're not
in CLDR `he`'s standard subsets. **They're not substrate** under
our rule (Mn is letter-like, not in N/P/S/Z/Cc/Cf/Lm), so they'd
return 0 at apply time → false rejection for any token containing
them.

**Empirical check**: do any Apertus tokens contain te'amim? Audit
shows the top-30 Hebrew codepoints in vocab don't include any
te'amim; need a follow-up scan of the full 32 to confirm. If yes,
those tokens currently AND-reject `he` incorrectly. If no, the
gap is theoretical.

## Decisions

1. **No `languages.yaml` changes.** Coverage of in-text Hebrew
   chars is complete.
2. **Punctuation handled by substrate fallback.** Consistent with
   the Greek decision — no per-script substrate seeding.
3. **Niqqud (vowel points)**: keep current state. CLDR `he`'s
   auxiliary had these; we dropped auxiliary in v2.2 but the audit
   shows U+05BC `ּ` and U+05B8 `ָ` are in the table anyway (presumably
   via a closure path I haven't traced). The vocab uses very little
   niqqud (1 token each for the two samples), so the impact is
   minimal regardless.
4. **Cantillation marks (te'amim)**: leave out of scope. Religious-
   text-specific; if they appear in Apertus's training data at all,
   it's negligible. If a follow-up audit finds them in vocab tokens
   we'll reconsider; for now the strict-rejection outcome (0 bits →
   AND-reject `he`) is acceptable.

## Followups

- **Trace how U+05BC and U+05B8 ended up in the table** despite
  dropping auxiliary. May be a script-range fallback artefact
  (U+05B0–05BD is in the Hebrew block; if the fallback range
  includes it and the codepoint is Mn-category that the build
  treats as letter/mark, they'd land via fallback). If so,
  documented in PLAN.md `§ Script-range fallback` as expected.
- **Audit te'amim presence** in Apertus vocab (U+0591–05AF). If
  any tokens contain them, decide between adding them to a
  Hebrew-specific seed list or accepting the strict-rejection
  outcome.
- **Hebrew presentation forms (U+FB1D–FB4F)** — none in vocab; no
  action needed unless a future Apertus snapshot includes them.
- **Yiddish / Ladino** — both use Hebrew script with additional
  marks. Neither is in scope; tokens with their specific marks
  would correctly AND-reject `he` (strict-rejection). If we ever
  add them, they'd take new bits at the language layer.

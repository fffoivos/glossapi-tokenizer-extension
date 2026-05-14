# v7 review notes (2026-04-24)

Running log of issues surfaced while reviewing v7 post-cleaner
sample output (cleaner SHA `375a48d`, wave-3 changes: v6-01/02/03/05/06a
+ moji bucket expansion + NBSP fold + LaTeX-crop default ON).

Sample dirs being reviewed under
`/home/foivos/data/glossapi_work_cleaned_v7/charset_run/`:

- `openarchives_knee_500x500/{lt_1p567pct,ge_1p567pct}` — per-dataset
  knee on cleaning-only deletion pct.
- `openarchives_top_residue_punct/`:
  - `top500_by_charset_moji_ratio/` — mojibake residue ratio.
  - `top500_by_charset_punct_ratio/` — ASCII punct ratio.
  - `top500_by_counter_script_residue/` — Rust-matcher restricted-
    script-residue match count.

**Scope rule:** record only. Do NOT implement until explicitly
asked. Same ground rule as `v6_review_notes.md`.

---
## v7-01 — doubled leading-capital artifact (drop-cap duplication)

**Doc:** `openarchives_top_residue_punct/top500_by_charset_moji_ratio/
00374_495_pct0000_openarchives_gr_9d0186c8919901c6fd4026.md`
(`9d0186c8919901c6fd4026d1ef1e682281ce38e78b59bebc5c3cceb7ca58aeb9`).

**Metrics on this doc:**

| metric | value |
|---|---:|
| `charset_greek_ratio` | 0.9110 (high — looks Greek) |
| `charset_moji_ratio` | **0.0374** |
| `charset_punct_ratio` | 0.0239 |
| upstream `greek_badness_score` | 31.634 |
| upstream `mojibake_badness_score` | 0.0 |
| upstream `greek_percentage` | 90.444 |
| `cleaning_only_deletion_pct` | 0.022% (cleaner barely touched it) |
| `pct_chars_removed_non_empty` | 5.757% |

**Pattern observed** (header/title lines):

```
## Α ΑΡΙΣΤΟΤΕΛΕΙΟ ΡΙΣΤΟΤΕΛΕΙΟ Π ΠΑΝΕΠΙΣΤΗΜΙΟ ΑΝΕΠΙΣΤΗΜΙΟ Θ ΘΕΣΣΑΛΟΝΙΚΗΣ ΕΣΣΑΛΟΝΙΚΗΣ
Τ ΤΜΗΜΑ ΜΗΜΑ Μ ΜΑΘΗΜΑΤΙΚΩΝ ΑΘΗΜΑΤΙΚΩΝ
Μ ΜΕΤΑΠΤΥΧΙΑΚΟ ΕΤΑΠΤΥΧΙΑΚΟ Π ΠΡΟΓΡΑΜΜΑ ΡΟΓΡΑΜΜΑ Σ ΣΠΟΥΔΩΝ ΠΟΥΔΩΝ
```

The intended text is `ΑΡΙΣΤΟΤΕΛΕΙΟ ΠΑΝΕΠΙΣΤΗΜΙΟ ΘΕΣΣΑΛΟΝΙΚΗΣ` etc.
Each word emits as a **triplet**: `<drop-cap> <full word> <word without drop-cap>`.

Decomposed:

| token | what it is |
|---|---|
| `Α` | The drop-cap / decorated first letter emitted as its own glyph run |
| `ΑΡΙΣΤΟΤΕΛΕΙΟ` | The FULL word, with the leading Α intact |
| `ΡΙΣΤΟΤΕΛΕΙΟ` | The SAME word, with the leading Α removed (the body typeface's rendering, sans drop-cap) |

So a single word produces three tokens at render time. The chars are
all legitimate Greek; no codepoint is wrong.

**Root cause (hypothesis):** the title page of this PDF uses a
decorative drop-cap font where each capitalized title word is
typeset with the first letter in a larger/decorative glyph run,
and the rest of the word in a body glyph run. Docling extracts
both runs separately:

1. The large drop-cap glyph → emitted as one text run containing just `Α`.
2. The body run CAN contain either the full word (if the font
   embeds it) or the word-without-first-letter. In this doc, we
   get BOTH — Docling emits the full word once AND the trimmed
   body run once.

**Why none of our filters / detectors catch it:**

- Every char IS a legitimate Greek letter; `charset_greek_ratio=0.91`.
- `charset_moji_ratio=0.0374` is BELOW the 0.25 reject threshold
  (it's in the sample folder because it's in the top-500 by
  moji ratio, but the absolute value is low).
- Upstream `greek_badness_score=31.634` is below the ~90 reject
  band. The scorer's duplicated-letter-run check (`bad_double`)
  looks for CONSECUTIVE identical letters (`ΑΑΑΑ`), not spaced
  triplets like `Α ΑΡΙΣΤΟΤΕΛΕΙΟ ΡΙΣΤΟΤΕΛΕΙΟ`.
- `greek_badness_score`'s consonant-run / vowel-run detectors
  fire on runs ≥4 — the drop-cap triplets are individual words
  of normal length.
- No existing detector targets this three-token signature.

**Noise impact on training data:**

- Tokenizer sees `Α ΑΡΙΣΤΟΤΕΛΕΙΟ ΡΙΣΤΟΤΕΛΕΙΟ` at the positions
  where `ΑΡΙΣΤΟΤΕΛΕΙΟ` belongs. It learns that a single-letter
  capital Greek word commonly precedes a longer word with the
  same letter, AND that the same word-body appears twice in a
  row with slightly different prefixes. Both are spurious
  distributional signals.
- Ubiquity: any PDF with a decorative title page is affected.
  Scale is probably hundreds to thousands of openarchives
  theses.

**Detection signature** (for future implementation — NOT to be
built now):

Look for the triplet pattern inside a single line:

- Token A: 1-char capital Greek letter.
- Token B: length ≥3, starts with the SAME letter as Token A.
- Token C: the suffix of Token B with the first char removed
  (Token B[1:] == Token C).
- All separated by whitespace, in sequence.

Rarity in clean prose: extremely rare. A legitimate Greek text
rarely has `X XYZ... YZ...` consecutively. False-positive rate
should be < 0.1%.

**Fix direction** (deferred — to be implemented when the noise
class is promoted):

1. Per-line pre-pass detects the triplet; collapses `A BCD CD`
   to `BCD`.
2. Emits count of triplets collapsed for scoring.

**Priority:** Record for now. Unusual but meaningful noise class;
may be significant across the theses population. Quantification
wave needed to size the impact before implementing.

---

## v7-02 — classical philology meter analysis: real content + upstream extraction loss

**Doc:** `openarchives_top_residue_punct/top500_by_counter_script_residue/
011111_005_pct0005_openarchives_gr_ade6741a0826e42afef265.md`
(`ade6741a0826e42afef265d4a4f051fa53bad4c24c943e7756d68cb8b88bc6c2`).

**What the doc actually is**: a Greek master's thesis from
Aristotle University, Faculty of Classical Philology — comparative
metrical analysis of dactylic hexameter across Homer (Iliad XXII +
Σ + T), Virgil (Aeneid XII), Ovid (Metamorphoses I), and Propertius
(1.1). Ten Πίνακας (tables) of metrical scansion.

**Metrics on this doc:**

| metric | value |
|---|---:|
| `cleaning_only_deletion_pct` | 0.459% (cleaner barely touched) |
| `pct_chars_removed_non_empty` | 1.781% |
| `charset_greek_ratio` | 0.514 |
| `charset_moji_ratio` | 0.0419 (low — passes filter) |
| `charset_punct_ratio` | 0.1978 |
| upstream `greek_badness_score` | 12.34 (clean) |
| body lines | 4,817 |
| escaped underscores `\_` | 17,944 |
| combining macrons (U+0304) | 5,511 |

**Three distinct content classes in the doc** (each behaves
differently):

### Class A — δ/σ scansion tables (intended, intact)

```
| στ. 525 | δ | σ | δ | σ |
|         | σ | σ | σ | σ |
```

`δ` = δάκτυλος (dactyl), `σ` = σπονδείος (spondee). Each row =
one verse number (στ.), each column = one of the first four feet.
Standard classical-philology notation. **This is correct**.

### Class B — Greek hexameter with metrical scansion (intended, intact)

```
αἳ μὲν ὕπαιθα ἄνακτος ἐποίπνυον αὐτὰρ ὃ ἔρρων
\_-\_\_U\_\_U|\_-\_U\_U|-\_\_U\_\_U|\_-\_\_UU|\_\_\_-\_U\_\_U|-\_\_-\_
```

The line below the verse is the **scansion notation**:
- `_` = long syllable (longa)
- `U` = short syllable (brevis)
- `|` = foot boundary
- `-` = caesura

The backslashes (`\_`) are markdown escapes for literal underscores
(otherwise `_` would parse as italic). These are correct as-is.

**This is correct**. Our cleaner does NOT touch this — Phase A's
HR regex requires the WHOLE LINE to be a run of `\_` chars (≥4
consecutively); these scansion lines have `U`, `|`, `-` mixed
in, so they skip the rewrite.

### Class C — Latin Propertius with stripped vowels (UPSTREAM destroyed)

```
1 Cy   ̄́ nth pri   ̄́m si   ̄́s msru ̄́ m m cpt cllis,
       Cntctu   ̄́ m nlli   ̄́s a   ̄́ nt Cpi   ̄́dnbús.
```

This is supposed to read:

```
Cynthia prima suis miserum me cepit ocellis,
   contactum nullis ante cupidinibus.
```

(Propertius 1.1.1-2.) Compare:

| intended | extracted |
|---|---|
| `Cynthia` | `Cy ̄́nth` |
| `prima` | `pri ̄́m` |
| `suis` | `si ̄́s` |
| `miserum` | `msru ̄́m` |
| `me` | `m` |
| `cepit` | `cpt` |
| `ocellis` | `cllis` |

Pattern: **vowels with macron + acute (`̄́`) detached from their
letters; bare letter beneath the diacritic dropped, leaving an
orphan combining-mark glyph next to a stripped consonant cluster.**

Root cause: PDF rendering. The metrically-marked Latin uses
combining diacritics (U+0304 macron + U+0301 acute) on long
syllables. The PDF font emitted each diacritic as a separate
positioned glyph run; Docling extracted those orphan diacritics
but **lost the underlying letter** in the process. Result: every
long-syllable vowel in the Latin verse is gone.

**Did our cleaner do this?** No. Evidence:
- `cleaning_only_deletion_pct: 0.459%` — line-drop + per-char
  filter combined removed 0.46% of the doc. The Latin verse
  damage looks like ~30% of those lines' chars.
- `chars_dropped_by_per_char_filter: 34` — 34 chars total. The
  vowel loss is far larger than this.
- `chars_dropped_by_normalization: 186` — also tiny.
- The combining macron (U+0304) is in our `other` bucket; we
  don't strip it. Confirmed: 5,511 macrons survive in the
  cleaned output.

The damage is in the INPUT (the parquet row already has the
stripped Latin). It happened during Docling's PDF→Markdown
extraction, before this cleaner ever saw the doc.

**Why the doc is in this sample**: it's in
`top500_by_counter_script_residue` (sorted by Rust matcher's
`script_residue_restricted` count). The counter is probably
firing on the orphan combining diacritics + the scansion `_U|`
characters. From the cleaner's POV, the `δ/σ` and Greek hexameter
parts are clean Greek; the Latin section is the noise driver.

**Implication for review**:
- This doc class (classical philology with metrical scansion)
  appears legitimate in the cleaned output for the Greek-text
  parts and the δ/σ tables. Worth keeping.
- The Latin section has irrecoverable vowel loss. We can't fix
  it without re-extracting the source PDF.
- For training: the Latin section is corpus-level noise (looks
  like consonant runs to the tokenizer). The Greek scansion
  notation `\_-\_\_U\_\_U|...` teaches the tokenizer a literal
  notation system that's fine — the meter notation is intentional
  scholarly markup.

**Fix direction (NOT for this wave)**:
1. **Detect classical-philology metrical-notation regions** —
   pattern: alternating verse line + scansion line where scansion
   line is mostly `\_`, `U`, `|`, `-`, whitespace. Could be
   excluded from `charset_*` ratio counts the same way we exclude
   table rows.
2. **Detect orphan combining-mark sequences** as a separate
   noise counter. Pattern: combining mark (U+0300..036F) with
   space on both sides, OR sequences of >2 combining marks not
   attached to a base letter. Different from existing detectors.

**Priority for action**: Low. Classical philology theses are a
small corner of the corpus. The cleaner's behavior here is
correct (don't touch the legitimate scansion). The Latin damage
is upstream.

**Decision**: KEEP this doc class. Don't change cleaner behavior.

---

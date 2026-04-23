# User-review notes — 2026-04-23 onward

Running log of problematic sample docs the user surfaces during the
1000-doc deletion-band review. Each entry captures:
- the specific doc
- the noise class it exposes
- whether any existing signal caught it
- a concrete handling proposal for wave 2

---

## Case 1 — `0006_openarchives_gr_098337a3a07f52084f3eabd8.md`

**Input (sample md)**: `/home/foivos/data/glossapi_work_cleaned_v3/charset_run/deletion_band_500x500/0006_openarchives_gr_098337a3a07f52084f3eabd8.md`

**Source**: openarchives.gr, doc_id `098337a3a07f52084f3eabd8a501db1beb2f45928999b1984a98e333776391a7`. A PhD dissertation on Manos Kalomoiris's song composition.

**Deletion metrics as-cleaned**:
- `pct_chars_removed_non_empty`: 0.58% (almost nothing cleaned)
- `charset_greek_ratio`: 0.82 (passes `> 0.02` cutoff)
- `charset_moji_ratio`: 0.0024 (well below 0.25)
- `charset_punct_ratio`: 0.079 (well below 0.30)
- `mojibake_noise_ratio`: 0.082 (diagnostic, would not rate-limit)
- upstream `greek_badness_score`: 15.8 (below the 90-ish mojibake-catching range)
- upstream `mojibake_badness_score`: 0.0

**Noise classes visible in the text** (both missed by every current signal):

### (A) Font-substitution mojibake — throughout headers/TOC/body captions

Greek capitals render as visually-identical Latin capitals **with spaces
inserted between letters**:

- intended: `ΑΡΙΣΤΟΤΕΛΕΙΟ ΠΑΝΕΠΙΣΤΗΜΙΟ ΘΕΣΣΑΛΟΝΙΚΗΣ`
- actual:   `API Σ TOTE Λ EIO Π ANE Π I Σ THMIO Θ E ΣΣ A Λ ONIKH Σ`

Confusable-Latin swaps observed: `A↔Α B↔Β E↔Ε H↔Η I↔Ι K↔Κ M↔Μ N↔Ν O↔Ο P↔Ρ T↔Τ X↔Χ Y↔Υ Z↔Ζ`. The "real Greek" characters
(Δ, Λ, Ξ, Π, Σ, Φ, Ψ, Ω) stay in-block, which is why
`charset_greek_ratio` still passes — about 80% of chars are
real-Greek; only the confusables got swapped.

Secondary signal: unusually many single-capital-letter runs separated
by single spaces (`A Λ ONIKH Σ Σ XO Λ H`). This is PDF-extraction
layout noise (the font embedded each glyph as a separate positioned
text run; Docling put single-space between them).

### (B) LaTeX-escape mojibake — at end of doc (index/bibliography)

```
T0 T0pxy0b0iT\nC\n\n7,7,9,1,9,8,1,10,1,10,2, 1224, 1334, 1997-2.03, 2233,
T0 Xi0vi: 24, 68, 69
T0pxy0v\delta\k\n1,3, 19, 1,3,0, 1664-1170
T0pxy0\delta\x0\alpha\v\n\t0: 24, 55, 56, 62, 73,
T0p\epsilon\alpha\p\epsilon: 24, 75, 141
T0p\epsilon\k\pi\epsilon\v\gamma\epsilon: 15, 21, 22, 28,
```

Patterns:
- Literal `\alpha \beta \gamma \delta \epsilon \kappa \pi \sigma \tau
  \eta \omega \ldots` — LaTeX math-mode Greek-letter macros that
  survived into the extracted text
- Digit-for-letter substitution: `T0` (for "Το"), `T4` (for "Τα"),
  `X0` (for "Χο"), `\i8` (probably for "ι8" or "ιβ")

**Why every current signal missed both**:

| signal | value | why it missed |
|---|---:|---|
| `charset_moji_ratio` | 0.002 | the mojibake chars ARE in the Latin block OR in the Greek block — they're not in Latin-1 Supp / IPA / PUA / Specials |
| `charset_punct_ratio` | 0.079 | `\alpha\beta` sequences use letters not punct; space-padded capitals also count as letters |
| `charset_greek_ratio` | 0.82 | plenty of real Greek survives in the body; the mojibake is concentrated in headers + appendix |
| `greek_badness_score` (upstream) | 15.8 | upstream scorer targets modern-Greek badness signatures; single-letter Latin-capital runs + LaTeX escapes are outside its pattern inventory |
| `mojibake_badness_score` (upstream) | 0.0 | upstream mojibake scorer targets byte-level encoding failures (UTF-8 → Latin-1 damage); this is extraction-layer damage, not encoding damage |

**Proposed detectors for wave 2**:

1. **LaTeX-escape density detector**. Count occurrences of
   `\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)(?:\\|\b)`
   (and capital variants). Above some density per non-whitespace char
   → flag. Trivial regex, cheap.

2. **Digit-in-Greek-word detector**. Find tokens where ASCII digits
   appear mid-word mixed with Greek/Latin letters (not surrounded by
   whitespace/separator). Specifically `[A-Za-zΑ-Ωα-ω]+[0-9]+[A-Za-zΑ-Ωα-ω]+`
   patterns like `T0p`, `T4`, `X0`, `1\n,3`. Rare in clean text
   (measurement units, model numbers, maybe 0.5%). High density = bad.

3. **Confusable-Latin-in-Greek-context detector**. Harder. Need to
   spot runs of `[ABEHIKMNOPTXYZ]` (the Greek/Latin confusables)
   interleaved with real Greek letters. Heuristic: within a sentence
   that contains `[α-ωΑ-Ω - minus confusables]` characters, count
   confusable-Latin chars that have no Greek neighbor in a window.
   Above ratio threshold → flag. This one needs more design work.

4. **Single-capital-letter-space-padded pattern**. Count runs of
   `\b[A-Z]\s` (single uppercase followed by single space). Legit
   Greek prose has this rarely (initials: `A. Π. Θ.`); mojibake docs
   have it hundreds of times. Easiest of the three signals to
   implement and likely catches both (A) AND the general PDF-space-
   padded-glyph failure mode.

**Recommendation**: (4) alone would probably catch case (A), which
is the more pervasive issue across openarchives. (1) is trivial
and catches case (B). (2) catches both subtly. All three are cheap
regex/count scans — should be added to `analyze_charset` as new
ratios in wave 2.

---

## Case 2 — `lt_20pct/0028_eurlex-greek-legis_doc_6872.md`

**Input (sample md)**: `/home/foivos/data/glossapi_work_cleaned_v3/charset_run/deletion_band_500x500/lt_20pct/0028_eurlex-greek-legis_doc_6872.md`

**Source**: eurlex-greek-legislation, doc_id `doc_6872`. EU regulation
(ECE/UN regulation 75, motorcycle tyres).

**Deletion metrics as-cleaned**: ~2.8% (low). Doc has 712 lines starting
with `|`.

**Issue**: PDF-extracted "tables" that don't render as markdown. Most
rows are empty single-cell forms:

```
|  |  |  |  |
| --- | --- | --- | --- |
| 30.3.2011 | EL | Επίσημη Εφημερίδα της Ευρωπαϊκής Ένωσης | L 84/46 |
```

or:

```
|  |  |
| --- | --- |
| 1. | Πεδίο εφαρμογής |
```

**Origin**: upstream Docling-extracts the PDF TOC / layout into markdown
table syntax. Columns mis-align, rows are mostly empty. Present BEFORE
our cleaning — the cleaner didn't break them.

**Did our cleaner break anything?** No. The `|` chars pass through
unchanged. They inflate `charset_punct_ratio` in the old (pre-2026-04-23)
definition, but as of the `is_format_scaffolding_line` update, table
rows are excluded from the ratio denominator so this is no longer a
signal-quality issue.

**Can we improve the rendering/content?**

Three options:
1. **Add `normalize_empty_table_rows` cleaner pass**: drop rows matching
   `^\|(\s*\|)+\s*$` (pipe-only rows, no content between pipes). Safe —
   removes non-rendering scaffolding without touching legit tables.
2. **Collapse single-column tables to plain text**: when a `|` row has
   only one non-empty cell, strip the `|` and the separator row. Safer
   than (1) for bibliographic lists that use `| text |` formatting.
3. **Leave as-is.** Tokenizer will learn `|` tokens anyway; doesn't
   affect downstream training quality significantly.

**Recommendation for wave 2**: add (1) as a cheap normalize pass; it's
a pure-format cleanup with zero semantic risk.

---

(future cases to be appended below)

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

## Case 3 — URLs inflate `charset_punct_ratio` (open question, 2026-04-23)

**User observation**: URLs should be excluded from the punctuation
metric. URL-rich docs (web archives, opengov, openarchives reference
sections) carry many `: / . - _ ? = & #` characters that all land in
`ascii_punct` even though they are not mojibake / font-substitution
noise — they're legitimate identifiers.

**Status**: NOT yet handled. `is_format_scaffolding_line` and
`strip_html_comments` exclude tables / separators / LaTeX block math /
HTML-comment lines, but URL spans pass through untouched. Inline `$…$`
LaTeX, triple-backtick code fences, and `\begin{…}` environments also
pass through (see "NOT excluded" list in the wave 1 sampler review).

**Provisional plan for wave 2 cleaner**: add a `strip_url_spans` helper
in `charset_module.rs`, called BEFORE the per-char count (analogous to
`strip_html_comments`). Patterns:
- `https?://\S+`
- `ftp://\S+`
- `www\.\S+`
- bare email `\S+@\S+\.\S+`

Use a stricter terminator `[^\s)>\],;]+` so trailing sentence punctuation
isn't eaten. The strip is for ratio computation only, NOT for output
text — URLs stay in the cleaned doc.

---

## Case 4 — `&amp;` literal in cleaned text (regression, 2026-04-23)

**User observation**: HTML entity `&amp;` is appearing literally in the
cleaned text. Should decode to `&` (and similarly the other common
named entities + numeric).

**Status**: NOT yet handled. The cleaner's normalize phase has an
"entity fallback" in the per-class drop attribution
(`chars_dropped_by_normalization` is supposed to include entity
collapse) but in practice `&amp;` survives — either the entity-decode
pass isn't running or its pattern set doesn't include named entities.

**Where it amplifies**: each `&amp;` contributes 4 ASCII-punct chars
(`&` + `;` × 1, plus the lowercase letters which are latin not punct
— so net +1 punct +1 entity-context — small per occurrence but
frequent in HTML-extracted sources, compounding case 3.

**Provisional plan for wave 2 cleaner**: extend the normalize pass
with a Rust equivalent of `html.unescape`:
- named: `&amp; &lt; &gt; &quot; &apos; &nbsp;`
- numeric: `&#\d+;` and `&#x[0-9a-fA-F]+;`

Apply BEFORE the per-char strip so resulting chars (e.g. `&` itself)
are subject to charset filtering normally. Add regression tests in
`charset_module.rs` mirroring the html-comment test pattern.

---

## Case 5 — TOC-as-table + single-line `$$…$$` LaTeX (architectural, 2026-04-23)

**User observation, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/lt_1p567pct/00555_pct0001_openarchives_gr_07aaa4611661bb805464b9.md`:

The doc has high `charset_punct_ratio` (0.055), but visible punct is
all in TOC dot-leaders (`....`) inside `|...|` table rows or stats
tables. User asks: do we have a malformed-table detector? Does it run
before further cleaning? The validity check should sit **before** the
table minimizer (assuming the minimizer is reliable on valid tables).

**Empirical accounting on this doc**:

| metric | value |
|---|---|
| total chars | 552,312 |
| post-line-exclusion chars (denominator) | 136,966 |
| `ascii_punct` count | 6,213 |
| `ascii_punct_ratio` | 0.055 |
| line classes: `|...|` table rows (excluded) | **1,425** |
| html-comment lines (excluded) | 113 |
| separator/dot-leader lines (excluded) | 0 |
| content lines (counted) | 789 |

→ Tables ARE being excluded (1,425 of them). The residual 5.5% punct
sits on 789 content lines and decomposes into:

1. **Single-line `$$…$$` LaTeX equations** (Docling collapses
   multi-line math into one line). Top punct-rich lines in this doc
   are `$$ … \cdot \frac{…}{…} … $$` patterns, 40-80 punct chars each,
   ×many. **`charset_module.rs:140-156` only excludes MULTI-line `$$…$$`**
   regions: tracker counts `dollar_pairs % 2 == 1` per line, so a line
   with `$$ ... $$` (two `$$` tokens, even count) yields
   `starts_or_ends_math = false` and is NOT excluded. **Bug.**
2. `&amp;` literal entities (Case 4) — small contribution but real.
3. Legitimate sentence punctuation — commas, parens, periods in long
   Greek prose lines.

**Architecture currently in place** (`rust/glossapi_rs_cleaner/src/`):

- `table_analysis_module::core_detect_malformed_tables` — scans for
  GFM tables and flags issues: header/separator column-count mismatch,
  body-row column-count mismatch, separator-without-header. Runs in
  pipeline Stage 2.
- `table_remover_module::remove_tables_from_content` — removes tables
  by line-range, given a list of `LineRange`s. Runs in pipeline Stage 3.
- `cleaning_module::core_clean_text` — per-char strip + line drop,
  runs in Stage 4 (after table removal).

**Gap user is naming**: the existing detector flags STRUCTURAL
malformations (column mismatch, missing header). It does NOT flag
**semantic** malformations:
- Single-column `|...|` wraps that are actually TOCs, not tables.
- Tables with mostly-empty cells (`|  |  |  |` rows).
- Tables where every cell is one short token (column-of-numbers stats
  tables that are real but don't render usefully).

**Provisional plan for wave 2 cleaner**:

1. **Fix the LaTeX-block bug**. In `count_charsets`: track ANY span
   between two `$$` tokens on the same line as math, not just multi-
   line blocks. Cleanest fix: `strip_inline_math(line)` ahead of
   per-char counting (mirrors `strip_html_comments` pattern). Add
   regression test `excludes_inline_double_dollar_math_from_counts`.
2. **Add `is_pseudo_table(table_lines: &[&str]) -> bool` gate** that
   sits between `core_detect_malformed_tables` and
   `remove_tables_from_content`. Returns true when:
   - All rows have 1 column (strict single-column wraps), AND
   - rows match a TOC heuristic (most rows end with `…\d+\s*` or
     contain runs of `\.{4,}`)
   OR
   - >80% of cells are empty after strip.
   When pseudo-table: strip the `|`/separator syntax and emit cell
   contents as plain prose lines. Don't fully delete (TOC info has
   value as a section list).
3. **Tighten `is_format_scaffolding_line` punct exclusion**: also
   exclude lines that match `^(.{0,160})\s*\.{4,}\s*\d+\s*$` —
   classic TOC entry — even outside `|...|` wrappers.

**Order**: gate (2) runs before the minimizer; LaTeX fix (1) and
charset tightening (3) sit in `charset_module.rs` and affect the
pre-clean ratios.

---

## Case 6 — `*[...truncated N chars...]*` is a SAMPLER display artifact, not cleaner deletion (clarification, 2026-04-23)

**User observation, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/lt_1p567pct/00560_pct0007_openarchives_gr_f4ab1064e35ccc16cbcfb4.md`:

> "we have notes of `*[...truncated 156493 chars...]*` very large n of
> chars being removed. How can that place the doc under 1.5% removals?
> what did the original doc have? it couldnt all be normalizations."

**Resolution**: the truncation marker is a **sampler display artifact**,
not cleaner deletion. `pull_deletion_band_samples.py:252-256` truncates
the rendered text to `--max-text-chars` (default 8000) by showing the
first 4000 + last 4000 chars and inserting the marker. The cleaner saw
the full doc; deletion % is computed against the full text.

**Additional bug identified**: the sampler renders the doc with header
"the cleaned doc text" (line in the rendered comment), but `_bulk_find`
reads from the **input parquet**, which is the PRE-cleaner text. So
samples are showing BEFORE-cleaner content but labelled as "cleaned".

**Wave 1 fix (in flight)**:

1. Sampler emits TWO files per pick: `<prefix>_<id>_BEFORE.md` and
   `<prefix>_<id>_AFTER.md`.
2. Both files include explicit truncation banner: "TRUNCATED for display
   to N chars; cleaner processed M chars in full."
3. AFTER file runs `cleaner.clean_text(text, scripts_to_keep)` with the
   same script set the cleaner used in the v5 run.

---

## Case 7 — `GLYPH&lt;216&gt;` bullet-marker surviving the cleaner (bug, 2026-04-23)

**User observation, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/ge_1p567pct/00249_pct0016_openarchives_gr_ef2efc052f31a835984997.md`:

Nine occurrences of `GLYPH&lt;216&gt;` survive cleaning, each at the
head of a bullet-like line, e.g.:

```
- GLYPH&lt;216&gt; Logical call source contained η οποία ...
- GLYPH&lt;216&gt; Logical call destination contained που ...
- GLYPH&lt;216&gt; Media destination είναι ο προορισμός των media ...
```

**Why this survived** — three converging gaps:

1. **HTML-entity encoding hides the angle brackets**. The matcher's
   `glyph_marker_extended_literals` literal set contains bare `GLYPH`
   (so it fires on each occurrence for counting) but the extended regex
   `/uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+` does NOT target the `GLYPH<\d+>`
   form. And even if it did, the actual on-disk form is `GLYPH&lt;…&gt;`
   — HTML-escaped angle brackets, which Case 4 (entity decode) has
   already flagged as un-decoded.
2. **Doc-level threshold**. `glyph_font_like ≥ 14` is the drop rule
   (from `thresholds_v2_calibrated.json`). This doc has 9 occurrences →
   below threshold → doc survives. The three-counter system is
   calibrated for doc-level drop, not span-level surgical removal.
3. **No span-level substitution pass exists**. The cleaner has
   per-char strip (removes single chars outside the allowed scripts)
   and line-drop (removes lines that are mostly junk). It has no
   "replace this regex span with neutral placeholder" pass for inline
   tokens like `GLYPH<216>` sitting in an otherwise-Greek line.

**Empirical on this doc**:
- non_empty_chars_in: 88,483; `chars_dropped_by_line_drop`: 1,420
  (9 lines); `chars_dropped_by_per_char_filter`: 8
- 9 `GLYPH&lt;216&gt;` spans × ~20 chars each = ~180 chars of
  bullet-marker noise surviving.
- The lines weren't candidates for line_drop because they are mostly
  legitimate Greek prose — the `GLYPH&lt;216&gt;` is just a bullet
  replacement (Docling renders the PDF's `•` glyph codepoint U+2022
  via a font-subset lookup as `GLYPH<216>`, then HTML-escapes it).

**Provisional plan for wave 2 cleaner**:

1. **Fix Case 4 first** (entity decode) — once `&lt;` / `&gt;` are
   decoded back to `<` / `>`, the token becomes a clean `GLYPH<216>`
   and regex matching is straightforward.
2. **Add a substitution pass in the Rust cleaner's normalize phase**
   that matches the regex `GLYPH<\d+>` (and, defensively, the escaped
   form `GLYPH&lt;\d+&gt;` in case entity decode is skipped) and
   rewrites to a single `•` (or strips entirely — defer to user
   review on the replacement choice: bullet vs. nothing).
3. **Generalize to similar Docling placeholders** flagged by prior
   waves: `/uniXXXX`, `/gN`, and any other "missing-glyph stand-in"
   patterns. These are all font-subset fallback markers — the
   underlying character was unmapped, and the PDF reader substituted
   the font's internal glyph ID. Substitute all of them to `•` or
   empty.
4. Accounting: these substitutions should land in
   `chars_dropped_by_normalization`, NOT per_char_filter (each is a
   multi-char pattern, not a single-char strip).
5. Regression tests in `normalize.rs` paralleling the html-comment
   tests.

**Open question for the user**: drop the token entirely, or replace
with `•`? Bullet preserves list structure for downstream tokenization;
empty string loses structure but reduces noise further.

---

## Case 8 — Soft-wrap `\n` + inter-word tabs fragment Greek prose (bug, 2026-04-23)

**User observation, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/ge_1p567pct/00264_pct0090_openarchives_gr_1749a6acae8546c7aae5ac.md`:

> "the txt has extremely short lines with 1 or 2 words, but with md
> preview it looks fine."

**Structure of the noise**: the doc contains passages like

```
το
  σύνολο
  του
  εμπορικού στόλου,
   σε
   παγκόσμια
   κλίμακα.
```

which renders fine in the markdown preview (single `\n` within a
paragraph collapses to a space) but reads as column-width-fragmented
noise as raw text / tokenizer input. Each inter-word separator is a
`\t\n<leading spaces>` sequence — Docling preserved the PDF's visual
column layout as literal whitespace. Line-length distribution for this
doc skews heavily toward < 100 chars with a long tail up to 1054.

**Why this matters for tokenizer training**: the tokenizer sees the
raw text. Each `\n` and `\t` becomes its own token, splitting what
should be a single continuous word sequence into many fragments. The
tokenizer learns `word1 <TAB> <NEWLINE> <SPACES> word2` patterns that
never appear in real Greek corpus text, wasting vocab budget and
degrading downstream generation quality.

**Why the cleaner missed it**: there's no normalization pass that
collapses soft-wrap `\n` + `\t`-between-words back into paragraph flow.
Existing normalize passes handle dot-leaders, separators, ellipses,
entities (partial — Case 4) — but NOT paragraph reflow.

**Provisional plan for wave 2 cleaner** — add two normalize passes:

1. **Intra-word whitespace collapse**: any run of `[\s]+` (including
   `\t`, `\n`, spaces, NBSP) between two non-whitespace chars on
   adjacent lines within a "paragraph block" → single space. A
   paragraph block is delimited by blank lines, `#`-headings, `|`-table
   rows, `---`-separators, `>`-blockquotes, and list markers at line
   starts. Pattern: `([^\s])[ \t]*\n[ \t]+([^\s])` → `\1 \2`.

2. **Paragraph-reflow heuristic** (safer subset of above): only join
   adjacent lines when the joined chars are both letters (no sentence
   terminator, no markdown punct). Hard-break conditions: prior line
   ends in `.?!:;·;` or Greek final-sentence punct, or next line
   starts with `#|>*-` etc.

Order: run AFTER per-char strip + AFTER entity decode (Case 4) so the
target characters are already in their final form. Account the
collapsed whitespace in `chars_dropped_by_normalization`.

Tests: pair of regression tests with the actual observed Docling
pattern (`word1\t\n  word2`) → `word1 word2`, and a negative test
preserving intentional paragraph breaks (blank line between blocks).

---

## Case 9 — Correction: `GLYPH<N>` is NOT a Docling-global code (clarification, 2026-04-23)

**User question, following Case 7**:

> "your understanding of the previous glyph made me wonder if there are
> standard ways to reconstruct other glyph structures instead of
> deleting them. My understanding was that each PDF has a different
> glyph system and that is why we cant recover them. But you are saying
> that it's docling's global system."

**The user's understanding is correct; mine in Case 7 was wrong / misleading.**

**How `GLYPH<N>` actually works**:

- PDFs embed **font subsets** with internal CID → glyph-index mappings.
- Each font subset has its own numbering. `GLYPH<216>` in PDF A and
  `GLYPH<216>` in PDF B could be totally different characters if the
  two PDFs use different font subsets.
- Docling resolves CID → Unicode using the PDF's **ToUnicode CMap**
  when present. When the CMap is missing or incomplete, Docling falls
  back to emitting the raw font-glyph index as `GLYPH<N>` or
  `/uniXXXX` (if hex-encoded) or `/gN` / `/gidN` — these are
  **un-mapped placeholders**, per-font, not global.
- The number 216 is font-local. NOT Unicode, NOT a Docling constant.

**Why some numbers appear to recur**: common Type 1 / TrueType fonts
(Times, Arial, CMU, Helvetica) use similar default encodings (Adobe
Standard Encoding or WinAnsi), so within a single-corpus of PDFs
produced by similar tools, some glyph indices DO happen to map to the
same character most of the time — e.g. `GLYPH<216>` often turns out to
be `•` (bullet U+2022) because that's glyph 216 in Adobe Standard
Encoding. But this is **statistical**, not a guarantee, and varies by
font / PDF producer / subset order.

**Standard recovery options** (ordered by practicality here):

1. **Re-extract the source PDF with a better extractor** — e.g.
   Marker, Nougat, or pdfminer.six with its ToUnicode awareness. Only
   feasible if we keep the original PDFs around. Expensive at corpus
   scale.
2. **Per-corpus statistical decoding dictionary**: tabulate every
   `GLYPH<N>` / `/uniXXXX` / `/gN` in our corpus, bucket by count and
   context (what chars surround it on its line), manually validate a
   mapping for each common value (e.g. `GLYPH<216>` → `•` after
   confirming it sits at bullet-list positions in 95% of occurrences).
   Feasible but manual — scales to maybe the top 20-30 most-common
   glyph numbers.
3. **Delete / substitute with neutral marker** (what Case 7 proposed)
   — safe default for the long tail.

**Suggested wave-2 plan**:

Combine (2) + (3). Build a dictionary pass:

```
GLYPH_MAPPING = {
  216: "•",    # empirically bullet in this corpus (VERIFY on sample)
  # add others after validation
}
```

For glyph numbers in the dictionary, substitute to the mapped char.
For glyphs not in the dictionary, delete (or replace with `•` as a
safe-assumption neutral marker — user-configurable). The dictionary
lives alongside `category_specs/` so it can evolve with the corpus.

Scope decision for the user: build the dictionary (worth the manual
effort for the most common ~20 glyphs), or skip straight to deletion
of all unmapped `GLYPH<N>` / `/uniXXXX` / `/gN`?

---

## Case 10 — Adobe Symbol font PUA chars are 100% recoverable + add numerals counter (bug + feature, 2026-04-23)

**User observation, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/ge_1p567pct/01500_pct0033_openarchives_gr_9011088e7bd361d8428875.md`:

> "unique failure mode with box-like characters and many numbers. Maybe
> also count numerals in text section; where text section = not md
> tables, separators and TOC runs, or latex, or whitespaces"

**What the "box characters" actually are**: PUA chars from the Adobe
Symbol font, preserved by the PDF extractor instead of mapped to their
real Unicode equivalents. Empirical decode of top-30 PUA chars in this
doc:

```
U+F02D (1,953×) → -       U+F03D (1,513×) → =      U+F02B (1,442×) → +
U+F06C (  489×) → λ       U+F03C (  404×) → <      U+F06D (  326×) → μ
U+F03E (  264×) → >       U+F0A3 (  245×) → ≤      U+F0B3 (  222×) → ≥
U+F028 (  112×) → (       U+F029 (  112×) → )      U+F0DE (  101×) → ⇒
U+F06E (   99×) → ν       U+F0DB (   67×) → ⇔      U+F0CE (   57×) → ∈
```

The pattern: each PUA codepoint is `0xF000 + ASCII-position-of-Symbol-
font-glyph`. So `U+F02D` = position 0x2D = `-` in Symbol encoding = the
math minus. `U+F061` = position 0x61 = `α`. **100% of the top-30 PUA
chars (7,778 total occurrences) map cleanly to Adobe Symbol encoding**
— no guessing, no per-PDF variation: the Symbol font has a
standardized encoding used by every PDF that embeds it.

This is **Case 9's (2)-option made concrete**: we have a
deterministic, corpus-independent dictionary for this class of noise.

**Context correction for Case 9**: `GLYPH<N>` is per-font (we were
right), BUT this PUA-Symbol case is its own thing — fully
standardized and recoverable because Adobe Symbol is an old pre-
Unicode "private" encoding that's globally identical across any PDF
using that font.

**Provisional plan for wave 2 cleaner**:

1. **Adobe Symbol PUA decode pass** — static dictionary of ~80
   entries covering the Greek alphabet, math operators, set-theory
   relations, and math delimiters. Runs BEFORE the per-char strip so
   the recovered Greek letters are counted as Greek and the recovered
   math symbols survive (if they're in allowed-scripts) or get logged
   as real math noise rather than unrecoverable PUA.
2. Source the dictionary from the authoritative reference: Adobe
   "Symbol Encoding Vector" (PostScript) — it's public. Should cover:
   - Greek lower (α..ω) and upper (Α..Ω)
   - Math: `+ − × ÷ = ≠ < > ≤ ≥ ≈ ≡ ∈ ∉ ⊂ ⊃ ∪ ∩`
   - Arrows: `→ ← ↑ ↓ ⇒ ⇐ ⇔`
   - Logic: `∀ ∃ ∧ ∨ ¬`
   - Operators: `∑ ∏ ∫ √ ∂ ∇ ∞`
   - Delimiters: `( ) [ ] { } ⌈ ⌉ ⌊ ⌋`
3. Tests: regression tests with real input samples from the corpus
   (this doc is a good fixture — 100% coverage known).
4. Attribute decoded chars to `chars_dropped_by_normalization` with a
   new sub-category label `pua_symbol_decode` (multi-char substitution,
   per Case 7 accounting rule).

**User's second suggestion — numerals counter**:

Add a fourth charset ratio:

```
charset_numeral_ratio = ascii_digits / (non-whitespace chars on
                                         content lines)
```

where "content lines" uses the SAME exclusion set as the existing
ratios:

- MD table rows (`|…|` with ≥2 pipes)
- Standalone separator/dot-leader lines
- HTML-comment-only lines (and inline `<!-- … -->` stripped)
- LaTeX block math `$$…$$` regions
- (to add per Cases 3, 5, 8): URL spans, inline `$…$` math, code
  fences, `\begin{…}` envs, paragraph-reflow pre-join tabs

**Why add it**: statistical tables that slip through as prose (e.g.
this doc at 10.4% digits on a math/stats thesis — within bounds) vs.
pure-number junk (docs where 40%+ of non-ws chars are digits — usually
corrupted table dumps, OCR of price sheets, port stats, etc.) look
similar on the existing ratios because digits are NOT ASCII punct and
NOT Greek-letter — they sit in the `digits` bucket of CharsetCounts
that currently has no exposed ratio.

**Threshold estimate**: no data yet; calibrate on a fresh sample once
the counter is wired. Expect most docs under 5%, math/stats docs
5-15%, likely pure-junk tables > 25-30%.

**Implementation**:

1. Add `charset_numeral_ratio` to `CharsetRatios::from_counts` in
   `charset_module.rs`. Numerator: existing `c.digits`. Denominator:
   existing `non_ws`.
2. Expose in `analyze_charset` PyDict alongside the existing ratios.
3. Emit in cleaner driver stats JSONL alongside
   `charset_{greek,moji,punct}_ratio`.
4. Add to sampler metadata block.
5. Threshold stays unset pending user review — just a diagnostic for
   now. Add to `THRESHOLDS.yaml` as null.
6. **Code-block exclusion** (user add-on, 2026-04-23): the numerals
   counter — and retroactively the existing greek/moji/punct ratios
   — should also skip code-block content. Three code-block classes
   to exclude (both numerator and denominator):
   - **Fenced code blocks**: region between matching ` ``` ` /
     ` ~~~ ` fences. Handle like the `$$…$$` LaTeX-block state
     tracker already in `count_charsets` — add a `in_code_fence`
     flag.
   - **Indented code blocks**: lines starting with 4+ spaces or a
     tab (classic markdown code block) — add as a predicate in
     `is_format_scaffolding_line`.
   - **Inline backtick spans**: `` `...` `` — strip via a helper
     analogous to `strip_html_comments` / (planned) `strip_url_spans`.

   Motivation: source code snippets in mixed Greek/technical docs
   contain high ASCII-punct, high digits, and low Greek — ratios
   become meaningless on code regions. See also Case 11 for the
   XML-source-as-plain-text variant.

---

## Case 11 — `&lt;`/`&gt;` as literal XML source (clarifies case 4, 2026-04-23)

**User question, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/ge_1p567pct/01511_pct0053_openarchives_gr_dce33baa08b6ca85221e87.md`:

> "I wonder if lt and gt are part of the text"

**Answer: yes, and specifically as XML source code**. This doc
contains 201 `&lt;` + 129 `&gt;` — not mojibake, not corruption. The
body is a dumped Android string-resource file, with escaped angle
brackets:

```
&lt; string-array name="poleis" &gt;
&lt; item &gt;Επιλογή...&lt;/ item &gt; &lt; item &gt;Άγιος Ευστράτιος&lt;/ item &gt; ...
&lt;/ string-array &gt;
&lt;? xml version="1.0" encoding="utf-8" ?&gt;
&lt; resources &gt;
```

The Greek content inside the tags (city names from a weather app) IS
valuable; the tag syntax around it is code. Two independent problems
are stacked:

1. **Entity decode missing** (Case 4): once `&lt;` / `&gt;` → `<` / `>`,
   the structure is visible as proper XML.
2. **Code-region detection missing**: even after decoding, `<item>`,
   `</item>`, `="utf-8"`, `name="poleis"` etc. inflate the `ascii_punct`
   and `latin_letters` counts. None of this is markdown-fenced (no
   triple-backticks around it) — so the standard fenced-code-block
   detector wouldn't catch it.

**Provisional plan for wave 2** — three additional code-region
detectors for `is_format_scaffolding_line` / inline strip helpers:

1. **XML/HTML source lines**: heuristic match on lines containing
   ≥2 angle-bracket tag patterns `<[^>]+>` (post-entity-decode). If
   a line is ≥60% tag syntax by char count, exclude. Alternative:
   multi-line detector that finds contiguous XML-tag blocks.
2. **URL-dense lines** (Case 3 extension): if a line is mostly URL +
   surrounding syntax, exclude even if no tags.
3. **Key=value assignment lines**: `name="…" value="…"` style.

Order: run entity decode (Case 4) FIRST, then these detectors work
on readable input.

**Caution**: detection must be precise — Greek prose about XML/HTML
itself (e.g. a tutorial explaining `<div>` in Greek) should NOT be
excluded wholesale. Safe path: exclude only lines with ≥3 tags AND
where tag-char-density > 40%.

---

## Case 12 — Bilingual CS papers are being shredded by per-char filter (critical, 2026-04-23)

**User worry, on two representative CS papers**:

- `01542_pct0182_openarchives_gr_a3fd9560bdbc32454109a3.md` —
  `cleaning_only_deletion_pct=18.2%`, `per_char_filter=68,171 chars`
- `01563_pct0026_openarchives_gr_07cabfda74bff106cf1c40.md` —
  `cleaning_only_deletion_pct=2.6%`, `per_char_filter=3,145 chars`

> "unfortunately it seems we will be losing CS papers like …"

**Empirical profile**:

| field | 01542 | 01563 |
|---|---:|---:|
| non_empty_chars_in | 374,394 | 123,883 |
| ASCII letters | 116,054 | 32,218 |
| Greek letters | 93,887 | 52,796 |
| ASCII punct (post-exclusion) | 38,716 | 15,827 |
| digits | 4,160 | 886 |
| PUA (top-3 shown) | U+F0D8/F0E0/F0FC ×221 | — |
| `charset_greek_ratio` | 0.371 | 0.519 |
| `charset_punct_ratio` | 0.153 | 0.155 |
| `per_char_filter` drop | 68,171 (18.2%) | 3,145 (2.5%) |
| `line_drop` | 0 | 71 |
| `normalization` | 61,674 (16.5%) | 4,093 (3.3%) |

**Shape**: bilingual Greek + English academic CS, with heavy use of
code snippets, math notation, references like `[1,2,3]`, URLs, and
(in 01542) PUA chars from Symbol / technical fonts. 55/45 or 38/62
Latin/Greek split — both halves are legitimate prose.

**Why the cleaner is so aggressive here**: per_char_filter strips any
char not in the allowed-scripts set
(`greek / latin / french / spanish / punctuation / numbers /
common_symbols`). CS papers carry lots of chars OUTSIDE that set:

- Math operators (U+2200-U+22FF) — `∀ ∃ ∈ ⊂ ∑ ∏ ∫`
- Arrows (U+2190-U+21FF) — `→ ← ⇒ ⇔`
- Geometric shapes + box-drawing (U+2500-U+25FF) used in algorithm
  pseudocode boxes or diagrams
- Super/subscripts (U+2070-U+209F)
- Greek-via-Symbol-font PUA (Case 10) — the 18% deletion on 01542 is
  substantially this: those `U+F0D8` / `U+F0E0` / `U+F0FC` and friends
  are Symbol-font Greek / math, **recoverable** to real chars, NOT noise
  to strip.

**Consequence**: if the deletion threshold for doc rejection is set at
or near the CDF elbow (1.57% for openarchives), these CS papers ALSO
hit the threshold — we'd reject them. But they are high-value
content: bilingual academic text, a tokenizer-training target par
excellence.

**Provisional plan for wave 2** — four compounding fixes, each
independently needed, all bearing on CS-paper preservation:

1. **Adobe Symbol PUA decode** (Case 10) — recovers 18% of chars in
   01542 BEFORE they hit per_char_filter. `per_char_filter` drop
   should collapse from 18% → 1-2% on this class of doc.
2. **Widen `common_symbols` set** to include math operators (U+2200
   range), arrows (U+2190), geometric shapes used as bullets/markers,
   super/subscripts. The tokenizer should learn them; the cleaner
   should not strip them as "non-allowed".
3. **Code-block / XML-source exclusion from charset ratios** (Cases
   10 + 11) — so `charset_punct_ratio=0.153` doesn't fire the
   `> 0.30` punct drop rule on CS papers. (Currently 0.153 < 0.30 so
   these survive the charset filter, but the user's concern is about
   deletion-% thresholds, which are still unset.)
4. **Deletion-threshold calibration** must be done AFTER fixes
   (1)-(3) land. Setting a threshold against current v5 stats
   punishes CS papers for noise classes the cleaner is supposed to
   RECOVER, not delete. Re-run stats, re-pull samples, re-assess
   the elbow.

**Hard rule for threshold setting**: before locking any
`deletion_too_high` threshold in `THRESHOLDS.yaml`, manually inspect
the top-N per-dataset rejected docs at that threshold and confirm
none are CS-bilingual / math / technical papers. If they are, the
threshold is wrong — first go fix the cleaner to not over-strip, then
re-calibrate.

**Open question**: should the deletion-% threshold be per-dataset
(relaxed for openarchives / Apothetirio — which contain bilingual
theses) or uniform? Per-dataset is more work to maintain but
preserves the "don't punish CS papers" invariant automatically.

**User verdict (2026-04-23)**:

> "so far deletions dont seem to be an issue at all actually...
> if we need to set a threshold I will inform you, do not add
> any rules on top until we do though."

So **no deletion-% threshold work** until explicitly asked. The
fixes (1)-(3) above are still valid — they target cleaner
over-stripping behavior, not rejection rules — but item (4)
"deletion-threshold calibration" is parked. Cleaner behavior fixes
proceed; threshold setting is user-driven. (Captured as memory
`feedback_no_threshold_rules_unprompted.md`.)

---

## Case 13 — Symbol audit on doc 02003: latin1_supp false-positives (audit, 2026-04-23)

**User question, on**
`/home/foivos/data/glossapi_work_cleaned_v5/charset_run/openarchives_knee_500x500/ge_1p567pct/02003_pct0277_openarchives_gr_9457071677120feed2db7f.md`:

> "can you find any frequent symbols we havent added to punctuation
> here as well like the previous one I asked?"

**Top non-alphabet chars NOT counted in `ascii_punct`** (after
applying line-level exclusions: tables, separators, HTML-comments,
LaTeX block math):

| count | char | name | current bucket | comment |
|---:|---|---|---|---|
| **3,281** | `«` | LEFT GUILLEMET (U+00AB) | latin1_supp → **moji** | Greek/EU quote — false positive |
| 2,805 | `£` | POUND (U+00A3) | latin1_supp → **moji** | could be currency or mojibake substitute, suspicious volume |
| **2,574** | `·` | MIDDLE DOT (U+00B7) | latin1_supp → **moji** | Greek ano teleia — false positive |
| 892 | `»` | RIGHT GUILLEMET (U+00BB) | latin1_supp → **moji** | Greek/EU quote — false positive |
| 519 | `€` | EURO (U+20AC) | currency | legitimate currency |
| 501 | `■` | BLACK SQUARE (U+25A0) | geo_shapes | bullet/marker |
| 168 | `°` | DEGREE (U+00B0) | latin1_supp → **moji** | technical |
| 97 | `§` | SECTION (U+00A7) | latin1_supp → **moji** | technical |
| 93 | `®` | REGISTERED (U+00AE) | latin1_supp → **moji** | technical |
| 68 | `•` | BULLET (U+2022) | gen_punct | bullet/marker |
| 51 | `\xad` | SOFT HYPHEN (U+00AD) | latin1_supp → **moji** | invisible — strip silently |
| 45 | `™` | TRADE MARK (U+2122) | letterlike | technical |
| 28 | `♦` | BLACK DIAMOND (U+2666) | misc_symbols | bullet/marker |
| 21 | `✓` | CHECK MARK (U+2713) | misc | bullet/marker |
| 18,890 | 0-9 | ASCII digits | digit | counted in (planned) `numeral_ratio` per Case 10 |

Tail of one-off real noise (single occurrences each, definitely
mojibake / OCR misclassification):
- Canadian Syllabics (U+18EB), Devanagari (U+A8EB), Myanmar (U+102B),
  CJK ideographs (U+38EB / U+572E), one PUA, one Plane-14 codepoint.

**Same finding as Case 13 on 01533**: the `latin1_supp` bucket → `moji`
mapping is a structural false-positive source. Across these two docs
the four chars `«` `·` `»` `£` alone produce **9,552 false-moji
contributions**. Three of the four are legitimate Greek punctuation;
the fourth (`£`) is suspicious-volume and merits its own investigation.

**Pattern across 01533, 02003, and likely others**: `charset_moji_ratio`
is being inflated by Greek punctuation. The current 0.25 doc-drop
threshold is robust against the inflation only because real mojibake
docs typically push the ratio much higher (0.40+). But the signal is
muddied for borderline / sample-review purposes.

**Provisional refactor for the next moji-bucket revision** (NOT a
threshold change — purely a categorization fix):

1. **Move out of `moji` bucket** (legitimate Greek/EU chars):
   - `« » · § ° ® © ™ €` and family → new bucket
     `legit_extras_ratio` or fold into `ascii_punct_ratio` definition.
2. **Strip silently** (invisible / format-only): `\xad` SOFT HYPHEN.
3. **Bullet/marker bucket** (geometric shapes, check marks, diamonds,
   pointers): `■ ► ♦ ✓ • ❖` → new `bullet_marker_ratio`.
4. **Keep in moji** (real mojibake-residue substitutes): IPA-extensions,
   PUA (where not Adobe-Symbol — Case 10 decodes those out first),
   specials/FFFD, latin_ext_b, unmapped CJK / random-script one-offs.
5. **Investigate `£`** specifically: in this doc 2,805 occurrences in
   a Greek thesis is implausible as legitimate currency. Could be
   mojibake of `λ` (lambda) or a font-substitution artifact — needs
   a small Gemini sample review or manual inspection of context.

NO threshold changes proposed (per `feedback_no_threshold_rules_unprompted.md`).
The bucket refactor is a definitional improvement; thresholds against
the new buckets stay user-driven.

---

(future cases to be appended below)

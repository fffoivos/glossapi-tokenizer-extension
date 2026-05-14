# v6 review notes (2026-04-24)

Running log of issues found while reviewing post-cleaner v6 output.
Unlike `user_review_notes.md` (which tracked issues in pre-cleaner
samples), this file tracks **surviving noise in the v6 training
corpus** — i.e. things the cleaner should have handled but didn't.

**Scope rule:** record only. Do NOT implement until explicitly asked.

---

## v6-01 — `\quad \\ \quad \\ \quad …` escapes LaTeX element-repeat detector

**Doc:** `openarchives_top_residue_punct/top500_by_charset_punct_ratio/
997003_000_pct0001_openarchives_gr_e2cbfdac1be51532f69c43.md`, line 2059.

**Pattern observed:** **321 `\quad` tokens** on one line inside a `$$…$$`
block, separated by `\\` (LaTeX display-math line break):

```
\quad \\ \quad \\ \quad \\ \quad \\ \quad \\ … (321×)
```

Neighbouring ordinary context is clean math (`\Rightarrow \exists f
\in L^1(\Omega)`), so the issue is localised to the repetition run.

**Why the element-repeat detector missed it:**

`next_latex_element` skips LaTeX formatting spacers `\,` `\;` `\!` `\ ` as
separators between elements, but does NOT skip `\\`. So the
tokenizer's stream for `\quad \\ \quad \\ \quad` is:

1. `\quad` (element)
2. `\` (fallback single-char element — no letter follows the first `\`)
3. `\` (another single-char element — the second `\` of `\\`)
4. `\quad` (element)
5. `\` `\`
6. `\quad`
7. …

Because the `\`-char elements sit between `\quad`s, the
`run_count` in `detect_repeated_element_cut` resets before it can
exceed the threshold. Run never reaches 4 consecutive `\quad` in
the token stream.

**Fix direction (not implemented):**

Extend `next_latex_element`'s separator-skip to include `\\`:

```rust
if next == b',' || next == b';' || next == b'!' || next == b' ' || next == b'\\' {
```

`\\` is a display-math line-break — pure formatting, not content.
Treating it as a separator semantically correct AND fixes this case.

**Verification needed before landing:**

- Regression test: `\quad \\ \quad \\ \quad \\ \quad \\ \quad` should
  trigger `detect_repeated_element_cut` at threshold 4.
- Negative: `a \\ b \\ c \\ d` (short multi-line math where each line
  has DIFFERENT content) — currently `\\` forces element-run reset; if
  `\\` is skipped, four distinct letter atoms in a row is fine (still
  doesn't trigger exact-repeat threshold).
- Negative: `\begin{cases} a \\ b \\ c \end{cases}` — each of a/b/c
  different → safe.

**Risk:** low. `\\` is almost never content-bearing in math mode.

---

## v6-02 — cyclic / alternating compound-element repetition (`\intertext{…}` loop)

**Doc:** same `997003_000_…_e2cbfdac` doc, line 2401.

**Pattern observed:** on one line, repeated alternation of two
`\intertext{…}` expansions with varied braced content:

```
\intertext { l e n $ \mathbb { N } $ } \intertext { d e g $ \mathbb { N } $ }
\intertext { l e n $ \mathbb { N } $ } \intertext { d e g $ \mathbb { N } $ }
\intertext { l e n $ \mathbb { N } $ } \intertext { d e g $ \mathbb { N } $ }
\intertext { l \\
```

Two distinct compound elements alternating (cycle length = 2). Enough
to visibly spam the line, but our existing detectors can't catch it:

- `detect_repeated_element_cut` requires IDENTICAL consecutive
  elements; alternation resets the run.
- `detect_monotonic_element_cut` requires a numeric sub/sup
  progression; there isn't one here.
- `detect_small_vocab_run_cut` requires every element's base to be in
  the fixed Greek/decorator vocabulary; `\intertext` isn't there.
- `detect_repeated_char_cut` and `detect_repeated_lines_cut` miss it
  for the usual reasons (no single char repeats 30×, single line).

This is a **different class of looping** than any we currently detect:
**cyclic pattern with a dynamic small vocabulary** (derived from the
doc at runtime rather than from a static list).

**Reference in existing code base:**

`src/glossapi/corpus/phase_clean.py:145` has
`LATEX_SEGMENT_ALTERNATING_RUN_MIN = 6` — suggesting the Python
code-base ALREADY detects alternating runs. Worth reading that path
before inventing something new; likely direct port possible.

**Fix direction (not implemented):**

Add `detect_cyclic_element_cut(s, cycle_max=6, threshold=6)` —
sliding-window detection of a short cycle of length 1..`cycle_max`
that repeats ≥ `threshold` times. Pseudo-algorithm:

1. Tokenize into elements (via `next_latex_element`, already in Rust).
2. For each candidate cycle length L = 1..cycle_max:
   - Maintain a rolling window of L elements ending at position i.
   - Check if window[i-L .. i] == window[i-2L .. i-L]. If so increment
     cycle-run counter; else reset.
   - If cycle-run ≥ threshold × L tokens, cut at start of run.
3. Shortest repeating cycle wins (for determinism on e.g.
   `a a a a …` which matches both L=1 and L=2).

**False-positive considerations (to design tests for):**

- `\cdot` between terms `a \cdot b \cdot a \cdot b` — cycle length 2,
  but these are DIFFERENT variables with `\cdot` — a naive tokenizer
  sees `a, \cdot, b, \cdot, a, \cdot, b` which is a cycle-4 pattern
  (`a, \cdot, b, \cdot`). With `\cdot` treated as separator this
  collapses to `a, b, a, b` — which IS a cycle. But this is legitimate
  math if short. Threshold needs to be high enough: ≥6 full cycles =
  12 tokens minimum.
- Matrix rows `a \\ b \\ c \\ a \\ b \\ c` (3-cycle) — could be
  legitimate in a matrix listing.
- Test dataset required before landing (per
  `feedback_dont_generalize_beyond_test_parameters`).

**Risk before testing:** moderate. Cycle detectors are notorious for
false positives on legitimate structured math; need careful threshold
calibration + thorough negative dataset.

**Priority:** lower than v6-01 (which is a simple bug in the existing
detector); v6-02 is a new detector class.

---

---

## v6-03 — `charset_punct_ratio` still counts single-line `$$…$$` LaTeX

**Docs:** multiple, e.g.:
- `openarchives_top_residue_punct/top500_by_charset_punct_ratio/997013_004_pct0000_openarchives_gr_42896f9c40ed48a4b2f5a8.md`
- any math-heavy doc where Docling collapses equations onto single lines

**User observation:**

> "punctuation still seems wrong though. how can …997013_004…md have
> 30% without calculating the latex?"

**Empirical measurement (997013_004):**

| Metric | Value |
|---|---:|
| `charset_punct_ratio` (as reported) | **0.2986** |
| `charset_greek_ratio` (as reported) | 0.3415 |
| ascii_punct ratio with `$$…$$` spans stripped | **0.0925** |
| greek_letter_ratio with `$$…$$` spans stripped | **0.7031** |
| single-line `$$…$$` spans in doc | **366** |
| multi-line `$$…$$` block boundaries in doc | **0** |

If LaTeX were properly excluded, punct would be ~9%, not ~30%.

**Root cause:**

`charset_module::count_charsets` excludes LaTeX via a `$$`-toggle
state machine: counts `$$` occurrences per line, toggles `in_latex_block`
when count is ODD. For a single-line `$$…$$` the count is 2 (even), so
the state machine does not enter the math state and the line's chars
are all counted. Only MULTI-line `$$\n…\n$$` blocks are excluded.

This is the exact bug recorded in wave-1 Case 5 but never actually
fixed in charset_module. The wave-2 plan deferred it with the
rationale "improves ratio reporting, doesn't affect output text" —
that rationale is wrong when the ratio feeds downstream filter rules
(`charset_punct_ratio > 0.30` drops docs). This doc sits at 0.2986 —
one crumb below the drop threshold — purely because of LaTeX
inflation. Math-heavy papers could be rejected solely on this signal.

**Fix direction (not implemented):**

Replace the home-grown `$$` toggle in `count_charsets` with a call to
`latex_module::find_dollar_dollar_spans(text)`, which correctly
identifies BOTH single-line and multi-line spans (already built for
the repetition cropper). Use the returned spans to exclude those byte
ranges from the per-char ratio count.

Implementation sketch:

```rust
let latex_spans = crate::latex_module::find_dollar_dollar_spans(text);
// Build a set of "excluded byte ranges"; iterate chars, skip any
// whose byte offset falls inside one of the ranges.
```

Since `find_dollar_dollar_spans` is already in `latex_module`,
`charset_module` consuming it matches the co-location rule: one
detector (latex_module), many consumers (charset_module, cropper).

**Test needed:** regression test that asserts a doc made entirely of
single-line `$$x^2 + y^2 = z^2$$` math has `charset_punct_ratio`
close to 0, not to whatever ratio the `$$` syntax plus LaTeX-internal
punct produces.

**Verification target:** the 997013_004 doc. Expected post-fix
charset_punct_ratio ≈ 0.09 (not 0.30).

**Priority:** HIGH. Currently inflates punct ratio on any math-heavy
doc, which risks false drops if we ever tighten the 0.30 cutoff, and
pollutes the top-by-punct sample pool with math papers that shouldn't
be there.

---

## v6-04 — doc 997013_003: no hallucination detected, just dense real math

**Doc:** `openarchives_top_residue_punct/top500_by_charset_punct_ratio/997013_003_pct0000_openarchives_gr_42896f9c40ed48a4b2f5a8.md`.

**Observation:** same underlying doc as 997013_004 (hash
42896f9c40ed48a4b2f5a8 appears twice — probably counted as multiple
elements within the corpus / different shards). Zero `repetition
cropped` markers in the body. Top-token-density line has 140 distinct
LaTeX tokens — but manual inspection shows these are legitimate math
expressions (integrals over `[0, \infty)`, expectation calculations,
convolution integrals). `\sum` appears ×36 on line 900 but each is
inside a distinct summation over a different variable / index. Not
hallucination.

**No action needed.** Filed so we know the detector made the right
call on this one.

**Implication:** the doc sits at 0.2986 charset_punct_ratio NOT
because of hallucinated repetitions but because of legitimate
LaTeX syntax density. v6-03's fix (excluding `$$…$$` from the
ratio) would move this doc well below the 0.30 cutoff and
correctly classify it as a clean math paper.

---

---

## v6-05 — `\frac{A}{A}` / structurally-degenerate LaTeX constructs (new pattern class)

**Doc:** `openarchives_top_residue_punct/top500_by_charset_punct_ratio/
997003_001_pct0001_openarchives_gr_20ca0a6d2eb31e759f4cda.md`, line 785.

**Pattern observed:**

Inside a single-line `$$…$$` block on line 785 (2,815 chars, 147
LaTeX tokens — otherwise legitimate-looking math derivation), there
is a fraction whose numerator and denominator are IDENTICAL:

```
\frac{ \int _ { k } ^ { s } e ^ { x } v _ { \kappa } ( x ) d x }
     { \int _ { k } ^ { s } e ^ { x } v _ { \kappa } ( x ) d x }
```

The 40-char `\int _ { k } ^ { s } e ^ { x } v _ { \kappa } ( x ) d x`
expression appears as both num and denom.

**Why this is noise, not math:**

`\frac{A}{A}` always simplifies to 1 (for A ≠ 0). A mathematician
would never write it out literally — they'd collapse to `1`. So
when we see it, it's an extraction/OCR/hallucination artifact:
the model emitted the same substring twice instead of producing
the actual denominator.

**Why none of our detectors caught it:**

- char-level: no single char repeats 30×
- line-level: single line
- element-repeat: the identical span is inside brace arguments; the
  outer `\frac` counts as ONE element; there's no consecutive
  repetition at the element level
- monotonic: no numeric progression
- small-vocab-run: `\frac` and `\int` not in the small-vocab set

**New detector class needed:**

`detect_degenerate_latex_constructs(text)` — checks for
structurally-degenerate LaTeX expressions:

1. `\frac{A}{A}` where A == A after whitespace normalization.
2. `\frac{A}{B}` where A and B are canonically equal (same as 1
   but with possibly re-ordered or re-parenthesized forms — harder
   to catch without a real parser).
3. (future) `\int_a^a`, `\sum_{i=k}^{k}` — equal bounds.
4. (future) `\lim_{x \to x}` — limit against self.

Implementation: parse `\frac`/`\int`/`\sum` calls with brace
balancing (the `next_latex_element` tokenizer already does the hard
part); extract arg groups; compare. O(n) single pass.

**False-positive considerations:**

- Pedagogical explanation of fraction identity: "we know that
  `\frac{x}{x} = 1`" — written out in a textbook. Rare; acceptable
  FP rate.
- Legit equality check expressions: `\frac{a+b}{a+b} \cdot c = c` —
  extremely rare outside derivation proofs.
- Cancellation proofs: `\frac{f(x)g(x)}{f(x)} = g(x)` — but num ≠
  denom here so not caught.

Expected FP rate: <0.1% of math docs.

**Cut strategy:**

- Option A: delete just the degenerate `\frac{A}{A}` span (replace
  with `1` or empty).
- Option B: cut from the degenerate point onward (assumes
  everything after is hallucinated too — conservative, aligns with
  OCR-style "tail-stop" semantics).

Option A is cleaner; the hallucination is localised and the rest of
line 785 looks like legitimate math.

**Priority:** MEDIUM. Less common than repetition classes 1-4 but
genuinely un-caught by existing detectors and meaningfully harmful
(it represents a SEMANTIC error in the math).

---

---

## v6-06 — LaTeX that looks "math-shaped" but is structurally / semantically broken

**Doc:** `openarchives_top_residue_punct/top500_by_charset_punct_ratio/
997038_007_pct0001_openarchives_gr_37d19e752f0740160531a0.md`
(`\charset_punct_ratio=0.2961`, 412 `$$…$$` spans).

**User question:** "Do the latex here make sense?"

**Answer:** no, multiple distinct defects. Not spam-repetition — these
are harder cases where each line LOOKS like math and passes our five
repetition detectors but is actually broken at the syntax / semantic
level.

### Example defect 1 — partial-pattern re-emission

Span 1 (3,003 chars single-line `$$M_2(s) = …`):

```
... - \frac{\tau_{21}(\rho_1)}{\rho_1 - \rho_2} T_s T_{\rho_1} \omega(0)
    - \frac{\tau_{21}(\rho_2)}{\rho_2 - \rho_1} T_s T_{\rho_2} \omega(0)
    - \frac{\tau_{21}(\rho_2)}{\rho_2 - \rho_1} T_s \omega(0)             ← missing T_{\rho_2}
    - \frac{\tau_{21}(\rho_2)}{\rho_2 - \rho_1} ...                       ← 4th with same prefix
```

The `- \frac{\tau_{21}(\rho_2)}{\rho_2 - \rho_1}` *prefix* repeats
with inconsistent tail subscripts. Classic hallucination signature:
the model generated the prefix correctly a first time, then kept
repeating it with noisier tails. Not caught by element-repeat
(distinct full elements because of tail variation) or small-vocab
(`\frac`, `\tau` etc. not in small-vocab set).

### Example defect 2 — mismatched braces

Span 2:

```
+ [ 1 - \mu_2 s }{ 1 - \mu_2 s } \\
```

The `[1 - \mu_2 s ]{1 - \mu_2 s}` opens with `[` but the matching
close is `}`. Invalid LaTeX — no grammar can produce this.
Docling/model transcription error.

### Example defect 3 — self-referential index

Span 3: `\sum_{j_1 = 1}^{m_{j_1}}`.

Upper limit references the dummy index itself. Mathematically
meaningless (what's the range of `j_1`?). Almost certainly the
outer index (should be `m_i` or `m_{n}`) was dropped.

### Why this matters

Training data containing SYNTACTICALLY broken LaTeX teaches the
tokenizer / LM to reproduce the defects. Training data containing
SEMANTICALLY broken math (self-reference, partial re-emissions)
teaches bad reasoning patterns. These are worse than pure spam —
they look plausible.

### Fix direction (not implemented)

Three candidate detectors, in order of tractability:

1. **Brace-balance check inside `$$…$$`**: simple `{`/`}` depth
   counter with `[` / `]` / `(` / `)` tracking. If unbalanced, flag
   the span as broken. Reject OR pass through a second-chance parser.
   Low false-positive rate.
2. **Prefix-repeat-with-noise-tail**: sliding window of
   element-canonicalised prefixes (first K elements of a
   `\command{...}...\command{...}` pattern) checked for repetition
   with tolerance on the tail. Higher FP risk — real math does
   repeat prefixes of sums/integrals with different tails.
3. **Self-referential index detection**: parse `\sum_{j = 1}^{EXPR}`
   and check whether `EXPR` references `j`. Unambiguous (if `j`
   appears as a standalone token inside the expr, flag). Low FP rate.

(1) is the cleanest first win. (3) is a small parser extension.
(2) is the hardest and most FP-prone — defer.

### Priority

MEDIUM-LOW. These cases are a minority of the corpus. But they're
the RESIDUAL noise that slips past everything else; worth a wave-3
pass once the high-volume patterns (v6-01 through v6-05) are resolved.

---

---

## v6-07 — pseudo-tables (TOC / figure-index as 2-col markdown tables)

**Doc:** `openarchives_top_residue_punct/top500_by_counter_script_residue/
952131_000_pct0007_openarchives_gr_84741e3d2ba3eb26bfb10d.md`
(the ADR dangerous-goods regulation manual; 4.45 MB input → 1.87 MB
body).

**User question:** "what are these crazy looking tables?"

**Answer:** they are **pseudo-tables** — Docling wrapped the
document's table-of-contents and figure/image indexes into 2-column
markdown tables where the second column is almost always empty
whitespace. 311 table clusters in this doc; biggest examples:

```
| ΚΕΦΑΛΑΙΟ 1 - ΓΕΝΙΚΕΣ ΔΙΑΤΑΞΕΙΣ   .................... 1 |                    |
| --- | --- |
| 1.1 ΕΙΣΑΓΩΓΗ .................... 1                    |                    |
| 1.2 Η ΟΔΗΓΙΑ ADR   .................... 1              |                    |
| 1.3 ΔΟΜΗ ΤΗΣ ΟΔΗΓΙΑΣ ADR   .................... 3      |                    |
...
| Εικόνα 1 - ADR Πιστοποιητικό ….17                      |                    |
| Πίνακας 19 - Κωδικοί Δεξαμενών ….…83                   |                    |
```

Each "row" is a single TOC entry or figure caption + page reference.
The second column exists only because Docling's layout detection
inferred two columns on the source-PDF page (probably because of
right-aligned page numbers being visually separated). But the
content is a one-dimensional list.

**Why our existing table-scaffolding-exclusion handles it partially:**

`charset_module::is_format_scaffolding_line` already excludes these
lines from the greek/moji/punct ratios — each line starts AND ends
with `|` and has ≥2 pipes. That's why the doc has
`charset_punct_ratio=0.041` (low) — the pipe-syntax chars and
dot-leader runs inside pipes don't count. ✓ working as intended.

**Why it's still a problem:**

The pseudo-table REMAINS in the cleaned output text and feeds the
tokenizer. Each row's surface form is `| <Greek title> ...<page> |
<whitespace> |` — the tokenizer learns that `|` often appears in
sequences of 3+, that whitespace between pipes is a valid "cell",
that Greek headings end with dot-leader + digits. None of those
patterns occur in the natural Greek corpus the model is supposed
to be producing. Bad training signal at scale (311 clusters × ~20
rows/cluster = 6,000+ pseudo-table rows in THIS one doc alone).

**Fix direction (not implemented — restating from wave-1 Case 5):**

Add a pseudo-table detector between `detect_table_regions` and
`remove_tables_from_content` that flags:

1. **Single-column wrapper TOCs**: all body rows have 1 column (only
   content between the outer pipes, no interior pipes) AND ≥50% of
   rows match the TOC pattern `…\.{4,}\s*\d+\s*$` (ends with
   dot-leader + page number).
2. **Near-empty second columns**: all body rows have 2+ columns AND
   the N-th column is whitespace-only in ≥80% of rows. The content
   is actually 1D, just visually formatted as 2D.
3. **>80% empty cells across the whole table**: catches tables
   whose content is almost entirely missing (render failure).

For (1) and (2): **unwrap** — strip the `|` and separator row,
emit cell contents as plain lines. The TOC info is valuable as a
section listing; only the table scaffolding is wrong.

For (3): **drop** — emit `<!-- table-removed -->` marker.

**False-positive considerations:**

- Legit single-column tables with TOC-like content (rare; modern
  markdown docs would use a bulleted list). Would unwrap to bullet-
  less lines — fine.
- Intentional 2-col layouts with sparse second column (e.g.
  glossary tables with translation in col 2). Would unwrap some
  of these — needs empirical test on a sample.
- Calibrate thresholds (≥50% TOC pattern, ≥80% empty col N) on a
  dataset of verified pseudo vs. real tables before landing.

**Scale impact:**

Just openarchives alone likely has tens of thousands of docs with
similar TOC wrappers (any PDF with a table of contents). Fixing
removes a lot of structural noise from training data.

**Priority:** MEDIUM-HIGH. Large volume of affected content;
cleaner fix is mechanical; false-positive risk manageable with
sample-based threshold calibration.

---

---

## v6-08 — PDF custom-CID corruption: chars are "Greek" but words are garbage

**Doc:** `openarchives_top_residue_punct/top500_by_counter_script_residue/
975804_001_pct0388_openarchives_gr_0a1d83aac1ecfc18f5aa49.md`.

**User observation:** "bad in a way neither greek badness nor residue
captures fully. maybe it has too many accents?"

**Empirical profile (post-cleaner body, 186k content chars):**

| metric | value |
|---|---:|
| `charset_greek_ratio` | 0.6941 (HIGH — looks Greek) |
| `charset_moji_ratio` | 0.0551 |
| `charset_punct_ratio` | 0.0223 |
| upstream `greek_badness_score` | None |
| Unicode blocks of top chars | 100% Greek |

Top 5 chars:
```
21,437 Ν  (GREEK CAPITAL NU)
12,744 α
 9,347 δ
 9,136 κ
 8,129 θ
```

Top 5 bigrams:
```
3,911 μΝ
2,640 θΝ
2,293 δΝ
2,136 αΝ
2,042 κυ
```

Body sample:
```
δπζπηαδεάΝλΰαέαΝυπκίζβγέαΝκΝΣηάηαΝΟδεκθκηδευθΝπδβηυθΝκυΝΠαθπδβηέκυΝ...
## ΠΝΠΙΣΜΙΟΝΠΙΡΙΩΝ ΣΜΜΝΟΙΚΟΝΟΜΙΚΝΠΙΣΜ
```

**Diagnosis:**

Two stacked PDF-extraction corruptions:

1. **Space → `Ν` substitution.** `Ν` is the top char overall (more
   common than α) and appears almost exclusively adjacent to other
   letters in positions where word-boundary spaces should be. The
   custom CID font in the PDF mapped the space glyph to Unicode
   U+039D (Greek Capital Nu) instead of U+0020.
2. **Vowel loss.** `ΠΝΠΙΣΜΙΟ` is `ΠΑΝΕΠΙΣΤΗΜΙΟ` with Α/Ε/Τ/Η dropped.
   The font's vowel glyphs were either not embedded or mapped to
   unassigned codepoints that got stripped.

**Why the filters miss it:**

Every char IS Greek. All charset ratios look clean. Script residue
is only partially elevated (from `Ϊ Ϋ ΰ` at a few thousand each —
enough to put this doc in the top-500 by counter_script_residue,
but way below the threshold that would trigger a drop).

**New detection class needed:**

Lexical / word-shape validation — check whether the Greek letter
sequences actually form Greek words. Candidates:

1. **Dictionary lookup** — split on whitespace, check each "word"
   against a Greek lexicon. Reject docs where <X% of words match.
   Best coverage; highest implementation cost (need lexicon).
2. **Character n-gram coherence** — compare bigram/trigram
   distribution against a known-good Greek reference corpus. Large
   Kullback-Leibler divergence → flag. Cheaper than dictionary,
   works without per-word lookup. Catches this case obviously
   (bigram `μΝ` at 3,911 is far from natural Greek distribution).
3. **Consonant/vowel ratio** — natural Greek is ~40-45% vowels.
   This doc has <20% of top-30 chars as vowels. Cheapest check.
4. **Space-to-letter ratio** — if the doc's char count is high but
   whitespace count is suspiciously low, spaces are probably
   encoded as something else. This doc has 21k `Ν` that should
   be spaces — detectable via `whitespace_chars / non_ws_chars`
   ratio compared to Greek norms (~18-22%).

**Implementation recommendation:** start with (3) and (4) — both
are O(n) single-pass char counts, no external data needed. Add
(2) as a second pass using a small built-in reference bigram
distribution (top-100 Greek bigrams by frequency). (1) is deferred —
significant effort for marginal gain over (2).

**False-positive considerations:**

- Very short docs — statistics unreliable; apply min-length gate.
- Polytonic / ancient Greek texts — bigram distribution differs
  from modern; need polytonic-aware reference OR separate threshold.
- Multi-language docs (Greek + Latin technical terms) — Latin
  fraction lowers vowel-ratio slightly; threshold must tolerate
  that.

**Priority:** HIGH. This class of corruption passes every filter
we have. Probably affects many older-PDF / non-standard-encoding
docs in openarchives specifically (old theses with non-Unicode
Greek fonts). Needs empirical sampling to estimate corpus
prevalence.

---

---

## v6-09 — 67% of the corpus has NO upstream `greek_badness_score`

**User observation:** "it cant be right that we dont have greek
badness for it" (re: doc `0a1d83aac1ecfc18f5aa49` showing `None`).

**Empirical:** scoped query across all non-HPLT canonical parquets.

| dataset | total rows | missing greek_badness | missing % |
|---|---:|---:|---:|
| HuggingFaceFW__finepdfs-edu | 209,039 | **209,039** | 100% |
| HuggingFaceFW__finewiki | 242,517 | **242,517** | 100% |
| OPUS__OpenSubtitles-el-v2018 | 143,441 | **143,441** | 100% |
| AI-team-UoA__greek_legal_code | 47,563 | **47,563** | 100% |
| openarchives.gr | 192,038 | 1,027 | 0.5% |
| all other 15 datasets | 322,079 | 0 | 0% |
| **TOTAL** | **956,676** | **643,587** | **67%** |

**Root cause:** four "newer" source parquets (the two HuggingFaceFW
releases, OPUS OpenSubtitles, and AI-team greek_legal_code) were
merged into the canonical release directory WITHOUT being routed
through the upstream quality-scoring pipeline. They have null
greek_badness_score, null mojibake_badness_score, null
greek_percentage, null latin_percentage — every one, for every row.

The openarchives gap is smaller (0.5%) but real — 1,027 docs fell
through despite the dataset being intended for scoring. Looking at
`quality_method` on openarchives: 146,038 rows via
`glossapi_rs_noise`, 46,000 via `existing_pipeline_exact`, and 1,027
via neither (method is also null). The specific doc the user flagged
(`0a1d83aac1ecfc18f5aa49...`) is in the 1,027 "neither" group.

**Consequence:**

The charset filter in the wave-2 cleaner uses only local ratios
(`greek_letter_ratio < 0.02`, `moji > 0.25`, `punct > 0.30`) — so the
cleaner itself doesn't rely on upstream scores. Those ratios passed
this doc despite it being corrupted (see v6-08 — Greek-looking chars,
garbled words). Upstream `greek_badness_score` would likely have
caught it (score tracks semantic Greek-ness at a higher level).

**So: filtering is flying blind on 2/3 of the corpus.** If any
downstream step conditions on `greek_badness_score != null AND
score < threshold`, those 643,587 docs silently fall through every
gate.

**Fix directions (not implemented):**

1. **Backfill upstream scores** for the four newer datasets. Route
   them through whichever scoring pipeline is canonical —
   `glossapi_rs_noise`-based, or the existing Python pipeline. Run
   on all 643,587 missing docs; persist into the parquets or an
   overlay file.
2. **Backfill the 1,027 openarchives stragglers** — smaller batch;
   same infra as (1).
3. **Add a null-score guard in downstream filters.** Currently any
   filter rule like `greek_badness_score > 60` reads `None > 60` as
   False (null silently passes). Explicit check: treat null as
   either (a) never-evaluated → skip filter AND log a metric, or
   (b) worst-case → fail the filter. User to decide; the current
   implicit (a) is the riskier of the two.

**Priority:** HIGH. This is a systemic gap in the data, not a single
bad doc. Every wave-2/wave-3 fix we add to the cleaner is muted on
643,587 docs because those docs bypass the upstream scoring that
would corroborate the cleaner's signal.

**Data reference:**

- Scoring methods present on openarchives.gr:
  - `glossapi_rs_noise`: 146,038
  - `existing_pipeline_exact`: 46,000
  - (null): 1,027

---

---

## v6-10 — TOC extracted as N-col table with duplicated cells + column-count mismatch

**Doc:** `openarchives_top_residue_punct/top500_by_counter_script_residue/
996194_018_pct0003_openarchives_gr_9d5a37603062a0197a69bf.md`.

**Pattern observed (verbatim excerpt):**

```
| ΜΕΡΟΣ ΤΡΙΤΟ... 193 | ΜΕΡΟΣ ΤΡΙΤΟ... 193 | ΜΕΡΟΣ ΤΡΙΤΟ... 193 | ΜΕΡΟΣ ΤΡΙΤΟ... 193 | <!-- text-missing --> |
| --- | --- | --- | --- |
| 3. Περιγραφή... 194 | 3. Περιγραφή... 194 | 3. Περιγραφή... 194 | 3. Περιγραφή... 194 |
| 3.1. Περιγραφή... | 3.1. Περιγραφή... | 3.1. Περιγραφή... | 3.1. Περιγραφή... |
| Τουρκικής για ελληνόφωνους...195 | ... (×4) | ... |
| | 3.1.1. | Ιωάννου...195 | |
| | 3.1.2. | Ιορδάνογλου...200 | |
```

Two compounding defects:

1. **Cell-content duplication across columns.** Docling inferred 4
   visual columns from the PDF's TOC layout. Full-width rows get
   their content emitted once per column (4× duplication). Indented
   rows (3.1.1, 3.1.2) get placed in one or two columns with others
   empty.
2. **Column-count mismatch between header and separator.** Header row
   has 5 cells (the last being `<!-- text-missing -->`), the
   separator row has 4 (`| --- | --- | --- | --- |`). GFM requires
   rows to match the separator's column count — parsers either
   reject the whole table (renders as raw markdown) or fall back to
   best-effort. User sees broken preview.

**Relation to v6-07:** v6-07 was 2-col tables where col 2 is empty
(TOC wrapped as 2-col). This is the multi-column variant where ALL
cols hold duplicated content. Same root cause (Docling column-
inference on TOC pages), different surface shape.

**Why this slips through existing filters:**

- `is_format_scaffolding_line`: every row starts and ends with `|`
  with ≥2 pipes → line is excluded from ratios ✓ (working as intended).
- `table_analysis_module::core_detect_malformed_tables`: DOES flag
  the column mismatch (header vs separator) but only REPORTS, does
  not drop/fix.
- `table_remover_module::remove_tables_from_content`: removes by
  external line-range spec, not by pattern detection.

The table TEXT survives into the cleaned corpus, feeding the
tokenizer 4× the same TOC entry per row.

**Three-part fix direction (not implemented):**

1. **Duplicate-cell row collapse.** For each table row: split into
   cells, whitespace-normalize each, deduplicate adjacent identical
   cells. If all cells collapse to one, emit the content as a plain
   prose line (unwrapping the `|…|` syntax). Safe; catches this
   doc's 100% full-width TOC rows cleanly.
2. **Column-count mismatch → drop.** If header/separator/body rows
   disagree on column count by >1, the table is malformed; drop
   with `<!-- table-removed -->` marker. Small FP risk on tables
   with trailing-cell extraction errors, but malformed tables are
   noise either way.
3. **TOC-cell unwrap** (restating v6-07 plan-item 1). After (1)
   collapses a row to a single text, if the text matches
   `^.{0,300}\.{4,}\s*\d+\s*$` (TOC-entry pattern), emit just the
   title without the dot-leader + page number, or emit the full
   line as prose. User-driven on the exact emission form.

Combined (1) + (2) + (3) land in one `table_region_module` refactor.
The work is already in the wave-2 plan (see
`CORPUS_CLEAN_WAVE2_PLAN.md` — "Table-region refactor" section).
v6-10 is empirical confirmation that the work is needed; the
refactor targets this shape.

**Scale impact:**

Thesis-scale docs (openarchives masters/PhD) are the typical
carriers — any doc with a multi-column TOC page. Rough order of
magnitude: 10s of thousands of docs in openarchives alone
(every thesis has a TOC, many are multi-column layout).

**Priority:** MEDIUM-HIGH. Same priority as v6-07 (they share the
fix). Large surface area, mechanical fix, low-to-moderate FP risk.

---

---

## v6-11 — cleaner's per-char filter strips NBSP (U+00A0), fusing words

**Docs (3 confirmed):**
- `openarchives_knee_500x500/ge_1p567pct/00148_958_pct0116_…_32653ca905ae85bd335cd9.md`
- `openarchives_knee_500x500/ge_1p567pct/00178_583_pct0108_…_d4c069f2617365ec4153fc.md`
- `openarchives_knee_500x500/ge_1p567pct/00194_647_pct0147_…_a7b8a49142f0ced4f18855.md`

**User hypothesis (verified correct):** "could it be that we have
removed the whitespaces here because they are special chars after we
measured greek badness?"

**Empirical confirmation (doc 00148, `32653ca905ae85bd335cd9`):**

| metric | input (raw parquet) | output (post-cleaner body) |
|---|---:|---:|
| char count | 1,749 | 1,547 |
| whitespace count | 264 | 62 |
| whitespace ratio | 15.1% (natural Greek) | 4.1% (broken) |
| longest word | 16 chars (`χαρακτηριστικών`) | 74 chars (word-fusion) |
| greek_badness_score (upstream) | 0.0 (clean) | (same value — frozen pre-cleaner) |

Whitespace breakdown in INPUT:

```
U+00A0 NO-BREAK SPACE  ×202
U+0020 SPACE            × 58
U+000A LINE FEED        ×  4
```

**202 of 264 whitespaces in the input are NBSP**, not regular space.
This is typical Docling output for PDFs where the PDF extractor
doesn't know which Unicode whitespace codepoint to emit and defaults
to NBSP (or the source PDF embedded NBSP literally — either way,
most word-boundary whitespace is NBSP).

**Root cause in the cleaner:**

`cleaning_module.rs::build_script_char_sets` constructs
`unusual_chars` by iterating `0x0080..=0x00FF` (Latin-1 Supplement)
and adding any char NOT in the `french_specific`, `spanish_specific`,
`accented_greek`, `common_symbols`, or `punctuation` allowed sets.

- NBSP (U+00A0) is in the Latin-1 Supplement range.
- `common_symbols` = `"€£¥©®™°§"` — no NBSP.
- `punctuation` = `".,;:!?()[]{}'\"&@#$%^*_-+=|\\<>/~`"` — no NBSP.
- None of the other allowed sets include NBSP.

→ NBSP lands in `unusual_chars` → per-char filter strips it as
`chars_dropped_by_per_char_filter`.

**Result:** 202 word-boundary spaces deleted from the 00148 doc.
Every Greek prose line becomes one big wordlike blob:

```
Ηεργασίααυτήέχεισανκύριοσκοπότηνέρευνα, προγραμματισμόκαιανάπτυξη
ιστοσελίδαςμετηγλώσσασήμανσης HTML5 καθώςκαιτωννέωνχαρακτηριστικών...
```

(Input was: `Η εργασία αυτή έχει σαν κύριο σκοπό την έρευνα,
προγραμματισμό και ανάπτυξη ιστοσελίδας με τη γλώσσα σήμανσης HTML5
καθώς και των νέων χαρακτηριστικών...`)

Spaces around commas and Latin tokens survive because the INPUT
happens to mix in regular spaces around punctuation/Latin — only the
pure-Greek word boundaries were NBSP.

**Why upstream `greek_badness_score` doesn't catch it:**

Score was computed on the PRE-cleaner text, which had NBSP spaces
intact. The text looked fine to upstream scoring. The cleaner then
destroyed the whitespace, but the (frozen) score in the parquet
still reports the doc as clean (0.0 - 4.14 range).

**Scale estimate:**

Every Greek PDF where Docling emits NBSP as the default whitespace
is affected. Probably thousands to tens of thousands of docs across
openarchives alone. The three docs above are just the ones the user
happened to open — this is a corpus-wide bug.

**Fix direction (not implemented):**

In `cleaning_module.rs::build_script_char_sets`, treat NBSP
(U+00A0) as whitespace — EITHER:

1. **Preserve as-is**: add NBSP to allowed chars explicitly. Simplest.
   Downstream tokenizer sees both U+0020 and U+00A0 as separators.
2. **Normalize to U+0020** in the normalize phase: add a pass
   `normalize_whitespace_variants` (or extend an existing one) that
   replaces U+00A0, U+00AD (already stripped), U+2000-U+200B,
   U+202F, U+205F, U+3000, etc. with regular U+0020 or strips the
   zero-width ones. Cleanest — tokenizer sees only normal spaces.

Option 2 is better semantically — the tokenizer won't need to
learn that NBSP is a separator, which bloats vocab. Add a static
map of "display-equivalent-to-space" Unicode chars.

**Tests needed:**

- Unit: input `Η\u{00A0}εργασία` → output contains space between
  words (either `Η εργασία` or `Η\u{00A0}εργασία`, depending on
  option).
- Regression on the three docs above: post-fix output whitespace
  ratio ≥ 14% (close to input).
- Negative: ensure we don't regress on docs that currently render
  correctly (they wouldn't have NBSP).

**Priority:** CRITICAL. Corpus-wide data-destruction bug. Every
wave-2 advantage is wiped out if prose is fused into 70-char blobs
— tokenizer trained on this produces garbage. Should be the NEXT
cleaner fix.

---

(future v6 cases to be appended below)

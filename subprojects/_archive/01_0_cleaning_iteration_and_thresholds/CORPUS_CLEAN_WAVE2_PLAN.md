> **Historical reference.** Pre-convergence cleaning-iteration work. The converged tokenizer arm is **C3** (see [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md)). Kept for traceability; do not treat as live planning.

# corpus.clean — wave 2 plan

Consolidates review findings from `reports/user_review_notes.md`
Cases 3–13 into a concrete change plan for the upstream cleaner
(`eellak/glossAPI/rust/glossapi_rs_cleaner/`). Organized **by
text-type** per `feedback_group_cleaner_features_by_text_type.md`.

Hard constraints carried in from memory:

- **All deterministic per-doc work in Rust.** Python is a thin driver.
  (`feedback_rust_for_corpus_pipelines.md`)
- **Co-locate per-text-type logic in one module per type.** One
  detector for a text-type, many consumers.
  (`feedback_group_cleaner_features_by_text_type.md`)
- **Normalization runs AFTER cleaning** — regression-tested invariant.
  (`feedback_normalize_after_cleaning.md`)
- **No deletion/rejection thresholds without explicit user request.**
  Wave 2 touches cleaner behavior only, not `THRESHOLDS.yaml`.
  (`feedback_no_threshold_rules_unprompted.md`)
- **Don't overwrite upstream pre-existing score columns** (`greek_*`,
  `mojibake_*`, `filter`, etc.). Add alongside.
  (subproject CLAUDE.md hard rule)

---

## Stage 0 — pipeline order (one pass per doc)

Target order of passes, each reading from the previous pass's output
buffer. Existing order before wave 2 is roughly: pre-clean charset
filter → matcher (three-counter) → table-remover → `core_clean_text`
(per-char + line-drop + normalize).

Wave-2 order inside `core_clean_text`:

1. **HTML-entity decode** (Case 4) — turn `&amp; &lt; &gt; &quot;
   &apos; &nbsp; &#\d+; &#x[0-9a-f]+;` back into real chars so all
   later passes see the final text. Runs first so regex-based detectors
   below don't need to handle the escaped forms.
2. **Adobe Symbol PUA decode** (Case 10) — dictionary substitution of
   `U+F020..U+F07F` (and friends) to their real Unicode equivalents.
   Recovers the Greek letters + math operators in docs like
   `01500_pct0033_…` (100% coverage on top-30 PUA chars observed).
3. **GLYPH/`/uni`/`/gN` marker deletion** (Case 7, corrected per
   Case 9) — after entity decode the angle brackets are real.
   **Default = delete** these markers (regex sub `GLYPH<\d+>` →
   empty, same for `/uni[0-9A-Fa-f]{4,6}` and `/g(?:id)?\d+`).
   **Not** substitute to `•`: `GLYPH<N>` is a font-local glyph index
   (N differs between PDFs depending on the embedded font subset),
   so a blanket mapping to any Unicode char is wrong. An optional
   per-corpus dictionary covering the top-N most-common observed
   values — each validated against context on a sample — is future
   work; until that exists, deletion is the safest default.
4. **Paragraph reflow** (Case 8) — collapse soft-wrap sequences
   `[^\s]\t?\n[ \t]+[^\s]` → `\1 \2` inside paragraph blocks. Block
   boundaries: blank lines, `#` headings, `|` table rows, `---`
   separators, `>` blockquotes, list markers. Fixes the Docling
   column-width breaks in docs like `00264_pct0090_…`.
5. **LaTeX-segment repetition crop** (opt-in, already landed in
   `latex_module.rs` commit 071b709) — pass-through unless explicitly
   enabled by the cleaner driver. Uses OCR-style char-run / line-run
   detection per `$$…$$` span.
6. **Existing passes** — per-char filter, line-drop, normalize.

Each stage writes accounting to the four-way attribution buckets:

- HTML-entity decode, PUA decode, GLYPH substitution, paragraph
  reflow → `chars_dropped_by_normalization` (they're multi-char
  transformations, not single-char strips).
- Per-char filter → `chars_dropped_by_per_char_filter`.
- Line-drop → `chars_dropped_by_line_drop`.
- Normalize (existing) → `chars_dropped_by_normalization`.

Counters stay **additive** — no weighted aggregation
(`feedback_split_counters_per_signal_type.md`).

---

## Module ownership (one module per text-type)

### `charset_module.rs` — charset-ratio accounting

Owned concerns:

- Line-level exclusion mask (`is_format_scaffolding_line`).
- Per-ratio numerator/denominator bookkeeping.
- Emit `charset_{greek,moji,punct,numeral}_ratio`.

Wave-2 additions:

- **Case 10, numerals counter**: add `charset_numeral_ratio =
  digits / non_ws` using the same line-exclusion mask as the other
  ratios.
- **Case 13, bucket refactor**: move from the current catch-all
  `moji_residue = latin1_supp + ipa_extensions + pua + specials_fffd +
  latin_ext_b` to a finer partition:
  - `moji_residue_ratio` → keep as true mojibake: `ipa_extensions`,
    `specials_fffd`, `latin_ext_b`, `pua` excluding Adobe-Symbol
    range (U+F020..U+F07F decoded out by stage 0-2 before counting).
  - `legit_extras_ratio` → new bucket for `latin1_supp` chars that
    are legitimate Greek/EU punctuation: `« » · § ° ® © ™ €` and
    similar. Drops the current false-positive load on moji.
  - `bullet_marker_ratio` → new bucket for `U+2022 • U+25A0 ■
    U+25BA ► U+2666 ♦ U+2713 ✓` etc. Bullet-like markers.
- **Case 5, single-line LaTeX exclusion**: switch the block-math
  state-machine to consume `latex_module::find_dollar_dollar_spans`
  (already implemented there) so single-line `$$…$$` is excluded
  from ratio counting too. This fixes `02247_pct0024_…` where 261
  single-line LaTeX lines inflated punct ratio from 0.094 (actual
  prose-only) to 0.201.
- **Cases 10/11, code-block exclusion**: extend the mask with:
  - Fenced code blocks (` ``` ` / ` ~~~ ` state-machine, mirrors the
    `$$` tracker).
  - Indented code blocks (line starts with 4+ spaces or a tab).
  - Inline backtick spans (` `…` ` stripped like `strip_html_comments`).
  - XML/HTML source density heuristic (lines with ≥3 tags AND tag-
    char-density > 40%, conservative threshold per Case 11).
- **Case 3, URL span exclusion**: add `strip_url_spans(line)` that
  removes `https?://\S+`, `ftp://\S+`, `www\.\S+`, `[^\s]+@[^\s]+\.[^\s]+`
  (terminator `[^\s)>\],;]+` to keep sentence punct). Apply before
  per-char count. URLs stay in the output text — strip is ratio-only.

All exclusions share the same line-scan — no per-exclusion re-scan.
Tests: each exclusion type gets a dedicated `excludes_X` test, plus
combined regression tests.

### `latex_module.rs` — already landed (commit 071b709)

Owned concerns:

- `find_dollar_dollar_spans(text)` — multi-line AND single-line.
- `detect_repeated_char_cut` / `detect_repeated_lines_cut` ports from
  the OCR utils.
- `crop_latex_repetitions(text, enable, char_threshold, line_threshold)`.
- PyO3 binding `crop_latex_repetitions_py`.

Wave-2 extensions (not yet implemented):

- Add `\begin{env} … \end{env}` detector (extends span types).
- Add inline `$ … $` detector (tighter terminator than the `$$`
  variant to avoid consuming legitimate `$` as currency).
- Charset module consumes `find_dollar_dollar_spans` instead of its
  own inline `$$` toggle (deduplicates detection).

### Table modules — refactor for single-source-of-truth

Current scatter (Case 5 notes):

- `charset_module::is_format_scaffolding_line` detects `|…|` rows
  for ratio exclusion.
- `table_analysis_module::core_detect_malformed_tables` scans for
  GFM tables + flags issues.
- `table_remover_module::remove_tables_from_content` removes by
  line range.

Wave-2 refactor:

- Introduce `table_region_module::detect_table_regions(text) -> Vec<TableRegion>`
  as the single detection pass. Output exposes:
  - line range
  - column count (after parsing header / separator)
  - issues: header mismatch, body row mismatch, pseudo-table (Case 5)
  - pseudo-table sub-types: single-column TOC wrapper; >80% empty cells
- Consumers: ratio exclusion (charset module), malformed detector
  (analysis module), remover (remover module), PSEUDO-TABLE gate
  between detect and remove.
- **Pseudo-table unwrap** (Case 5 new): before passing to the
  minimizer, detect:
  - All rows 1-column AND ≥50% of rows match TOC pattern
    (`…\.{4,}\s*\d+\s*` or `…\s+\d+\s*$` at row end) → unwrap: strip
    `|` + separator row; emit cell contents as plain prose lines.
  - ≥80% empty cells across the whole table → drop table, emit
    `<!-- table-removed -->` marker.
- **TOC-line detector outside tables** (Case 5 augmentation): also
  exclude lines matching `^(.{0,160})\s*\.{4,}\s*\d+\s*$` (classic
  TOC entry with dot-leaders) even without `|` wrappers.

### `normalize.rs` — additions

- HTML-entity decode pass (Case 4):
  - named: `&amp; &lt; &gt; &quot; &apos; &nbsp;`
  - numeric: `&#\d+;` and `&#x[0-9a-fA-F]+;`
  Position: first pass in the normalize phase, but INSIDE the
  per-char-after-line-drop ordering (normalize still comes after
  cleaning per `feedback_normalize_after_cleaning.md`).
- PUA-Symbol decode pass (Case 10) — static dict of ~80 entries for
  the Adobe Symbol font positions. See case notes for the mapping.
- GLYPH marker deletion (Case 7, per Case 9 correction) — regex
  `GLYPH<\d+>` → empty (also `/uni[0-9A-Fa-f]{4,6}` and
  `/g(?:id)?\d+`). Optional per-corpus mapping dict deferred until
  a validated top-N mapping exists.
- Paragraph reflow pass (Case 8) — soft-wrap collapse inside
  paragraph blocks.
- Soft-hyphen strip (Case 13) — `\xad` silent removal; accounted in
  `chars_dropped_by_normalization`.

### `cleaning_module.rs` — allowed-script set widening (Case 12)

Widen `common_symbols` script family to include:

- Math operators `U+2200..U+22FF`
- Arrows `U+2190..U+21FF`
- Geometric shapes `U+25A0..U+25FF` (for legitimate bullets/markers;
  works alongside the new `bullet_marker_ratio` bucket)
- Super/subscripts `U+2070..U+209F`
- Letterlike symbols `U+2100..U+214F` (covers `™`, `ℓ`, etc.)

**Why**: CS/math/bilingual papers carry these as meaningful content
(see `01542`). Current strip rate of 18% on such docs is a bug,
not a feature. Post-widening we expect CS-paper per_char_filter to
drop to 1–2%.

---

## Tests (per module; all must pass before rebuild)

New unit tests required:

- `charset_module`:
  - `numeral_ratio_counts_ascii_digits_only`
  - `numeral_ratio_excludes_code_fenced_digits`
  - `numeral_ratio_excludes_indented_code_digits`
  - `numeral_ratio_excludes_inline_backtick_digits`
  - `strip_url_spans_removes_http_url`
  - `strip_url_spans_removes_email`
  - `strip_url_spans_stops_at_sentence_terminator`
  - `excludes_inline_double_dollar_math_from_counts` (consumes
    `latex_module::find_dollar_dollar_spans`)
  - `excludes_fenced_code_block_from_counts`
  - `excludes_indented_code_block_from_counts`
  - `excludes_inline_backtick_span_from_counts`
  - `legit_extras_ratio_counts_greek_punct`
  - `bullet_marker_ratio_counts_geometric_shapes`
  - `moji_residue_ratio_no_longer_includes_greek_punct_regression`
- `normalize`:
  - `html_entity_decode_named`
  - `html_entity_decode_numeric_decimal_and_hex`
  - `pua_symbol_decode_greek_letters`
  - `pua_symbol_decode_math_operators`
  - `glyph_marker_substitutes_to_bullet`
  - `paragraph_reflow_joins_soft_wrapped_lines`
  - `paragraph_reflow_preserves_heading_breaks`
  - `soft_hyphen_stripped_silently`
- `table_region_module` (new):
  - `pseudo_table_toc_wrapper_unwrapped_to_plain_lines`
  - `pseudo_table_mostly_empty_cells_removed`
  - `real_table_preserved`
- `cleaning_module`:
  - `cs_paper_regression` — real sample from `01542_pct0182_…`, confirm
    per_char_filter drop ≤ 3% after widening + PUA decode + entity
    decode.

Existing tests must continue to pass. One pre-existing failure
(`table_remover_module::test_empty_content_with_remove_op`) is not in
scope — flagged separately.

---

## Order of implementation (suggested)

Smallest, highest-impact first:

1. **Case 4 (HTML-entity decode)** — touches `normalize.rs` only; tiny;
   unblocks Case 7 and cleans up the `&amp;` literal in countless docs.
2. **Case 10a (PUA Symbol decode)** — static dict; no detection
   complexity; massive impact on math/stats docs (18% → 2% per_char
   strip on `01500_pct0033_…`).
3. **Case 12 (widen common_symbols)** — single allowed-script list
   change; no new code paths; rescues CS papers.
4. **Case 5 (single-line LaTeX detection)** — use `latex_module::find_dollar_dollar_spans`
   from `charset_module`. Fixes `02247_pct0024_…` punct ratio 0.201
   → 0.094.
5. **Case 13 (moji bucket refactor + numeral ratio + code-block
   exclusion + URL strip + soft-hyphen strip)** — large mechanical
   change to `charset_module`; needs all the new tests above.
6. **Case 7 (GLYPH marker substitution)** — simple regex; depends on
   Case 4 having landed.
7. **Case 8 (paragraph reflow)** — the trickiest regex-wise; guard
   carefully with negative tests so it doesn't destroy intentional
   structure.
8. **Case 11 (XML-source detection)** — heuristic; last because
   precision matters and needs a small sample study.
9. **Table-region refactor + pseudo-table unwrap (Case 5
   continuation)** — biggest structural refactor; save for last so
   all the smaller wins land without being blocked.

After each item lands: rebuild wheel, re-run cleaner on full corpus,
regenerate samples, user review, iterate. **No threshold changes
without explicit user request**.

---

## Open user questions (parked until answered)

- **Case 7** (*resolved 2026-04-23*): default = **delete**. GLYPH<N>
  is font-local; no safe global substitution exists. Per-corpus
  dictionary is optional future work.
- **Case 9** (*resolved per Case 7 above*): skip the dictionary;
  default = delete-all for `GLYPH<N>` / `/uniXXXX` / `/gN` markers.
  Per-corpus dictionary remains an option the user can opt into
  after wave 2 lands.
- **Case 13**: does `£` at 2,805 occurrences in a Greek thesis merit
  its own investigation? Could be font-substitution mojibake for `λ`.
  Flagged but not acted on.

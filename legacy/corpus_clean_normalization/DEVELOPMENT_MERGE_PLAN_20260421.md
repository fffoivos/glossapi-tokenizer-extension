# Development-merge plan — 2026-04-21

Goal: move `origin/development` → `origin/development + current
specifications` *without re-bloating the code*. Every item promoted has
to be traceable to the strict scope in `STEERING_20260421.md`
(cleaning vs normalization). Anything that doesn't fit gets dropped or
deferred.

All paths are in `/home/foivos/glossAPI-development/`.

---

## Inventory A — what's already on `origin/development`

### CLEANING (confirmed by `git show origin/development:…`)

- `rust/glossapi_rs_cleaner/src/cleaning_module.rs`
  - `SCRIPT_SETS` (latin, greek, french, spanish, punctuation, numbers,
    common_symbols, unusual). `unusual` covers Latin-1 Supplement,
    Latin Extended-A/-B/Additional, IPA Extensions, Coptic,
    Cyrillic + Supplement.
  - `BAD_LINE_AC` — glyph/font artefact triggers (`glyph<c=`, `GLYPH<`,
    `MS-Bold-`, `font=/`, `FontName=`, plus regex variants).
  - `strip_tags_custom` — memchr-based HTML/XML tag removal,
    comments preserved.
  - `core_clean_text` — Step 4.1 HTML entity decode → Step 5.1 tag strip
    → char-level filter with `$$`-math awareness → `has_decoded_glyph_font_artefact`.
  - `clean_text` (PyO3-exported).
  - Mojibake repair (existing, same file).

- `rust/glossapi_rs_cleaner/src/table_analysis_module.rs`
  - `TABLE_SEPARATOR_REGEX`, `TABLE_ROW_REGEX`, `TableIssue` — detection
    + flagging only.
- `rust/glossapi_rs_cleaner/src/table_remover_module.rs`
  - Range-based table **removal** driven by CSV (not canonicalization).

### NORMALIZATION

- **Nothing at this scope.** No `normalize.rs`. No run-length
  bucketing, no separator-line canonicalization, no GFM canonicalization,
  no ellipsis / malformed-entity rules.

### TESTS

- `tests/test_corpus_clean_enhancements.py` — includes
  `test_clean_ocr_writes_cleaned_markdown_with_combined_loop`
  (asserts `<table>` not in cleaned output) and related OCR-table
  dropping tests.

---

## Inventory B — what's ahead on `codex/token-noise-review-debug`

(23 commits ahead of `origin/development`, per `git log`.)

### B.1 — `rust/glossapi_rs_cleaner/src/normalize.rs` (new file, 1259 LOC)

Public functions:

- `fold_codepoint(ch) -> Option<&'static str>` — Math Alphanumeric
  Greek + Latin → regular equivalents. Drives `fold_line`.
- `fold_line(line) -> Option<String>`.
- `normalize_ellipsis_runs(line) -> Option<String>`.
- `bucket_run_length(n) -> usize` — `{0|1→n, 2→1, 3→3, 4..=20→5, _→20}`.
- `normalize_char_runs_tiered(line, target) -> Option<String>`.
- `normalize_dot_runs(line) -> Option<String>`.
- `normalize_whitespace_runs(line) -> Option<String>` — preserves leading
  indent.
- `normalize_malformed_entities(line) -> Option<String>`.
- `normalize_separator_line(line) -> Option<String>` — standalone rows.
- `scan_gfm_table_separators(text) -> HashMap<line_idx, canonical_row>`
  — **the GFM canonicalization we thought didn't exist — it does.**
  Parses each row; compares cell count against the header; if they match
  and the row is a valid GFM separator, rewrites it to minimal form
  (`| --- |` / `| :--- |` / `| :---: |` / `| ---: |` preserving alignment).
  Skips lines inside fenced code blocks.
- `parse_gfm_separator_row`, `count_gfm_row_cells` (private helpers).
- `drop_low_salvage_pages(original, cleaned, min_retention_ratio)` —
  page-salvage helper.
- `synthetic_page_line_ranges`, `is_markdown_header_line`,
  `count_nonwhitespace_in_range`, `is_code_fence_marker`.

### B.2 — `rust/glossapi_rs_cleaner/src/cleaning_module.rs` (+410 LOC)

New / changed items:

- `is_unicode_noise_char` — U+00AD, U+03A2, U+200B/C/D, U+200E/F,
  U+2060, U+FEFF, U+FFFD, `<0x20`, `0x7F..0x9F`, all three PUAs.
- `has_decoded_glyph_font_artefact` — extra glyph/font detection.
- `normalize_layout_leader_runs` — wraps `normalize::normalize_dot_runs`.
- `core_clean_text` gained:
  - Pre-pass: `normalize::scan_gfm_table_separators(text)` → line-indexed
    replacements applied during iteration.
  - Code-fence state tracking (`in_code_fence`).
  - Invisible-noise-char filter via `is_unicode_noise_char`.
  - Math-Greek folding via `normalize::fold_line`.
  - Dot-leader / whitespace-run / separator-line / ellipsis /
    malformed-entity calls.
- SCRIPT_SETS extended: polytonic Greek range made explicit; accented
  Greek set includes micro sign.

### B.3 — `rust/glossapi_rs_cleaner/src/lib.rs` (+6 LOC)

- Exports `clean_text` to Python.
- Registers `mod normalize`.

### B.4 — `rust/glossapi_rs_noise/` (the matcher, noise-metric crate)

- `noise_metrics.rs` (+967 LOC):
  - `export_token_category_debug_pages_internal`,
    `match_token_category_debug_text_internal` — matcher entry points
    for the review pipeline.
  - `TOKEN_CATEGORY_SPEC_CACHE`.
  - `bad_double` metric fix (only exact-length-2 illegal doubles;
    today's commit `713c0c1`).
- `src/lib.rs` (+117 LOC) — two new `#[pyfunction]` wrappers for the
  above.

### B.5 — Python surface

- `src/glossapi/corpus/phase_clean.py` (+240 LOC):
  - `_build_token_category_page_metric_row`,
    `_build_token_category_match_index_rows`,
    `Corpus.clean_token_category_debug` — review-infrastructure path
    that uses the matcher to export debug parquets.
- `src/glossapi/scripts/`:
  - `aggregate_token_category_reviews.py` (+456 LOC) — aggregator.
  - `build_token_category_review_bundle.py` (+574 LOC) — bundler (the
    file I just edited for continuous-context prompts).
  - `export_token_category_debug.py` (+71), `export_token_category_debug_parquet.py` (+184),
    `review_token_category_with_gemini.py` (+1429),
    `token_category_debug_common.py` (+289) — Gemini-driven
    review pipeline + helpers.

### B.6 — Tests

- `rust/glossapi_rs_cleaner/src/cleaning_module.rs` inline tests:
  - `core_clean_text_rejects_decoded_glyph_font_tags`
  - `core_clean_text_normalizes_long_dot_leaders_without_badness_penalty`
  - `core_clean_text_rejects_bare_glyph_codes`
  - `core_clean_text_strips_lrm_rlm_direction_marks`
  - `core_clean_text_strips_unicode_noise_chars`
  - `core_clean_text_normalizes_separator_line`
  - `core_clean_text_normalizes_gfm_table_separator` — ONE case:
    `| A | B |\n| :------- | -------: |\n| 1 | 2 |` → `| :--- | ---: |`.
  - `core_clean_text_skips_fenced_code_block`
  - `core_clean_text_folds_math_italic_latin`
  - `core_clean_text_normalizes_toc_whitespace_leader_via_bucket`
  - `core_clean_text_collapses_ellipsis_runs`
  - `core_clean_text_preserves_polytonic_greek`
  - `core_clean_text_preserves_non_greek_latin_scripts`
  - `core_clean_text_folds_math_greek_to_plain_greek`
  - `core_clean_text_composite_roundtrip`
- `rust/glossapi_rs_noise/src/noise_metrics.rs` inline tests:
  - `bad_double_counts_only_exact_illegal_doubles`
  - `bad_double_ignores_allowed_greek_doubles`
  - `bad_double_ignores_long_expressive_runs`
- `tests/test_corpus_clean_enhancements.py` (+188 LOC):
  - `test_clean_token_category_debug_exports_synthetic_pages`
  - `test_build_token_category_review_bundle_materializes_cases`

---

## Keep / drop / refine decisions (mapped to scope)

### KEEP — in scope, ready to promote

**Cleaning rules:**
- `is_unicode_noise_char` with all 9 named codepoints + controls + PUAs.
- `has_decoded_glyph_font_artefact` — extra glyph detection.
- SCRIPT_SETS polytonic-Greek explicitness + micro-sign.

**Normalization rules:**
- `bucket_run_length` + `normalize_char_runs_tiered`.
- `normalize_dot_runs`, `normalize_whitespace_runs` (leading-indent-preserving).
- `normalize_separator_line`, `normalize_ellipsis_runs`,
  `normalize_malformed_entities`.
- `fold_codepoint` + `fold_line` — Math-Alphanumeric-Greek → Greek
  (and Math-Alphanumeric-Latin → Latin, if Apertus-checked).
- `scan_gfm_table_separators` — parser-validated GFM canonicalization
  (minimal form, alignment-preserving, code-fence-safe).
- `drop_low_salvage_pages` — page-salvage.

**Wiring:**
- `core_clean_text` pre-pass for GFM + code-fence state + per-line
  normalization hook.
- `clean_text` PyO3 export for per-row corpus-parquet use.

**Metric fix:**
- `bad_double` exact-length-2 counting.

### DROP — not cleaner/normalizer surface; keep out of the cleaner PRs

These are review-infrastructure; they belong to the `corpus_clean_normalization/`
pipeline, not to `glossAPI-development/Corpus.clean`. Do NOT bundle
them into cleaner promotions:

- `rust/glossapi_rs_noise/src/noise_metrics.rs` matcher entries
  (`export_token_category_debug_pages_internal`,
  `match_token_category_debug_text_internal`,
  `TOKEN_CATEGORY_SPEC_CACHE`) and the PyO3 wrappers.
- `src/glossapi/corpus/phase_clean.py` token-category-debug helpers.
- All `src/glossapi/scripts/*token_category*` and
  `*review_token_category*` files.
- `tests/test_corpus_clean_enhancements.py::test_clean_token_category_debug_*`.

These stay on a branch until the review pipeline design settles; they
are not wanted as permanent `Corpus.clean` surface.

### REFINE — ship, but only after the test matrix below is filled in

`scan_gfm_table_separators` currently has **one** test case (left+right
alignment, bloated hyphens). To call it complete we need the matrix in
the next section.

---

## GFM table auto-minimization — test matrix

Single existing test: `core_clean_text_normalizes_gfm_table_separator`
→ `| A | B |\n| :------- | -------: |\n| 1 | 2 |` collapses to
`| :--- | ---: |`.

That's not enough. Required additions, each one or two short
`#[test]` functions in `cleaning_module.rs` (or a new
`normalize::tests` module — preferred):

### Alignment coverage
1. **Default alignment** — `| A | B |\n| -------- | ------- |` →
   `| --- | --- |`. Must not inject colons.
2. **Left alignment** — `| A |\n| :------- |` → `| :--- |`.
3. **Right alignment** — `| A |\n| -------: |` → `| ---: |`.
4. **Center alignment** — `| A |\n| :------: |` → `| :---: |`.
5. **Mixed per-column alignment** — three cells: default / left / right
   / center — assert each column keeps the correct marker.

### Minimal-form preservation
6. **Already-minimal separator** — `| --- | --- |` → unchanged (no-op,
   same bytes, no spurious delta).
7. **Minimal with colons** — `| :---: |` → unchanged.

### Column-count & validity
8. **Mismatched cells** — header has 2 cells, separator has 3 cells →
   leave separator untouched (do NOT rewrite a malformed table).
9. **Single-cell separator** with matching 1-cell header →
   `| ------- |` → `| --- |`.
10. **No leading/trailing pipe** — `A | B\n------- | -------` →
    either rewrite consistently or leave alone; document the choice.
11. **Extra whitespace inside cells** — `|   --------   |   ------   |` →
    canonical `| --- | --- |`.

### Safety — must NOT rewrite
12. **Inside fenced code block** — backtick-fenced block containing a
    table-looking line → untouched.
13. **Inside tilde-fenced code block** (`~~~`) — same.
14. **Standalone `----` line (no pipes)** — must go through
    `normalize_separator_line`, NOT GFM canonicalization. Regression
    test: `parse_gfm_separator_row` returns `None` on pipeless input.
15. **Pipe line that isn't a separator** (e.g. `| col1 | col2 |` with
    no separator row following) → untouched.
16. **Separator row without a header row above** (line 0) — untouched.

### Semantics preservation
17. **Badness neutrality** (already present in the one test) — for any
    rewritten separator, `kept_chars == original_chars` of that row.

### Repeated special chars INSIDE tables (future item — not yet
### implemented)
18. **Placeholder** — documents that today `| \_\_\_\_\_\_\_\_\_\_\_\_ |`
    table cells are NOT normalized. When the inside-table rule lands,
    add test cases like: footnote-divider row with 24-char `\_` run →
    canonical collapsed form that keeps the cell valid and the column
    count unchanged.

---

## Suggested migration order (small, reviewable PRs)

1. **Metric fix PR** — `bad_double` exact-length-2 change
   (`rust/glossapi_rs_noise/src/noise_metrics.rs`). Pure bugfix, tests
   included. Low risk.
2. **Invisible-char strip PR** — `is_unicode_noise_char` + its wiring
   into `core_clean_text` + inline tests
   (`core_clean_text_strips_unicode_noise_chars`,
   `core_clean_text_strips_lrm_rlm_direction_marks`). Self-contained.
3. **`normalize.rs` foundation PR** — new file with
   `bucket_run_length`, `normalize_char_runs_tiered`,
   `normalize_dot_runs`, `normalize_whitespace_runs`,
   `normalize_separator_line`, `normalize_ellipsis_runs`,
   `normalize_malformed_entities`, `fold_codepoint`/`fold_line`,
   `is_code_fence_marker`. Plus the `core_clean_text` wiring changes
   for per-line normalization + code-fence state. Plus all inline tests
   except the GFM one. One PR because the pieces depend on each other.
4. **GFM canonicalization PR** — `scan_gfm_table_separators` +
   `parse_gfm_separator_row` + `count_gfm_row_cells` + the
   `core_clean_text` pre-pass wiring + the **full test matrix** from
   the section above (not just the one existing case). Separate PR
   because GFM has the biggest semantics blast radius.
5. **Page-salvage PR** — `drop_low_salvage_pages` +
   `synthetic_page_line_ranges` + helpers. Separate because the
   retention ratio needs its own calibration history.
6. **PyO3 export PR** — `clean_text` export + `lib.rs` module hookup.
   Small.

Review-infrastructure (matcher entries, token-category scripts,
review-bundle builder) stays on branch, not merged into
`development` — it lives in the review pipeline, not in
`Corpus.clean`.

---

## How to apply

- Before opening each PR, re-grep for "already on development" claims
  and verify against `git show origin/development:<path>` — don't trust
  stale notes.
- Each PR must include its inline tests; no PR lands without tests for
  every public function it adds.
- After PR 4 (GFM), re-run the fresh BPE on the cleaned corpus and
  confirm the expected reduction in `hyphen_runs` / `table_border`
  tokens in the bad-token inventory.

---

## Planned features added 2026-04-21 end-of-day

Two additions to the feature list, both **future** (not on branch, not
on development):

### Feat-A. Font-name contextual evidence for glyph detection

Extend BAD_LINE_AC + surrounding detection with bare font-name
literals as evidence of broken PDF font-switch tags whose `font=/`
prefix was stripped in Step 5.1.

- **Candidate literal set**: `Palatino`, `Linotype`, `+Palatino`,
  `Helvetica`, `Arial`, `TimesNewRoman`, `Times-Roman`, `Courier`,
  `CourierNew`, `Symbol`, `ZapfDingbats`, `CMU`, `CMR`, `CMBX`,
  `LMRoman`, `LMSans`, `STIX`, `+Garamond`, `Minion`. Needs an
  empirical inventory pass on the bad-token / post-clean samples
  before pinning the final list.
- **Placement on grid**: cleaning × direct, *if* we go with direct
  rejection. Cleaning × review-gated (evidence weighting pipeline),
  *if* we go with weighted.
- **Open design question**: direct line-rejection vs. evidence-weighted
  (font-name triggers only when co-occurring with another glyph
  signal — PUA char, `GLYPH<`, unusual ASCII-digit bigrams). User
  framing "contextual evidence" argues for evidence-weighted.
  Direct is simpler and cheaper; weighted is safer against false
  positives (e.g., an article that genuinely names a font). Decide
  after a 30-case sampling wave.
- **Test coverage when it lands**:
  - Positive: `Παλιό δοκίμιο σχετικά με την τυπογραφία.\nPalatino\n<PUA char here>` → line rejected as glyph-noise.
  - Negative (weighted path): `Ο Παύλος προτίμησε τη Palatino για το βιβλίο του.` → **not** rejected (prose context, no other glyph signal).
  - Negative (direct path, if chosen): same as positive input with font name on its own line → rejected.

### Feat-B. Unified dot + ellipsis bucketing

Replace the two separate `normalize_dot_runs` + `normalize_ellipsis_runs`
rules in `normalize.rs` with a single unified pass:

- **Rule**:
  1. Every `…` (U+2026) counts as exactly 3 dots when measuring run length.
  2. Every `…` is rewritten to three ASCII dots (`...`) in the output.
  3. The resulting all-ASCII dot run is fed through the existing
     `bucket_run_length` (targets `{1, 3, 5, 20}`).
- **Placement on grid**: normalization × direct.
- **Target BPE category**: `dot_ellipsis_mixed` — 10 inventory tokens
  (`….`, `…..`, `……..`, `……….`, `………..`, `…...`, …) that slip through
  today because neither rule alone covers them.
- **Test matrix** (add to `normalize::tests` or inline in
  `cleaning_module.rs`):
  - `….` (4 dots) → `.....`
  - `……` (6 dots) → `.....`
  - `……..` (8 dots) → `.....`
  - `…` alone (3 dots) → `...` (no-op-ish: maps to 3)
  - `..` (2 dots) → `.` (bucket `2→1`)
  - `.….` (4 dots from 1 + 3) → `.....`
  - Run of 32 `…` (= 96 dots) → `....................` (20)
  - `ab…cd` (not a run of length ≥ 2) → unchanged in content beyond `…`→`...` expansion? **Open question**: do we expand stray single `…` chars even when not in a dot run? User said "ellipsis always counts for 3 dots in our scheme"; cleanest implementation is always-expand `…`→`...`, then bucket. Write the test for single `…` → `...` to pin this decision.
  - `…` inside a GFM table cell → still expanded (cells are not
    guarded by the code-fence rule). **Confirm** that this is
    desired; if not, add a table-cell guard.
- **Migration note**: when this feature lands, delete
  `normalize_ellipsis_runs` — it's superseded.

### Feat-C. Bigram-token denoising (corrected terminology)

The user's "bigram" task is vocab-level cleanup of length-2 BPE tokens
in the fresh GlossAPI-only discovery vocab — NOT the Greek-orthography
`is_invalid_bigram_pair` / `bad_double` runtime metrics (those are
unrelated, Greek-only).

**Evidence pool**:
- `short_le2_decoded.json` — 3,690 tokens (712 length-1, 2,978
  length-2). Of the 2,978 bigrams:
  - ~1,954 ASCII English bigrams (`'in'`, `'on'`, `'er'`) — **not noise**.
  - ~202 space+replacement-char (`' �'`) — already stripped by
    `is_unicode_noise_char` (U+FFFD).
  - ~77 pure `'��'` pairs — same strip.
  - ~233 punctuation pairs (`'--'`, `'..'`, ` |`) — covered by
    run-length bucketing.
  - ~57 space+Latin-1 (` «`, ` »`, ` ·`) — legitimate typography.
  - ~21 **script-salad mimics** — Latin-Ext-A + Latin-Ext-B pairs
    (`'ĮȚ'`, `'ȚĮ'`, `'Ƞȣ'`, `'ȠȪ'`, `'Țț'`, …). **This is the
    un-handled target.** They render as pseudo-Greek letters
    (`Ț` looks like `Τ`, `Ƞ` like `Η`, etc.) and appear in PDF-derived
    Greek text as OCR mis-classifications.
- `short_nonascii_latin_like_decoded.json` — 246 tokens (filtered
  subset of short_le2 restricted to non-ASCII-Latin). All present in
  short_le2.

**Gemini review evidence** (wave10_v5 pilot):
- 10 cases sampled from the 246 `short_nonascii_latin_like`.
- **10/10 is_noise:yes** (100%).
- Decision split:
  - 4× no_regex_needed — the individual char is already in SCRIPT_SETS
    `unusual` and gets stripped once it appears outside a bigram
    token boundary.
  - 4× existing_regex_extension — an existing BAD_LINE_AC / glyph
    regex can be extended.
  - 2× new_regex — a new pattern needed.
- `regex_inventory.jsonl` has only weak single-vote candidates
  (`\u01c2`, etc.) — not ship-ready.

**Proposed rule** (when it lands):

- **Script-salad detector**: flag any BPE token (or line at cleaning
  time) where Latin-Ext-A **and** Latin-Ext-B codepoints co-occur
  within a ≤3-char span. Rationale: monolingual Greek text should
  rarely mix these two extended Latin blocks. A bigram with one
  codepoint from each is a strong noise signal, distinct from plain
  Latin-Ext-A usage (Romanian alone) or plain Latin-Ext-B (isolated
  IPA chars). Per-char strip is already handled by `unusual` in
  SCRIPT_SETS — the detector adds line-level rejection for mixed-range
  tokens.
- **Placement on grid**: cleaning × direct (the criterion is
  deterministic once defined), after one more review wave confirms
  the ~21 mimic tokens are universally noise.
- **Test coverage** when it lands:
  - Positive: line containing `'ĮȚ'` between Greek chars → line
    rejected OR token-span stripped.
  - Positive: `'Ƞȣ'` alone → stripped.
  - Negative: pure Romanian prose (Latin-Ext-A only, no -B) →
    untouched (scripts are already out of `unusual`'s aggressive reach
    per 2026-04-21 policy).
  - Negative: an isolated `Ț` (Latin-Ext-A only) inside Greek
    text → may still be stripped by `unusual`, but not by this rule
    specifically. Proves the rule is additive, not replacement.

**What NOT to do**:
- Do not attempt to edit the BPE vocab directly as a "cleaning" step.
  The denoising happens at corpus-cleaning time; the fresh BPE is
  re-trained on cleaned text and the mimic bigrams don't recur.
- Do not widen `unusual` to aggressively strip entire Latin-Ext-A
  or -B ranges — per scope, script sets are frozen without evidence
  of full-range noise.

### Verification side-note — runtime vs. vocab (keep in mind)

The runtime `bad_double` / `invalid_bigram` metrics in
`rust/glossapi_rs_noise/src/noise_metrics.rs` (lines 127, 149) are
**Greek-orthography** checks on text spans. They have nothing to do
with Feat-C. Confirmed 2026-04-21: zero matches across the 3,690 +
246 non-Greek inventory tokens.

# Wave-3 cleaner patch plan — tokenizer-guided scope (2026-04-28)

This is the current implementation plan after the independent F1/F2
vocab review plus a full F1 train-corpus scan. It narrows the broader
`corpus_clean_normalization/NORMALIZATION_DESIGN_20260420.md` findings
into the rules we should actually land next.

## Evidence used

- F2 tokenizer pulled locally to
  `tokenizer_analysis/inspection/F2_glossapi_plus_hplt_70_30_50k_wave2_20260426/tokenizer.json`.
- Independent tokenizer analysis artifacts:
  `tokenizer_analysis/inspection/wave2_bad_token_analysis_20260428/`.
- Full F1 train scan:
  `/home/foivos/runs/wave2_20260426/splits/glossapi_only/exports/train.parquet`
  on the gcloud instance, 310,019 docs / 60,825,820,152 chars.

High-signal corpus findings:

| pattern family | affected docs | decision |
|---|---:|---|
| table separator fragments | 254,640 (82.14%) | implement |
| intentional HTML comment placeholders | 240,006 (77.42%) | keep |
| dot leader runs | 192,254 (62.01%) | implement |
| long dash runs | 159,256 (51.37%) | implement |
| escaped Markdown runs | 27,683 (8.93%) | implement |
| long equal / underscore / asterisk / hash / slash / pipe runs | 0.08% to 2.20% | implement |
| bare `GLYPH` / glyph-name residue | 735 to 1,566 docs | implement, narrowly |
| soft hyphen / non-newline controls | 225 (0.07%) | verify code-fence path |
| mojibake markers | 27,861 (8.99%) | defer |
| Cyrillic / homoglyph markers | 8,267 (2.67%) | defer |

The HTML comments counted here are expected placeholders such as
`<!-- image -->`, `<!-- text-missing -->`, and
`<!-- formula-not-decoded -->`. They are not cleanup targets.

## Wave-3 implementation scope

### 1. Run quantization first

Primary target: the run families that dominate both vocab and corpus
frequency.

Implement / correct:

- `bucket_run_length`: make the ladder a true floor. In particular,
  length `4` becomes `3`, not `5`.
- Extend run quantization beyond dots / underscores to dash, equal,
  asterisk, hash, tilde, slash, backslash, pipe, exclamation, percent,
  at, caret, and standalone accent/tonos runs.
- Keep semantic Markdown / URL contexts protected:
  ATX `#{1,6}`, list markers, blockquotes, code fences, math
  delimiters, URL protocol/path spans, and intentional placeholder
  comments.

Code placement:

- `/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner/src/normalize.rs`
  for the bucket and generic run helpers.
- `/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner/src/cleaning_module.rs`
  for the existing line-cleaning normalization chain.
- `/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner/src/md_module.rs`
  for the Markdown canonicalization path.

### 2. Generalize escaped Markdown run handling

Current code handles escaped underscores. Generalize that same pattern
to `\_`, `\*`, `\-`, `\#`, `\.`, `\=`, and `\~` outside code fences:
first unescape the run character, then apply the normal run bucket.

Code placement:

- Replace or generalize `normalize_escaped_underscore_runs` in
  `normalize.rs`.
- Keep the call adjacent to the existing normalization chain in
  `cleaning_module.rs` and `md_module.rs`.

### 3. Markdown table / heading / setext run quantization

Table separator rows are the largest corpus-wide signal. Use the
Markdown-aware code path for Markdown constructs rather than trying to
fix them with blind regexes.

Code placement:

- `md_format.rs`: table separator emission and setext underline
  emission.
- `md_module.rs`: ATX heading guard. Keep depths 1–6; quantize 7+ as
  noise/separator.

### 4. Minimal code-fence cleanup for impossible noise

Do not rewrite code blocks broadly. Do allow already-decided character
cleanup inside fenced code when the character has no useful code
semantics:

- strip soft hyphen and non-newline C0/C1 controls;
- fold MICRO SIGN `µ` to Greek `μ` if sampling confirms this is how it
  survived into vocab.

Code placement:

- Existing fenced-code branch in `cleaning_module.rs` around the early
  `in_code_fence` pass-through.

### 5. Glyph residue belongs in the existing glyph Rule A/B machinery

Do not create a separate glyph-cleaning subsystem. Extend
`apply_glyph_span_strip_and_rule_b` so structured glyph tags, bare
`GLYPH` runs, and high-confidence PDF/PostScript glyph-name residue all
share the existing count / coverage accounting and line-drop threshold.

Land only the high-confidence cases now:

- `glyph[...]` and `GLYPH(...)` style structured variants;
- repeated or standalone bare `GLYPH`;
- `/hyphenminus`;
- `/ellipsis` and `/elipsis` when glyph-like or repeated/leader-like;
- `/period`, `/comma`, `/space`, `/colon` in glyph-like contexts.

Avoid for wave 3:

- generic single-letter glyph names like `/A` or `/m`;
- `/pi`, `/alpha`, and other semantic-looking names without context;
- slash acronyms, URL path components, units, and legal/document paths.

Code placement:

- `cleaning_module.rs`: extend `PDF_GLYPH_NAME_REGEX` and
  `apply_glyph_span_strip_and_rule_b`.
- Existing tests that assert bare `GLYPH` is preserved should be
  updated to reflect the new bounded exception to the old
  "no-bare-words" rule.

## Deferred from this wave

Mojibake repair and Cyrillic / homoglyph folding are deliberately not
wave-3 implementation work. They have enough signal to deserve a
calibrated test set, but not enough time-safe evidence to land now
without false-positive risk. Tracking issue:

- https://github.com/eellak/glossAPI/issues/99

Line-level math-fence cleanup and page-marker footer cleanup also stay
deferred.

## Suggested implementation order

1. Run quantization + escaped Markdown runs + unified dot/ellipsis.
2. Markdown table / setext / ATX quantization.
3. Minimal code-fence impossible-noise cleanup.
4. Narrow glyph Rule A/B extension.
5. Re-clean a sample, train/analyze fresh tokenizers, then decide
   whether the deferred mojibake/homoglyph work is worth a calibrated
   follow-up.

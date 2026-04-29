# Pre-Restart Code And Analysis Checkpoint - 2026-04-28

This checkpoint was written before restarting the full wave-3 workflow after
the HPLT badness-filter issue was caught. It covers local and instance code
state relevant to the restart, including uncommitted changes.

## Current Safety State

- Instance: `foivos@35.204.86.203`.
- Relevant run root: `/home/foivos/runs/wave3_20260428/production`.
- Previous production wave-3 run was aborted on `2026-04-28T09:20Z`.
- Abort marker:
  `/home/foivos/runs/wave3_20260428/production/ABORTED_2026-04-28_HPLT_FILTER_MISSING.txt`.
- Reason: HPLT source selection used `HPLT/ell_Grek_ge8_no_mt_clean60`
  (quality bin >= 8 by dataset construction) but split export still allowed
  null `greek_badness_score` rows. That made the HPLT-only and 70/30 tokenizer
  inputs invalid for final judgment.
- F1 GlossAPI-only and F2 HPLT-only tokenizers from the aborted run remain
  useful diagnostics, but they are not final artifacts.
- No full production restart has happened after the fixes below as of this
  document.

## Cleaner Repo: `glossAPI-development`

Branch:

- `/home/foivos/glossAPI-development`
- `cleanup/cleaner-pipeline-20260425`

Relevant changed files:

- `rust/glossapi_rs_cleaner/src/cleaning_module.rs`
- `rust/glossapi_rs_cleaner/src/normalize.rs`
- related earlier wave-3 cleaner files already dirty in this branch:
  `charset_module.rs`, `latex_module.rs`, `md_module.rs`,
  `md_format.rs`, `md_format_surgical.rs`, `md_verify.rs`,
  `cmark_gfm_oracle.rs`, `lib.rs`.

Cleaner changes now in scope:

- Added the bounded bare-glyph exception:
  `GLYPH`, `GLYPHGLYPH`, `GLYPHGLYPHGLYPH`, etc.
- Kept this inside the existing PDF glyph artifact family, not as a generic
  normalization rule.
- Structured glyph forms remain in the same rule family:
  `GLYPH<...>`, `GLYPH(...)`, `glyph[...]`, `glyph<c=...,font=/...>`,
  `<c=...,font=/...>glyph`, PDF font subsets, `/uniXXXX`, `/gN`, `/gidN`.
- Selected PostScript glyph literals remain Rule A literals and contribute to
  the same count/coverage line-drop gate.
- Glyph/PostScript spans are stripped inline in normal prose.
- Whole-line removal still uses the existing threshold:
  combined Rule A + Rule B count >= 10 and coverage >= 0.09.
- Glyph/PostScript matching is guarded from:
  fenced code blocks,
  `$$...$$` math context,
  HTML comment spans and marker comments,
  URL/path-like tokens.
- This means strings like `/g123`, `/uni03B1`, `/space`, or `GLYPHGLYPH`
  inside a URL are preserved and not counted.
- Added impossible-noise cleanup inside fenced code blocks only for things that
  are never semantically meaningful there: soft hyphen/control noise and
  micro-sign folding.
- Kept mojibake/Cyrillic homoglyph folding out of this pass.
- Kept broad decorative-symbol stripping out of this pass.

Relevant cleaner analysis decisions:

- `GLYPH...` belongs with the glyph/PDF artifact rule because it is PDF
  extraction residue and already has span-strip plus line-drop accounting.
- It does not belong in markdown Phase A or normalization.
- PostScript glyph names should apply only in normal prose context.
- URL protection needs to be span-level because the false positive is usually
  a glyph-like substring inside one token, not a whole URL line.
- HTML comments are intentional corpus memory markers and must be preserved.

Cleaner validation:

- Local focused Rust test:
  `cargo test -p/glossapi_rs_cleaner equivalent via crate dir: cargo test glyph -- --nocapture`
  result: 9 passed.
- Local full cleaner test:
  `cargo test -- --nocapture`
  result: 382 passed, 0 failed, 3 ignored.
- Instance focused Rust test:
  `cargo test glyph -- --nocapture`
  result: 9 passed.
- Instance PyO3 rebuild:
  `PATH=$HOME/.cargo/bin:$PATH VIRTUAL_ENV=/home/foivos/venvs/glossapi-corpus-clean /home/foivos/venvs/glossapi-corpus-clean/bin/python -m maturin develop --release -m rust/glossapi_rs_cleaner/Cargo.toml`
  result: installed `glossapi_rs_cleaner-0.1.1` into the production venv.

## Tokenizer Extension Repo

Branch:

- `/home/foivos/Projects/glossapi-tokenizer-extension`
- `codex/cleaner-iteration-subproject-20260423`

Relevant changed files:

- `glossapi_corpus_cli/cli.py`
- `glossapi_corpus_cli/pipeline.py`
- `glossapi_corpus_cli/text_dedup.py`
- `subprojects/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`
- `subprojects/01_2_training_dataset_mix/scripts/create_real_subset_for_wave3_smoke.py`
- `subprojects/01_2_training_dataset_mix/scripts/wave3_orchestrate.py`
- `subprojects/01_2_training_dataset_mix/examples/hplt_only.json`
- `subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json`
- `subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json`
- `subprojects/02_1_tokenizer_experiments/scripts/train_continuous_bpe_tokenizer.py`
- `glossapi_corpus_cli/continuous_bpe.py`
- tests including `tests/test_export_text_budgeted_splits.py`,
  `tests/test_pipeline.py`, `tests/test_continuous_bpe.py`.

Pipeline efficiency changes:

- Added CLI stages:
  `mix-prepare-selected-input` and `mix-build-from-selected-input`.
- The selected input can now be materialized once after quality filters and
  existing dedup overlay, then reused for multiple mix builds.
- Added a DuckDB summarizer for mix outputs, with PyArrow/Pandas fallback.
- Added row-group sizing to split export so continuous-BPE workers have enough
  row groups to parallelize over.
- Wave-3 orchestrator drives:
  selected input -> three mixes -> text splits -> F1/F2 discovery tokenizers
  -> C1 continuous tokenizer.
- Wave-3 orchestrator is resumable by output markers and intentionally reuses
  existing builder-facing dedup metadata. It does not rerun dedup clustering.

Correctness changes after the HPLT catch:

- `export_text_budgeted_splits.py` now fails closed by default:
  missing/null/empty `greek_badness_score` is rejected.
- `mojibake_badness_score` also fails closed when `--mojibake-lte` is active.
- `try_cast(... AS DOUBLE)` is used so empty strings and non-numeric score
  values become null and are rejected.
- Added explicit debug escape hatch:
  `--allow-missing-badness-scores`.
- Production wave-3 orchestrator does not pass that escape hatch.
- Split summary now records:
  `allow_missing_badness_scores`,
  `has_greek_badness`,
  `has_mojibake_badness`,
  `row_group_size`.
- `create_real_subset_for_wave3_smoke.py` was corrected to fail closed on null
  badness values too.

Metric-preservation change:

- `batch_score_missing_quality` no longer refuses to score a row just because
  `quality_method` is already set.
- If `greek_badness_score` is missing, the row is scored.
- Existing metric values are preserved individually:
  `greek_badness_score`, `mojibake_badness_score`, `latin_percentage`,
  `table_ratio`, `polytonic_ratio`, `len_greek`, `greek_percentage`,
  `quality_method`, `reevaluated_at`.
- Important limitation: the current Rust noise scorer provides Greek badness
  and related script/table/polytonic metrics, not a new mojibake score. Existing
  `mojibake_badness_score` is preserved; rows with missing mojibake score still
  fail the production split filter.

HPLT selection rule for restart:

- HPLT must be from the quality-bin >= 8 slice:
  `HPLT/ell_Grek_ge8_no_mt_clean60`.
- HPLT must also pass cleaner-standard badness thresholds:
  `greek_badness_score < 60`
  and `mojibake_badness_score <= 0.1`.
- Missing/null/empty scores mean unscored and are not eligible.

Tokenizer-extension validation:

- Instance Python tests:
  `/home/foivos/venvs/glossapi-corpus-clean/bin/python -m pytest tests/test_export_text_budgeted_splits.py tests/test_pipeline.py -q`
  result: 26 passed.
- Tests added/covered:
  null/empty/high badness rejection in split export,
  required score columns by default,
  scoring missing Greek badness even with existing `quality_method`,
  preserving existing badness without rescoring,
  shared selected-input path matching direct streaming mix behavior.

## Pre-Production Amendment

Additional code changes made before the full production restart:

- `glossapi_rs_noise` now exposes in-memory text scoring APIs:
  `score_text_detailed(text)` and
  `score_texts_detailed(texts, n_threads=None)`.
- This avoids writing millions of temporary markdown files just to fill
  missing Greek badness for HPLT.
- A reusable production reclean driver was added at
  `subprojects/01_0_cleaning_iteration_and_thresholds/scripts/reclean_canonical_to_parquet.py`.
- The new driver preserves existing scores, fills missing
  `greek_badness_score` only for explicitly selected datasets, and leaves
  existing `mojibake_badness_score` untouched.
- It writes output parquets atomically through temp files, reports completed
  file tasks as they finish, and can resume by skipping existing outputs.
- It normalizes output Arrow types for canonical score columns, so HPLT files
  that originally carried an all-null Greek badness column can be rewritten
  with real `float64` scores.

Integration gate result before production restart:

- The strict tiny end-to-end workflow completed successfully at
  `/home/foivos/runs/wave3_20260428/integration_strict_tiny3/run`.
- All three split summaries had:
  `allow_missing_badness_scores=false`,
  `has_greek_badness=true`,
  `has_mojibake_badness=true`.
- All three arms produced train rows.
- The 70/30 mix contained no null Greek/mojibake scores and landed at
  `0.3000158` HPLT character share.

## Analysis So Far

- The aborted GlossAPI-only tokenizer is still useful for GlossAPI token
  inspection because its input was not the HPLT-null-filter failure case.
- The aborted HPLT-only and 70/30 inputs are not valid final analysis inputs
  because HPLT badness nulls were allowed.
- The largest immediate risk is not tokenizer training itself; it is feeding
  unscored HPLT rows into the analysis tokenizers.
- The cleaner improvements should reduce bad extraction artifacts without
  broad stripping:
  repeated `GLYPH...` terms,
  structured glyph residue,
  selected PostScript glyph literals,
  punctuation/dot/escaped-run bloat.
- Deferred issue already opened:
  `https://github.com/eellak/glossAPI/issues/99`
  for mojibake/Cyrillic/homoglyph calibration.

## Restart Plan After This Document

1. Run a real-subset end-to-end integration test with the corrected cleaner,
   strict score filtering, existing dedup metadata, split export, and tiny
   tokenizer workflow.
2. Verify the subset proves HPLT rows with missing badness scores do not pass.
3. Start a fresh production run root, not the aborted one, for the corrected
   full workflow.
4. Reclean GlossAPI and HPLT inputs with the rebuilt cleaner.
5. Preserve existing badness metrics where present; score missing Greek badness
   where needed; do not overwrite existing mojibake badness.
6. Do not rerun dedup clustering. Reuse the existing builder metadata bundle.
7. Build:
   - GlossAPI-only split/tokenizer,
   - HPLT-only split/tokenizer,
   - 70/30 GlossAPI + HPLT continuous tokenizer.
8. Bring tokenizer artifacts and summaries local.
9. Analyze vocab state and stop when good enough; open issues for non-urgent
   improvements.
10. Stop the instance after completion.

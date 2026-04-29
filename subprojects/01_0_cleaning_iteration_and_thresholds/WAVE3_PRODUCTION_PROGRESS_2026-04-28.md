# Wave 3 Production Progress - 2026-04-28

This document is the live progress record for the cleaner/tokenizer wave that started from
`HANDOFF_2026-04-28.md` and `PRE_RESTART_CODE_AND_ANALYSIS_2026-04-28.md`.

Current status as of the last check:

- Last checked:
  `2026-04-28T17:59:00+00:00`

- Production run root on the worker:
  `/home/foivos/runs/wave3_20260428/production_strict_v2`
- Clean canonical root on the worker:
  `/home/foivos/data/glossapi_work_wave3_20260428_strict_v2/canonical`
- GCP worker:
  `foivos@35.204.86.203`
- Active stage:
  production tokenizer wave complete; local artifact review complete
- C1 progress:
  `merge_completed_added=25600`, `merge_target_added=25600`,
  `current_vocab_size=156672`, `target_vocab_size=156672`
- The worker must remain running until C1 finishes, artifacts are copied locally,
  analysis is done, and then the worker is stopped.

## Plan Being Followed

1. Improve pipeline efficiency and parallelization for slow remaining phases.
2. Implement cleaner changes with focused and integration tests.
3. Reclean original datasets, preserving previous badness scores when present.
4. Reuse existing dedup metadata for the builder. Do not rerun dedup clustering.
5. Build strict train/val/test text splits:
   - GlossAPI-only
   - HPLT-only
   - GlossAPI + HPLT 70/30
6. Train:
   - fresh GlossAPI discovery tokenizer
   - fresh HPLT discovery tokenizer
   - continuous GlossAPI + HPLT 70/30 tokenizer
7. Bring artifacts back locally, analyze vocab/token artifacts, open issues only for
   non-blocking future work.
8. Stop the worker when done.

## Cleaner / Noise Code Changes Done

Cleaner repo:
`/home/foivos/glossAPI-development`

Changed files:

- `rust/glossapi_rs_cleaner/src/cleaning_module.rs`
- `rust/glossapi_rs_cleaner/src/normalize.rs`
- `rust/glossapi_rs_noise/src/lib.rs`
- `rust/glossapi_rs_noise/src/noise_metrics.rs`

Implemented:

- Added handling for bare PDF glyph tokens such as `GLYPH`, `GLYPHGLYPH`,
  and repeated `GLYPH...` through the existing glyph/PDF-artifact cleaner path.
- Kept glyph handling in normal prose only, consistent with the existing cleaner
  structure for code/math/table-like contexts.
- Protected URL/path-like tokens from glyph and selected PostScript removals.
- Preserved intentional HTML comments, per user request.
- Added URL-token span helper exposure from `normalize.rs`.
- Added Rust/PyO3 batch scoring APIs:
  `score_text_detailed(text)` and `score_texts_detailed(texts, n_threads=None)`.

Tests/builds run:

- Local cleaner full Rust test suite earlier:
  `382 passed; 0 failed; 3 ignored`
- Remote focused cleaner tests:
  `9 passed`
- Remote `glossapi_rs_noise` tests:
  `10 passed`
- Installed rebuilt wheels on the worker:
  - `glossapi_rs_cleaner-0.1.1`
  - `glossapi_rs_noise-0.1.0`

Deferred issue:

- Opened GitHub issue for later mojibake/Cyrillic/homoglyph calibration:
  `https://github.com/eellak/glossAPI/issues/99`

## Tokenizer / Pipeline Code Changes Done

Tokenizer repo:
`/home/foivos/Projects/glossapi-tokenizer-extension`

Changed or added files:

- `glossapi_corpus_cli/cli.py`
- `glossapi_corpus_cli/pipeline.py`
- `subprojects/01_2_training_dataset_mix/scripts/export_text_budgeted_splits.py`
- `subprojects/01_2_training_dataset_mix/scripts/create_real_subset_for_wave3_smoke.py`
- `subprojects/01_2_training_dataset_mix/scripts/wave3_orchestrate.py`
- `subprojects/01_0_cleaning_iteration_and_thresholds/scripts/reclean_canonical_to_parquet.py`
- `tests/test_pipeline.py`
- `tests/test_export_text_budgeted_splits.py`

Implemented:

- Strict badness filtering fails closed by default:
  - `greek_badness_score < 60`
  - `mojibake_badness_score <= 0.1`
  - missing/null/empty values fail closed unless explicitly overridden
- HPLT rows preserve `>=8` quality-bin scope and receive missing Greek badness
  scores while preserving existing mojibake scores.
- Reclean driver preserves existing badness values, fills missing HPLT Greek
  badness, writes atomically, resumes safely, and normalizes Arrow score column
  types.
- Pipeline efficiency improvements:
  - shared selected-input materialization
  - replay of existing builder dedup metadata
  - row-group sizing for continuous BPE parallelism
  - `--delete-mixes-after-split` to free disk after split export
- Added `--standard-split-filters` to `mix-build-from-selected-input`.
  This lets source-mix fractions be resolved after production split filters
  without materializing a huge filtered selected-input parquet.
- Updated `wave3_orchestrate.py` to use `--standard-split-filters` for source-mix
  construction going forward.

Focused tests run on the worker after the latest source-mix fix:

- `tests/test_pipeline.py::test_build_mix_from_selected_input_can_apply_standard_filters_before_source_share`
- `tests/test_pipeline.py::test_build_mix_export_supports_group_and_total_share_source_mix`
- `tests/test_pipeline.py::test_shared_selected_input_finalize_matches_direct_streaming_mix`
- `tests/test_export_text_budgeted_splits.py`

Result:

- `5 passed`

Earlier broader tokenizer test run:

- `26 passed`

## Integration Tests Done

Strict tiny integration:

- Run root:
  `/home/foivos/runs/wave3_20260428/integration_strict_tiny3/run`
- Completed all three tokenizers.
- Split summaries had:
  - `allow_missing_badness_scores=false`
  - `has_greek_badness=true`
  - `has_mojibake_badness=true`
- 70/30 train mix HPLT share was about `0.3000158`.

End-to-end reclean integration:

- Run root:
  `/home/foivos/runs/wave3_20260428/integration_reclean_strict_tiny4`
- Started from real original parquet subsets:
  - OpenArchives: 120 rows
  - Greek PhD: 70 rows
  - HPLT: 500 rows
- HPLT rows had missing Greek badness and existing mojibake.
- Reclean filled Greek badness for all 500 HPLT rows without overwriting mojibake.
- Completed selected input, existing dedup metadata replay, strict splits, fresh
  F1/F2 tokenizers, and C1 continuous tokenizer.
- Needed rerun with `--target-extension-units 128` because target vocab must be
  divisible by 128.
- Final strict summaries passed.

## Production Reclean Done

Run root:
`/home/foivos/runs/wave3_20260428/production_strict_v2`

Logs/artifacts:

- `reclean.log`
- `reclean_summary.jsonl`
- `reclean_validation.json`

Result:

- `tasks`: 274
- `errors`: 0
- `output_files`: 272
- rows total: 49,332,970
- HPLT rows Greek-badness scored: 48,728,774
- HPLT missing Greek badness after: 0
- chars before: 256,302,696,700
- chars after: 244,511,954,481
- chars removed: about 4.6 percent
- validation: `ok=true`

Intentionally dropped from production scope:

- `HuggingFaceFW__finepdfs-edu.parquet`
- `OPUS__OpenSubtitles-el-v2018.parquet`

## Production Split / Tokenizer Status

Selected input:

- Path:
  `/home/foivos/runs/wave3_20260428/production_strict_v2/shared/selected_input.parquet`
- Rows:
  49,124,434
- Chars:
  236,116,107,407
- Existing dedup metadata replay mode:
  `family_membership`

GlossAPI-only split:

- Train rows:
  119,357
- Train chars:
  32,473,766,114
- Fresh tokenizer:
  `F1_glossapi_only`
- Status:
  done
- Artifacts:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `training_summary.json`

HPLT-only split:

- Train rows:
  26,532,592
- Train chars:
  100,000,006,108
- Fresh tokenizer:
  `F2_hplt_only`
- Status:
  done
- Runtime:
  about 3143 seconds
- Vocab:
  50,000 requested, 50,000 actual

GlossAPI + HPLT 70/30 split:

- First production attempt selected 30 percent HPLT before strict split filters.
  After strict export, train was about 40.8 percent HPLT by chars.
- I treated that as incorrect for C1 and fixed the pipeline so 70/30 is resolved
  after the production badness/OCR filters.
- Corrected train split:
  - rows: 3,819,581
  - chars: 46,433,787,806
  - HPLT chars: 13,930,816,971
  - HPLT fraction: 0.300014658
- Corrected val/test are smaller and naturally wobble around 30 percent:
  - val HPLT fraction: about 0.3107
  - test HPLT fraction: about 0.2761
- Corrected mix was deleted after export to recover disk.

C1 continuous tokenizer:

- Name:
  `C1_glossapi_plus_hplt_70_30`
- Input:
  corrected `glossapi_plus_hplt_70_30/exports/train.parquet`
- Base tokenizer:
  Apertus base snapshot at
  `/home/foivos/data/glossapi_work/tokenizer_base_snapshots/apertus_8b_2509_20260415`
- Target vocab:
  156,672
- Base vocab:
  131,072
- Target added units:
  25,600
- Parallel stages complete:
  - segment shards: 1,894 / 1,894
  - sequence shards: 1,894 / 1,894
- Current stage:
  complete
- Last seen progress:
  - added merges/tokens: 25,600 / 25,600
  - current vocab size: 156,672
  - latest pair frequency: 47,589
  - tokenizer SHA256:
    `74a092cd55c258b725cbd55c28b21ac25da4afb728ce73b5c103c7463d190319`

Local artifact archive:

`/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/runs/production_strict_v2`

Tokenizer review document:

`/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_0_cleaning_iteration_and_thresholds/WAVE3_TOKENIZER_REVIEW_2026-04-28.md`

## Known Pipeline Lessons From This Run

- The 70/30 source mix must be resolved after the same strict production filters
  used by split export. Resolving it before strict export can skew the actual
  train split because non-HPLT rows lose more material.
- Materializing a filtered selected-input parquet before source selection is too
  expensive. It briefly created a 106G temp file. The final implementation pushes
  the predicate into the source-mix SQL instead.
- The split exporter currently scans the mix again for each split export. This
  made small val/test exports slower than expected. It is not blocking this run,
  but it is a useful future optimization target.
- Fresh tokenizer training is quiet until completion. Useful health signals are
  process liveness, CPU use, output artifacts, and training summaries.
- Continuous BPE monitoring should use `progress.json`; it gives accurate phase
  and task counters.

## Remaining Work

1. Stop the worker after this progress document is synced.

Completed after the earlier checklist:

- C1 merge loop finished and wrote the final tokenizer.
- `training_summary.json`, `front_end_contract_check.json`,
  `replication_check.json`, and `all_done.json` were validated.
- Key run artifacts were copied locally.
- Tokenizers were reviewed locally.
- Follow-up issues were opened for non-blocking cleanup:
  - `https://github.com/eellak/glossAPI/issues/99`
  - `https://github.com/eellak/glossAPI/issues/100`
  - `https://github.com/fffoivos/glossapi-tokenizer-extension/issues/1`

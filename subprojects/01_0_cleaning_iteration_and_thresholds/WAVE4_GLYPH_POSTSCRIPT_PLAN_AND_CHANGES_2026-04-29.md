# Wave 4 Glyph/PostScript Cleaner Plan and Code Changes - 2026-04-29

This document records the code changes and analysis before restarting the
GlossAPI/HPLT reclean, dedup replay, split build, and tokenizer run.

## Why This Wave Exists

The previous strict production run successfully reduced tokenizer residue, but
the fresh GlossAPI tokenizer still learned a small family of `GLYPH...` tokens
and `/hyphenminus` tokens.

Follow-up corpus scanning showed two concrete root causes:

- `GLYPHGLYPH...` came from a tiny number of dense failed display-math lines in
  `greek_phd`. These survived because the cleaner deliberately skipped
  glyph/PostScript cleanup inside `$$...$$` math.
- `/hyphenminus` survived in numeric or word ranges because the URL/path guard
  protected dotted slash tokens such as `4.600/hyphenminus5.600`.

The broader context-blind scan also matched many legitimate lowercase `glyph`
substrings (`glyphosate`, `hieroglyphs`, `Paraglyph`, glossary rows for
`glyph <typography>`). Those are intentionally not cleaner targets.

## Cleaner Code Changes

Repo:

`/home/foivos/glossAPI-development`

Changed file:

`rust/glossapi_rs_cleaner/src/cleaning_module.rs`

Implemented changes:

- Reused the existing Rule A / Rule B glyph and PostScript machinery.
- Added replacement-aware glyph span rewriting:
  - ordinary glyph/PostScript spans are still stripped;
  - glued `/hyphenminus` spans are rewritten to `-`.
- Narrowed `/hyphenminus` URL protection:
  - true URLs remain protected;
  - numeric dotted ranges such as `4.600/hyphenminus5.600` are cleaned;
  - word ranges such as `θεσμικο/hyphenminusδιοικητικών` are cleaned.
- Added a display-math glyph-residue check before the existing math skip:
  - contaminated display-math lines are dropped;
  - clean display math is preserved;
  - matching is restricted to high-confidence residue: uppercase `GLYPH`
    stems, structured glyph/PDF markers, and slash-prefixed PostScript glyph
    names.
- Kept generic lowercase `glyph` substrings untouched.

New tests:

- contaminated `$$...GLYPHGLYPH...$$` display math is dropped;
- clean `$$x^2 + y^2 = z^2$$` display math is preserved;
- `4.600/hyphenminus5.600`, `75/hyphenminus77`, and
  `θεσμικο/hyphenminusδιοικητικών` normalize with `-`;
- real URL paths containing `/hyphenminus` remain protected.

Verification so far:

- Focused cleaner tests:
  - `cargo test core_clean_text -- --nocapture`
  - result: `31 passed`
- Full cleaner suite:
  - `cargo test`
  - result: `385 passed; 0 failed; 3 ignored`

## Production Workflow Plan

1. Build/install the updated cleaner where the production reclean will run.
2. Run a real-subset end-to-end integration test:
   - original parquet subset;
   - reclean with preserved historical badness values;
   - strict filtering;
   - existing dedup metadata replay;
   - split export;
   - tiny tokenizer training;
   - inspect targeted residue.
3. Reclean the original production datasets.
   - Preserve existing historical badness metrics where present.
   - HPLT remains constrained to quality bin `>=8`.
   - HPLT receives missing Greek badness scores and must also pass standard
     cleaner badness filters.
4. Do not rerun duplicate discovery.
   - Reuse existing dedup/family membership values.
5. Rebuild the strict splits:
   - GlossAPI-only;
   - HPLT-only;
   - GlossAPI + HPLT 70/30 after filters.
6. Train tokenizers:
   - fresh GlossAPI tokenizer;
   - fresh HPLT tokenizer;
   - continuous GlossAPI + HPLT 70/30 tokenizer.
7. Bring artifacts local and inspect:
   - `GLYPH...`;
   - `/hyphenminus`;
   - `/uniXXXX`;
   - `/gNN`;
   - mojibake markers;
   - continuous-tokenizer added-token slice.
8. Document results and stop the worker instance.

## Exit Criteria

- No learned `GLYPHGLYPH...` ladder in the fresh GlossAPI tokenizer.
- `/hyphenminus` no longer appears due to numeric or word range artifacts.
- HPLT remains strict: quality bin `>=8` plus standard badness filters.
- Remaining non-blocking residue is documented as future work rather than
  triggering another perfection loop.

## Production Correction During Run

During validation of the first production pass, the split summaries and the
physical exported parquet metadata disagreed for the non-HPLT arms:

- GlossAPI train summary: `119,361` rows / `32.45B` chars.
- GlossAPI exported train parquet: `177,352` rows / `43.75B` chars.
- GlossAPI + HPLT 70/30 train summary: `3,816,878` rows / `46.39B` chars.
- GlossAPI + HPLT 70/30 exported train parquet: `3,874,908` rows / `57.70B`
  chars.

Root cause: `export_text_budgeted_splits.py` assigned split membership by
`source_dataset, source_doc_id`, then joined back to the source parquet on those
two fields to export `text`. Some GlossAPI sources have repeated
`source_doc_id` values, so the export join cross-multiplied those rows. HPLT did
not show the mismatch because its ids were unique in this run.

Fix:

- `export_text_budgeted_splits.py` now materializes `assigned_rows` with a
  per-row `source_split_row_id` and carries `text` through split assignment.
- Exported train/val/test parquet files are written directly from
  `assigned_rows`; no non-unique join is used.
- The manifest now includes `source_split_row_id`.
- The exporter also records `has_source_mix_chars` and can use
  `source_mix_chars` for budget accounting when present.
- Added a focused regression test:
  `test_export_duplicate_source_doc_ids_do_not_cross_join`.
- `continuous_bpe.py` now throttles `progress.json` writes during the serial
  merge loop. Checkpoint cadence is unchanged; this only avoids an atomic fsync
  after every merge.

Verification:

- Worker-side focused exporter tests:
  - `python -m pytest tests/test_export_text_budgeted_splits.py -q`
  - result: `3 passed`
- Worker-side continuous-BPE smoke:
  - `python -m pytest tests/test_continuous_bpe.py::test_run_continuation_training_appends_new_merge -q`
  - result: `1 passed`

Action taken:

- Stopped the in-progress C1 continuous tokenizer trained on the expanded
  70/30 export.
- Invalidated only affected artifacts:
  - GlossAPI mix/split/F1;
  - GlossAPI + HPLT 70/30 mix/split/C1.
- Preserved HPLT split and F2 tokenizer, which matched their physical parquet
  export.
- Relaunched the Wave 4 production driver to rerun the invalidated phases.

## Corrected Production Rerun Status

Run root:

`/home/foivos/runs/wave4_20260429/production_strict_v1`

Reclean validation:

- `ok: true`
- tasks: `272`
- errors: `0`
- output parquet files: `270`
- rows total: `49,332,081`
- rows with Greek badness score after reclean: `48,728,774`
- rows missing Greek badness after reclean: `291,107`
- rows missing mojibake badness after reclean: `448,025`
- chars before: `256,233,604,772`
- chars after: `244,407,580,918`

Corrected split summaries:

| split arm | train rows | train chars | val rows | val chars | test rows | test chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GlossAPI only | `119,349` | `32,445,655,014` | `180` | `49,881,654` | `189` | `50,084,964` |
| HPLT only | `26,535,070` | `100,000,050,052` | `12,742` | `49,987,034` | `12,993` | `49,959,998` |
| GlossAPI + HPLT 70/30 | `3,815,742` | `46,394,028,949` | `4,516` | `49,998,246` | `4,129` | `49,719,674` |

Fresh tokenizer training:

- `F1_glossapi_only`
  - trained from corrected GlossAPI split;
  - input parquet bytes: `11,996,347,903`;
  - rows: `119,349`;
  - vocab size: `50,000`;
  - runtime: `1,229.8s`.
- `F2_hplt_only`
  - preserved from the already-valid HPLT split;
  - input parquet bytes: `49,884,656,046`;
  - rows: `26,535,070`;
  - vocab size: `50,000`;
  - runtime: `2,898.3s`.
- `C1_glossapi_plus_hplt_70_30`
  - restarted from the corrected 70/30 split and completed successfully;
  - input rows: `3,815,742`;
  - target vocab size: `156,672`;
  - base vocab size: `131,072`;
  - added tokens: `25,600`;
  - added merges: `25,600`;
  - final tokenizer SHA-256:
    `fd56788dd0957fd11f111b43f2100849a330068017a93cbe54ddd5e150a0e39c`;
  - total runtime: `8,186.5s`;
  - phase times:
    - identity check: `3.4s`;
    - segment counting: `1,097.6s`;
    - sequence shard build: `177.2s`;
    - sequence aggregation: `573.6s`;
    - merge loop: `6,322.1s`;
    - tokenizer write: `4.5s`;
  - completed segment counting: `1,864/1,864` row-group tasks;
  - completed sequence shard build: `1,864/1,864` row-group tasks;
  - aggregate produced `27,703,730` unique base-token sequences;
  - front-end contract check completed.

Final artifact analysis:

- The `GLYPHGLYPH...` tokenizer ladder is gone from the corrected fresh
  GlossAPI tokenizer.
- `F1_glossapi_only` still has two learned plain `GLYPH` tokens and one
  `/hyphenminus` token.
- `C1_glossapi_plus_hplt_70_30` added-token slice has:
  - uppercase `GLYPH`: `0`;
  - repeated `GLYPH`: `0`;
  - `/hyphenminus`: `0`;
  - structured glyph markers: `0`;
  - slash-name matches: `31`, mostly broad PostScript-name false positives
    such as `/CP`, `/CX`, `/GE`, `/pi`;
  - mojibake marker tokens: `5`;
  - Latin Extended A/B tokens: `1`;
  - nondecodable boundary fragments: `22`.
- A context scan over the corrected GlossAPI train parquet found:
  - `GLYPH`: `136` rows / `61,662` hits;
  - repeated `GLYPH`: `44` rows / `6,326` hits;
  - `/hyphenminus`: `2,150` rows / `43,507` hits;
  - `/uniXXXX`: `25` rows / `169` hits;
  - `/g...` or `/gid...`: `605` rows / `10,642` broad-regex hits.

Interpretation:

- The latest cleaner changes solved the targeted display-math ladder and
  cleaned normal-text numeric/word `/hyphenminus` cases.
- Residue still present in the corrected train split appears concentrated in
  contexts the cleaner preserves or treats cautiously, especially fenced
  code-like blocks and bibliographic/code-ish fragments.
- The broad `/g...` scan intentionally overcounts URLs and DOI/path fragments;
  it is useful as a discovery scan, not as a direct deletion rule.
- This should become a future calibration issue rather than another production
  restart unless final tokenizer analysis shows a severe regression.

Final run artifacts were copied locally to:

`/home/foivos/Projects/glossapi-tokenizer-extension/tokenizer_analysis/inspection/wave4_production_strict_v1_20260429`

Follow-up issues:

- Residual protected-context cleaner calibration:
  `https://github.com/fffoivos/glossapi-tokenizer-extension/issues/2`
- Continuous-BPE merge-loop parallelism/progress ergonomics:
  `https://github.com/fffoivos/glossapi-tokenizer-extension/issues/3`

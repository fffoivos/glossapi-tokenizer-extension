# 01.2 — Training dataset mix (mix.parquet and the splitter)

> **In one line:** built the source-mix and split machinery that produced every tokenizer arm's training corpus, ending with C3's `glossapi + hplt` 50/50 `mix.parquet` — and with a known row-vs-document leak in the splitter that cost the project its verifiable held-outs.
> **Period:** 2026-04-10 (`f21eed85`) → 2026-05-14 (`002bddc5`, archive move). **Status:** completed; the C3 mix and splits shipped, the splitter bug was documented and worked around rather than fixed.
> **Came from / led to:** [`../01_1_corpus_dedup`](../01_1_corpus_dedup/README.md) (published the dedup overlay this builder consumes) → this → [`../../02_1_tokenizer_experiments`](../../02_1_tokenizer_experiments/README.md) (trained C3 on the resulting splits).

## Why this existed

Every tokenizer arm needed a reproducible answer to "what text did it see, and what text is safe to evaluate it on". This subproject owned both halves: the *mix* layer (which sources, in what proportion, by character mass) and the *split* layer (deterministic train/val/test export under a character budget, with the quality gates applied). It is the reason C3's training corpus can be described exactly today.

## History

### 2026-04-10 → 04-12 — the plan is written and then re-cut twice

Created with two comparison views (`GlossAPI-only`, `GlossAPI + HPLT`), a rule that held-out documents must not overlap training documents, and six named manifest deliverables — `nanochat_train`, `hplt_matched_sample`, the mixed manifest, and the `nanochat_eval` / `hplt_eval` / `modern_greek_eval` held-outs (`f21eed85`). Scope was widened the next day to use *the same CPT-ready dataset that will later feed Apertus continued pretraining*, not a tokenizer-only view (`2c803082`). On 2026-04-12 two further decisions landed: `openarchives.gr` rows with `needs_ocr == true` stay excluded, and local mix freezing may proceed without waiting for the HF upload (`0a8a50be`).

The sizing rule changed the same day. "HPLT should initially be matched roughly to nanochat scale" was replaced by a locked **70/30 by training-token mass** (`340dbf8f`) — the ratio the archived README still reports as current, although C3 ultimately shipped at 50/50.

The mix layer itself became a JSON config: each entry names sources to include or exclude and takes a fraction either `of_group` (its own available character mass) or `of_total` (of the final mix). The three committed configs are [`examples/glossapi_only_all_non_hplt.json`](examples/glossapi_only_all_non_hplt.json), [`examples/glossapi_plus_hplt_70_30.json`](examples/glossapi_plus_hplt_70_30.json) and [`examples/hplt_only.json`](examples/hplt_only.json).

### 2026-04-26 — thresholds become defaults, and the orchestrator is written under a course correction

The wave-2 standard exclusions were baked into [`scripts/export_text_budgeted_splits.py`](scripts/export_text_budgeted_splits.py) as defaults — `greek_badness_score < 60`, `mojibake_badness_score ≤ 0.1`, `charset_greek_ratio ≥ 0.5`, non-empty post-clean content (`f26d5c49`). Mid-run the user flagged that a whole-pipeline integration test had been skipped before the corpus-scale sweep; the recovery produced [`scripts/wave2_orchestrate.py`](scripts/wave2_orchestrate.py), a marker-file-driven, resumable orchestrator covering dedup → mix-prepare → mix-build → export-splits → four tokenizer arms (`b3e92fa1`).

The smoke run that the course correction forced immediately paid for itself, catching four real bugs before the corpus-scale run ([`../01_0_cleaning_iteration_and_thresholds/WAVE2_PIPELINE_RUN_2026-04-26.md`](../01_0_cleaning_iteration_and_thresholds/WAVE2_PIPELINE_RUN_2026-04-26.md) § Course corrections):

1. the smoke corpus contained no HPLT, so `include_sources: ["HPLT/ell_Grek_ge8_no_mt_clean60"]` matched zero rows — the selector was correct, the *sample* was incomplete. Lesson recorded: a smoke set must contain ≥1 part of every dataset family the mix configs reference;
2. share math — with HPLT at 30 % `of_total` against 4.51 GB of non-HPLT chars, HPLT needed ≥1.93 GB and the 30 k-row slice supplied ~520 MB, 4× short;
3. the orchestrator passed `--target-extension-units` to a continuous-BPE trainer that wanted `--base-tokenizer-dir` + `--target-vocab-size`;
4. on a small mix the exporter filled the train budget first and emitted **no val/test at all** — flagged as a budget-priority bug, judged not to affect production where the train budget dwarfs the mix.

### 2026-04-28 — the 70/30 was being measured in the wrong place

The first wave-3 production attempt selected 30 % HPLT *before* the strict split filters, so the exported train split came out **40.8 % HPLT by characters**. The pipeline was changed so source-mix fractions resolve *after* the production badness/OCR filters (`--standard-split-filters`), and the corrected split landed at **3,819,581 rows / 46,433,787,806 chars, HPLT fraction 0.300014658** ([`../01_0_cleaning_iteration_and_thresholds/WAVE3_PRODUCTION_PROGRESS_2026-04-28.md`](../01_0_cleaning_iteration_and_thresholds/WAVE3_PRODUCTION_PROGRESS_2026-04-28.md)).

### 2026-04-29 — the export cross-join bug, and the fix that later backfired

During wave-4 validation, split summaries and the physical exported parquets disagreed for the non-HPLT arms: GlossAPI train summary said 119,361 rows / 32.45 B chars, the exported parquet had 177,352 rows / 43.75 B chars. Root cause: the exporter assigned split membership by `(source_dataset, source_doc_id)` and then joined back to the source parquet on those two fields to fetch `text` — and some GlossAPI sources repeat `source_doc_id`, so the join cross-multiplied. The fix materialized `assigned_rows` with a **per-row `source_split_row_id`**, carried `text` through assignment, and added the regression test `test_export_duplicate_source_doc_ids_do_not_cross_join` ([`../01_0_cleaning_iteration_and_thresholds/WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md`](../01_0_cleaning_iteration_and_thresholds/WAVE4_GLYPH_POSTSCRIPT_PLAN_AND_CHANGES_2026-04-29.md) § Production Correction). The in-progress C1 tokenizer and the GlossAPI and 70/30 mixes were invalidated and rerun; the HPLT split and F2 were preserved.

That per-row key is precisely what the C3 verification later found leaking (below): the fix for the cross-join replaced a document-level assignment key with a row-level one.

### 2026-05-06 → 05-11 — the C3 mix, and the held-out integrity finding

C3 shipped at **50/50**, not the 70/30 the README still records: `mix.parquet` = **14,453,413 rows / 104.94 B chars**, from 546,920 GlossAPI rows (52.47 B chars, all of the available pool) plus 13,906,493 HPLT rows sampled to match (52.47 B chars). Splits: train 14,401,554 rows / 100 B chars, val 7,654 / 50 M, test 7,282 / 50 M ([`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)).

Verification on 2026-05-11 found the splits are **not disjoint**: 29,527 duplicate texts inside train, train ∩ val = **30** docs, train ∩ test = **36** docs, val ∩ test = 0. Because `stable_key` hashes `source_dataset:source_doc_id:source_split_row_id:salt`, duplicate texts get independent split assignments. Contamination is ~0.4–0.5 % of val/test — below the fertility metric's noise floor, but the slices are not *verifiable* held-outs. The fix path (hash the text, or pre-dedup the mix on text before assignment) was documented in the script header and **not back-ported to C3**; instead the cutoff sweep was re-anchored on a virgin HPLT eval slice built by anti-joining the wave-4 HPLT release against C3 train text-md5 ([`../../../docs/C3_CONVERGENCE.md`](../../../docs/C3_CONVERGENCE.md) § Held-out integrity).

## Outcome

- **Shipped**: the mix-config layer, [`scripts/export_text_budgeted_splits.py`](scripts/export_text_budgeted_splits.py), and three generations of orchestrator (`wave2_orchestrate.py`, `wave3_orchestrate.py`, `wave4_production_driver.sh`). C3's `mix.parquet` and its train/val/test exports are the concrete deliverable.
- **Decided**: mix shares resolve *after* the production quality filters, not before; quality gates live in the exporter's defaults and fail closed on missing badness scores.
- **Known defect carried forward**: the splitter partitions by row, not by document. Documented at the top of the script, in `C3_CONVERGENCE.md` and in this directory's own status block; never fixed.
- **Superseded**: the 70/30 lock (`340dbf8f`) and the six named manifest deliverables from `f21eed85`. What actually exists is `mix.parquet` + `train/val/test.parquet` per mix; of the named eval manifests only `modern_greek_eval` survives, as a planned eval slice in the C3 cutoff plan.

## Where things are

| Artifact | Role |
|---|---|
| [`scripts/export_text_budgeted_splits.py`](scripts/export_text_budgeted_splits.py) | The splitter. Read its header comment first — it documents the row-vs-doc leak and the unapplied fix. |
| [`scripts/wave2_orchestrate.py`](scripts/wave2_orchestrate.py), [`scripts/wave3_orchestrate.py`](scripts/wave3_orchestrate.py), [`scripts/wave4_production_driver.sh`](scripts/wave4_production_driver.sh) | The three run drivers, one per wave; the wave-4 driver shows the final shape (reclean → replay dedup metadata → strict splits → tokenizers). |
| [`examples/`](examples/) | The three mix configs (`glossapi_only_all_non_hplt`, `glossapi_plus_hplt_70_30`, `hplt_only`). |
| [`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md) | Authoritative record of the C3 mix, per-source row counts and the split table. |

## Working documents

Historical, kept for traceability:

- [`TODO.md`](TODO.md) — the open list at the archive move; its first four items (define `hplt_matched_sample` size, freeze the review-sample stratification, freeze the 70/30 manifests, freeze the held-out manifests) were overtaken by the 50/50 C3 mix.
- [`scripts/create_real_subset_for_wave3_smoke.py`](scripts/create_real_subset_for_wave3_smoke.py), [`scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh`](scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh) — smoke-sampling and worker-wrapper helpers from the wave-2/3 era.

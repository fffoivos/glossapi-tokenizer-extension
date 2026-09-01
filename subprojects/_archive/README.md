# 01 — Corpus phase (archived)

> **In one line:** the first phase of the program — build the Greek training corpus (HPLT slice, cleaner iteration, dedup, mix and splits) that every tokenizer arm and, later, every CPT run was built on.
> **Period:** 2026-04-10 (`f21eed85`, the restructure that created these directories) → 2026-05-14 (`002bddc5`, the commit that moved all four into `_archive/`). **Status:** completed; superseded as active work by the C3 convergence on 2026-05-11.
> **Came from / led to:** the pre-restructure monolithic plan (preserved in [`../../legacy/monolithic_docs/`](../../legacy/monolithic_docs/)) → this → [`../02_1_tokenizer_experiments`](../02_1_tokenizer_experiments/README.md), which trained C3 on the corpus produced here.

## Why this existed

Extending Apertus-8B's tokenizer for Greek — and later continuing its pretraining — needed a Greek corpus that was large, clean, deduplicated and *describable*. None of those four came for free. The GlossAPI collection alone was too small and too PDF-heavy; HPLT 2.0's Greek slice was large but full of machine translation and low-quality pages; the dedup implementation did not scale to 49 M documents; and the cleaner kept missing noise families that only became visible once a tokenizer had learned them. These four subprojects are the answers to those four problems, in that order.

## History

**2026-04-10 → 04-12 — the plan.** `f21eed85` restructured a monolithic plan into clean subprojects, creating `01_hplt_filtering`, `01_1_corpus_dedup`, `01_2_training_dataset_mix` alongside the `02_*`/`03` tokenizer directories. The next three days were planning churn visible entirely in README diffs: HPLT's deliverable moved from a *manifest* to an *upload-ready canonical parquet* (`2c803082`); the dataset track was explicitly decoupled from the tokenizer critical path (`0a8a50be`); the `greek_badness_score > 60` "clean60" rule was added (`ed598b30`); and the mix ratio was locked at 50 k discovery vocab / 70-30 mixed view (`340dbf8f`).

**2026-04-13 → 04-15 — infrastructure, and the first crisis.** This repo was declared the canonical home of the pipeline code (`a062d0aa`, `66e7886b`). Then near-dedup failed at scale: 16 workers on a 976 GB machine reached ~955 GB with 0/32 bands complete, prompting a study of HuggingFace DataTrove and a staged external-merge redesign that kept the project's own semantics ([`../../docs/_archive/HF_DEDUP_INVESTIGATION.md`](../../docs/_archive/HF_DEDUP_INVESTIGATION.md), [`../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md`](../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md)). By 2026-04-15 the whole worker chain — dedup → overlay publish → mix build → tokenizer training → uploader handoff — was verified end to end twice ([`../../docs/_archive/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md`](../../docs/_archive/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)).

**2026-04-22 → 04-29 — the cleaning loop.** `01_0_cleaning_iteration_and_thresholds` was created on 2026-04-23 (`296a7022`) — after the other three, and numbered `01_0` to place it upstream of them in the pipeline. It ran four waves in eight days, each one training tokenizers, reading the noise families they learned, patching the upstream Rust cleaner and re-cleaning all 49 M documents. Wave 2 (2026-04-26) was the first full-corpus run; wave 3 (2026-04-28) was aborted once and completed on the retry; wave 4 (2026-04-29) closed out the glyph/PostScript families and then deliberately stopped rather than chase the remainder.

**2026-05-06 → 05-11 — C3, and the held-out finding.** The shipping arm C3 (`glossapi + hplt` 50/50, 25,600 added units, total vocab 156,672) was trained and declared converged on 2026-05-11 ([`../../docs/C3_CONVERGENCE.md`](../../docs/C3_CONVERGENCE.md)). Verification the same day found the splitter partitions by row rather than by document, leaking 30 val and 36 test documents into train; the cutoff sweep was re-anchored on a virgin HPLT slice instead of fixing the splits.

**2026-05-14 — archived.** `002bddc5` moved all four directories under `_archive/`, added a "DONE for the C3 shipping path (as of 2026-05-11)" block to each README, and stamped a "Historical reference" banner onto every doc in `01_0`. It also landed a large backlog of wave-3/wave-4 documents that had been written on the worker between 2026-04-24 and 2026-05-04 but never committed — which is why almost nothing in these directories has a commit date matching the work it describes.

## Outcome

- **The corpus C3 trained on**: `mix.parquet` at 14,453,413 rows / 104.94 B chars, split into train 14,401,554 / 100 B chars, val 7,654 / 50 M, test 7,282 / 50 M ([`../../docs/C3_TRAINING_DATASETS.md`](../../docs/C3_TRAINING_DATASETS.md)).
- **Two published datasets**: [`fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`](https://huggingface.co/datasets/fffoivos/hplt-greek-ge8-no-mt-clean60-wave4) (48,728,774 rows) and [`fffoivos/glossapi-greek-nanochat-pretraining-dataset`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset) (19 source datasets, 810,638 raw rows).
- **One full-corpus dedup**, run once in wave 2 (49,292,755 → 49,090,905 kept) and replayed as frozen metadata by every later build.
- **Quality gates that outlived the phase**: `greek_badness_score < 60`, `mojibake_badness_score ≤ 0.1`, `charset_greek_ratio ≥ 0.5`, non-empty post-clean, `openarchives.gr needs_ocr == False`.
- **Left open**: the row-vs-document splitter leak (documented, never fixed); mojibake and homoglyph repair (deferred to `eellak/glossAPI` issue #99); the deletion-% and minimum-content thresholds wave 1 opened; a golden equivalence test for the repaired dedup exact stage.

## Sub-subprojects

Listed in pipeline order, which is not creation order — `01_0` was created last and renumbered to sit upstream.

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`01_hplt_filtering/`](01_hplt_filtering/README.md) | Build and integrate the filtered Greek HPLT slice | 2026-04-10 → 2026-05-14 | completed | `HPLT/ell_Grek_ge8_no_mt_clean60`, 48.7 M rows; 13.9 M sampled into C3's mix |
| [`01_0_cleaning_iteration_and_thresholds/`](01_0_cleaning_iteration_and_thresholds/README.md) | Tokenizer-guided cleaner iteration + rejection thresholds | 2026-04-22 → 2026-05-14 | completed (per-line successor branch left unmerged) | Four waves; glyph/PostScript residue driven to 0 in C1's added-token slice; `THRESHOLDS.yaml wave_2_20260426` |
| [`01_1_corpus_dedup/`](01_1_corpus_dedup/README.md) | Dedup contract, scaling repair, one full-corpus run | 2026-04-10 → 2026-05-14 | completed | 49,292,755 → 49,090,905 kept; exact stage moved off SQLite; near `length_ratio` gate removed |
| [`01_2_training_dataset_mix/`](01_2_training_dataset_mix/README.md) | Source-mix configs, budgeted train/val/test splitter | 2026-04-10 → 2026-05-14 | completed with a known defect | C3's 50/50 `mix.parquet` + splits; splitter leaks duplicate texts across splits |

## Where things are

| Artifact | Role |
|---|---|
| [`../../docs/C3_TRAINING_DATASETS.md`](../../docs/C3_TRAINING_DATASETS.md) | The authoritative record of what this phase produced: every source, every filter, the mix recipe and the split table. |
| [`../../docs/C3_CONVERGENCE.md`](../../docs/C3_CONVERGENCE.md) | The 2026-05-11 decision that ended the phase, plus the held-out integrity finding. |
| [`../../glossapi_corpus_cli/`](../../glossapi_corpus_cli/) | The pipeline engine these subprojects drive — `build`, `dedup-text`, `mix-prepare`/`mix-build`, `text_dedup.py`, `continuous_bpe.py`. Declared this repo's canonical code root on 2026-04-14 (`a062d0aa`). The `01_*` directories hold the drivers, configs and decisions; the engine lives there. Not documented further here. |
| `eellak/glossAPI` (external) | The Rust cleaner crates (`glossapi_rs_cleaner`, `glossapi_rs_noise`) that `01_0` patched wave by wave. Not vendored into this repo. |

## Working documents

Historical, kept for traceability:

- [`../../docs/_archive/`](../../docs/_archive/) — the repo-level companions to this phase: the dedup recovery and near-dedup redesign plans, the HF/DataTrove investigation, the pipeline E2E verification family and the stage-verification checklist. Its own [`README.md`](../../docs/_archive/README.md) indexes them.
- [`../../legacy/corpus_clean_normalization/`](../../legacy/corpus_clean_normalization/) — the rule-discovery pipeline that fed `01_0`'s wave-3 rule list (A–M).
- [`../../legacy/monolithic_docs/`](../../legacy/monolithic_docs/) — the pre-2026-04-10 plan this phase was carved out of.

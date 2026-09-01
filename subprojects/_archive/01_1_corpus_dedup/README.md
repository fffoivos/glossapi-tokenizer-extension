# 01.1 — Corpus dedup (contract, scaling crisis, repair)

> **In one line:** started as a plan to deduplicate the corpus, was immediately reduced to *documenting the contract owned by the upstream pipeline*, then had to take back ownership when that pipeline's exact and near stages failed at 49 M documents — the repair shipped and the frozen `dedup_metadata` bundle was replayed by every later run.
> **Period:** 2026-04-10 (`f21eed85`) → 2026-05-14 (`002bddc5`, archive move). **Status:** completed; the dedup metadata bundle was frozen and C3 trained on the deduped corpus.
> **Came from / led to:** [`../01_hplt_filtering`](../01_hplt_filtering/README.md) (integrated the HPLT slice into the working release that dedup ran over) → this → [`../01_2_training_dataset_mix`](../01_2_training_dataset_mix/README.md) (consumed the published overlay).

## Why this existed

BPE training on a corpus with duplicate documents over-weights the duplicated text in the merge statistics, so the 49 M-document corpus had to be deduplicated before any tokenizer arm could be judged. The pipeline code (`../../../glossapi_corpus_cli/text_dedup.py`) already implemented the semantics — strict exact → relaxed exact → MinHash near-dup at similarity ≥ 0.85, with a keeper-selection policy over OCR validity, length and badness. The question this subproject actually had to answer was not *what* dedup means but *whether it could be made to run at all* on the full corpus.

## History

### 2026-04-10 → 04-12 — scope collapses to a contract

Created with a five-step required order: same-source overlap reduction → canonical URL dedup → exact-text dedup → near-duplicate dedup → freeze the training manifest (`f21eed85`). One day later that whole order was deleted and the scope reversed: *"document and verify the dedup contract inherited from the upstream GlossAPI corpus dataset pipeline"*, with the explicit decision not to reimplement a second local dedup builder stage (`2c803082`). On 2026-04-12 the published-`dedup_metadata` refresh was declared a later operational step that does not block tokenizer progress (`0a8a50be`).

### 2026-04-14 → 04-15 — the near-dedup scaling crisis and redesign

The near-candidate stage did not scale. On the `m3-megamem-64` worker (976 GB RAM), 16 near-candidate workers drove memory to ~955/960 GB while still showing **0 / 32 bands complete** ([`../../../docs/_archive/HF_DEDUP_INVESTIGATION.md`](../../../docs/_archive/HF_DEDUP_INVESTIGATION.md)). That triggered an explicit diversion to study HuggingFace DataTrove's MinHash implementation. The conclusion was *not* to adopt it — DataTrove has no strict/relaxed exact path, no keeper policy and no builder-metadata outputs — but to transplant its staged external-merge execution shape while keeping the project's semantics.

The redesign landed the same two days ([`../../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md`](../../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md)):

- prefix-chunked candidate work instead of one-band-per-task (`90ba1299`, `4843d658`, `c27caa14`);
- batched pair-shard writes and a fix for parquet write amplification (`66512e9b`, `69f05934`);
- a measured `fork`-vs-`spawn` decision: at 16 workers over 16,384 synthetic docs, peak total child PSS was **587.6 MB (fork)** vs **1542.1 MB (spawn)** at essentially equal wall time, making shared-state `fork` the default on Linux workers (`c5a93c7d`, recorded in the redesign plan);
- one deliberate **semantic** change: the hard near-dedup `length_ratio ≥ 0.70` admission gate was removed, so short-vs-long high-similarity pairs now reach representative selection (`6100dc22`; regression test `test_near_dedup_accepts_high_similarity_pair_even_when_length_ratio_is_low` in [`../../../tests/test_text_dedup.py`](../../../tests/test_text_dedup.py));
- 2026-04-15 follow-ups parallelized partition prep and the near-cluster stage (`dd41904f`, `f75f07f2`, `5438cf95`, `cd8c6280`, `92e82fa7`).

### 2026-04-14 — the exact stage takes the subproject's second role

The README was rewritten to give the subproject *two* roles: document the contract, **and** own the scaling repair for the exact stage (`a062d0aa`). Diagnosis: the expensive chunk phase completed fine, but `build_stage_results(...)` rebuilt stage outputs through SQLite, `run_exact_results` was deleted and repopulated per stage, the WAL grew to hundreds of GB and finalization was effectively single-process ([`DEDUP_SCRIPT_REPAIR_PLAN.md`](DEDUP_SCRIPT_REPAIR_PLAN.md)). The governing rule for the repair was stated as a golden rule: *functionality stays the same, efficiency changes, semantics do not change* — including `exact_strict_hash`/`exact_relaxed_hash`, strict-then-relaxed ordering and `selection_priority_tuple(...)` as the keeper decision.

The repair shipped as [`scripts/parallel_exact_stages.py`](scripts/parallel_exact_stages.py), which builds the strict and relaxed stage parquets directly off `run_docs_inventory.parquet` and replicates `selection_priority_tuple` / `representative_score` exactly, so the library's `build_stage_results` takes its `reuse_existing_parquet` fast path and skips the serial rebuild.

### 2026-04-15 — end-to-end chain verified

The worker-side chain (dedup → overlay publish → mix build → tokenizer training → uploader handoff) was verified twice: a live large-chain rearm on the real worker, and a bounded real-document smoke run that completed all ten stages ([`../../../docs/_archive/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md`](../../../docs/_archive/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)). The wrappers proved by that report are [`scripts/wait_for_hplt_integration_and_run_dedup.sh`](scripts/wait_for_hplt_integration_and_run_dedup.sh) and [`scripts/wait_for_dedup_and_publish_overlay.sh`](scripts/wait_for_dedup_and_publish_overlay.sh).

### 2026-04-26 → 04-27 — the full wave-2 run, and the disk-full crash

Stage 1 (exact) processed 96,364 chunks at ~408 chunks/min. It was deliberately killed at 82 % (78,982 chunks, 55 GB of state) for an unrelated course correction, resumed from state, and finished. Result: **49,292,755 input docs → 49,090,905 kept**, Stage-1 drops 25,465 (0.052 %).

Stage 2 (near) then crashed: `OSError: [Errno 28] No space left on device` after writing 112 of 512 `near_candidates` chunks, with 3.0 TB full and 1.07 TB of it dedup intermediates. The post-mortem quantified why — MinHash at `num_perm=128, bands=32, rows_per_band=4` produces 1.58 billion bucket-row entries over 49 M docs, and per-band shards are kept *and* consolidated with no cleanup hook — and produced the standing rule of thumb: **budget 4–5× the canonical input size, ideally 6×**. 489 GB of already-consumed intermediates were identified as safely deletable (see [`../01_0_cleaning_iteration_and_thresholds/WAVE2_PIPELINE_RUN_2026-04-26.md`](../01_0_cleaning_iteration_and_thresholds/WAVE2_PIPELINE_RUN_2026-04-26.md) § 2026-04-27).

### 2026-04-28 onward — dedup stops being re-run

From wave 3 on, the pipeline deliberately stopped rediscovering duplicates and replayed the existing builder dedup metadata instead (`family_membership` replay). This is visible in the wave-3 plan ("Reuse existing dedup metadata for the builder. Do not rerun dedup clustering") and in the wave-4 driver, which points `DEDUP` at `runs/wave2_20260426/dedup_run/builder_metadata`. C3's `selected_input.parquet` was built the same way.

## Outcome

- **Dedup ran once at full scale** (wave 2) and its metadata bundle was frozen and replayed by waves 3, 4 and C3 — the exact stage's own numbers (49,292,755 → 49,090,905 kept) are the last full-corpus dedup figures the project has.
- **The repair worked**: SQLite was demoted to a control plane, exact-stage materialization moved to parquet shards, and resumability landed with tests (`test_resume_reuses_completed_exact_stage_even_if_progress_marker_was_overwritten`, `test_full_pipeline_resume_reuses_completed_stage_outputs`, `test_near_cluster_stage_can_resume_from_partial_component_chunks` in [`../../../tests/test_text_dedup.py`](../../../tests/test_text_dedup.py)).
- **One semantic change was made on purpose**: the near `length_ratio ≥ 0.70` gate was removed.
- **Not delivered as planned**: the repair plan's Phase-4 *golden equivalence* test — running old and repaired exact-stage implementations on the same input and requiring byte-equal outputs — has no counterpart in `tests/` (no test mentions "golden"). Resume and contract coverage exist; old-vs-new equivalence was never proven mechanically.
- **Left open** (from [`TODO.md`](TODO.md)): medium-scale resume tests for the repaired exact path, and recording which dedup metadata artifacts downstream consumers are allowed to rely on.

## Where things are

| Artifact | Role |
|---|---|
| [`scripts/parallel_exact_stages.py`](scripts/parallel_exact_stages.py) | The exact-stage repair — parallel strict+relaxed stages that bypass the serial SQLite rebuild. |
| [`scripts/publish_dedup_overlay_into_working_release.py`](scripts/publish_dedup_overlay_into_working_release.py) | Publishes `dedup_metadata/latest.json` into the working release — the contract boundary `01_2` consumes. |
| [`scripts/wait_for_hplt_integration_and_run_dedup.sh`](scripts/wait_for_hplt_integration_and_run_dedup.sh), [`scripts/wait_for_dedup_and_publish_overlay.sh`](scripts/wait_for_dedup_and_publish_overlay.sh) | The repo-owned worker wrappers verified end-to-end on 2026-04-15. |
| [`DEDUP_SCRIPT_REPAIR_PLAN.md`](DEDUP_SCRIPT_REPAIR_PLAN.md) | The design of the repair, including the semantics-preservation rules. |
| `../../../glossapi_corpus_cli/text_dedup.py` | The dedup implementation itself (repo-level pipeline code, not owned here). |

## Working documents

Historical, kept for traceability:

- [`TODO.md`](TODO.md) — the open action list at the archive move.
- Repo-level companions, all archived: [`../../../docs/_archive/PIPELINE_RECOVERY_AND_SCALE_PLAN.md`](../../../docs/_archive/PIPELINE_RECOVERY_AND_SCALE_PLAN.md) (freeze/salvage/upgrade/verify/resume phases), [`../../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md`](../../../docs/_archive/NEAR_DEDUP_REDESIGN_PLAN.md), [`../../../docs/_archive/NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md`](../../../docs/_archive/NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md), [`../../../docs/_archive/HF_DEDUP_INVESTIGATION.md`](../../../docs/_archive/HF_DEDUP_INVESTIGATION.md), and the E2E family ([`PIPELINE_E2E_STAGE_CHAIN`](../../../docs/_archive/PIPELINE_E2E_STAGE_CHAIN.md), [`_VERIFICATION_PLAN`](../../../docs/_archive/PIPELINE_E2E_VERIFICATION_PLAN.md), [`_VERIFICATION_TODO`](../../../docs/_archive/PIPELINE_E2E_VERIFICATION_TODO.md), [`_WORKER_RUN_REPORT_20260415`](../../../docs/_archive/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md), [`STAGE_VERIFICATION_CHECKLIST`](../../../docs/_archive/STAGE_VERIFICATION_CHECKLIST.md)).

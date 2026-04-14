# Current Status

## Active Phase

Parallel execution:
- tokenizer-spec freeze and GCP worker setup for the corrected HPLT rebuild
- tokenizer-data preparation from the corrected CPT-ready dataset once that slice exists
- plan-level separation of a cheap uploader-instance sidetrack for final HF publication

## What Is Settled

- the shipping method is merge-rule tokenizer extension, not whole-word `add_tokens(...)`
- Apertus compatibility is a hard constraint
- machine-translated HPLT content is excluded from the final training dataset
- `openarchives.gr` rows with `needs_ocr == true` must stay excluded from the CPT-ready dataset used for tokenizer work
- HPLT is being prepared for the upstream HF corpus dataset, not as a separate tokenizer-only corpus
- the downstream CPT/tokenizer builder is expected to stay lightweight after HF download
- the first discovery tokenizer runs are locked to `50k` vocab
- the mixed `GlossAPI + HPLT` tokenizer view is locked to `70/30` by training-token mass
- local tokenizer progress does not need to wait for the HF upload to finish once the filtered HPLT parquet slice exists locally
- final HF publication should happen from a separate cheap uploader instance using the official large-folder HF upload path
- the workspace has now been split into smaller subprojects
- the repo itself is now the canonical source for the active pipeline code, tests, and orchestration scripts

## What Exists Now

- the previously built `HPLT__ell_Grek_ge8_no_mt` slice is no longer sufficient for tokenizer/CPT use by itself
- the corrected HPLT slice must add a real `corpus.clean` pass and drop rows with `greek_badness_score > 60`
- the old score-only HF upload attempt has been stopped and should be treated as invalid
- the canonical Apertus constraints and tokenizer-extension direction are documented
- the old `add_tokens(...)` baseline is retained only as diagnostic background
- the repo now has uploader-handoff scripts under `ops/upload/` for:
  - validating a working release snapshot
  - preparing a cheap-uploader handoff manifest
  - launching or locally staging the uploader handoff
- the contract suite now covers:
  - synthetic stage-to-stage contracts
  - uploader-handoff contracts
  - tiny real-document smoke through HPLT build, integration, dedup, dedup overlay, mix build, uploader handoff, and tiny discovery training
- the repo-local validation matrix now also includes:
  - resumability regressions for dedup stage handoff
  - efficiency smoke coverage for streaming mix build and near-candidate execution
- the detailed per-stage verification state is tracked in:
  - [STAGE_VERIFICATION_CHECKLIST.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/STAGE_VERIFICATION_CHECKLIST.md)
- the plan has now explicitly diverged into an HF/DataTrove comparison for the near-dedup path:
  - [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)
- the active GCP tokenizer worker has been rearmed from the repo tree at:
  - `/home/foivos/Projects/glossapi-tokenizer-extension`
- the repo-backed worker chain currently includes:
  - dedup resume
  - dedup overlay watcher
  - mix watcher
  - training watcher
  - uploader handoff prep watcher
  - uploader launch watcher
- the current live dedup run has exposed a design bottleneck:
  - row-group chunk computation completed
  - the run is stuck in `stage_01_exact` finalization
  - the pathological path is the SQLite-heavy `relaxed_exact` export/finalization tail
- the recovery and repair plans are now tracked explicitly in:
  - [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
  - [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)
- the repair plans now explicitly treat semantic equivalence as the golden rule:
  - same dedup functionality
  - improved efficiency only
- the latest live scaling attempt established that the current near-candidate execution path is still not efficient enough:
  - `16` workers on `m3-megamem-64` (`976 GB`) drove memory to about `955 / 960 GB`
  - the stage still had `0 / 32` completed bands
  - the worker was stopped to preserve state and avoid guest instability
- the hard near-dedup `length_ratio` admission gate has been removed in the repo-backed code:
  - high-similarity short-vs-long pairs are now allowed to reach representative selection
  - `length_ratio` is still preserved as metadata and audit signal
- the first repo-backed worker stress checks have now run on the `m3-megamem-64` box:
  - worker-side correctness smoke for the efficiency harness passed
  - corrected worker-side stress comparison on `near_candidates` with `16,384` synthetic docs and `16` workers now shows:
    - `spawn`: `elapsed_seconds = 128.081`, `peak_total_child_pss_mb = 1542.139`
    - `fork`: `elapsed_seconds = 129.547`, `peak_total_child_pss_mb = 587.563`
  - this confirms the shared-state `fork` execution path materially reduces effective worker memory without hurting throughput on the tested workload
  - benchmark artifacts live on the worker under:
    - `/home/foivos/data/glossapi_work/perf_runs/near_candidates_redesign_20260414_v7/spawn/summary.json`
    - `/home/foivos/data/glossapi_work/perf_runs/near_candidates_redesign_20260414_v7/fork/summary.json`

## What Is Not Done Yet

- no frozen final HPLT filtering spec beyond the current working defaults
- no frozen HPLT upload-schema mapping writeup at the field-by-field level
- no rerun of the full prepared-source dataset with the corrected HPLT slice included as the finalized tokenizer/CPT input view
- no frozen downstream manifests for `GlossAPI-only` vs `GlossAPI + HPLT`
- no frozen held-out eval manifests derived from the refreshed upstream dataset
- no true Greek `BPE` discovery tokenizer
- no implemented merge-rule extension
- no model adaptation plan beyond high-level constraints
- no live armed cheap-instance uploader service yet for publishing the full updated dataset snapshot plus refreshed dedup metadata
- no repaired dedup exact-stage export path yet
- no finalized post-restart progress report for the repo-backed near-dedup continuation yet
- no replacement near-candidate implementation yet based on the HF/DataTrove investigation

## Current Trust Boundary

Active planning and execution should use:
- [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- the relevant subproject folders

Legacy baseline and exploratory material has been moved under:
- [legacy/](/home/foivos/Projects/glossapi-tokenizer-extension/legacy/README.md)

Execution note:
- `home` should not be used as a tokenizer worker
- tokenizer filtering/export/training workloads should run on GCP workers only
- GCP workers should be sized minimally for the step and stopped when done

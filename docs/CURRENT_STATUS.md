# Current Status

## Active Phase

**C3 cutoff decision.** C3
(`C3_wave2_broad_glossapi_plus_hplt_50_50`) has been converged to as the
shipping tokenizer arm — see [C3_CONVERGENCE.md](C3_CONVERGENCE.md).

The remaining tokenizer-side work is:
- assemble Apertus-compatible merged variants of C3 at each of the four
  frozen cutoffs `{10240, 15360, 20480, 25600}`
- run the intrinsic + fertility metric bundle on each merged variant
  across the four held-out evaluation slices (primary:
  `modern_greek_eval`)
- pick the cutoff at the elbow → freeze the shipped vocab size
- once frozen, hand off to `subprojects/02_2_tokenizer_implementation`
  for the merge-rule extension and then
  `subprojects/03_apertus_extension_and_embedding_adaptation` for the
  embedding + `lm_head` adaptation and CPT
- a pre-extension diagnostic of how Apertus already represents Greek
  on its E + U matrices is complete under
  `subprojects/03_apertus_extension_and_embedding_adaptation/03_1_greek_embedding_diagnostic/`
  (Greek-vs-¬Greek geometry, hull occupancy, binary classifier,
  morphological clustering, cross-language semantic baseline); the
  init-method LOO benchmark that was started under the v4 plan was
  found methodologically flawed and is archived

## What Is Settled

- the shipping method is merge-rule tokenizer extension, not whole-word `add_tokens(...)`
- Apertus compatibility is a hard constraint
- machine-translated HPLT content is excluded from the final training dataset
- `openarchives.gr` rows with `needs_ocr == true` must stay excluded from the CPT-ready dataset used for tokenizer work
- HPLT is being prepared for the upstream HF corpus dataset, not as a separate tokenizer-only corpus
- the downstream CPT/tokenizer builder is expected to stay lightweight after HF download
- the converged tokenizer arm is **C3** — continuous BPE from Apertus
  on `GlossAPI + HPLT` at `50 / 50` by training-token mass, trained on
  the wave-2 broad cleaner output, base `131072` + added `25600` =
  total `156672`
- the four-arm exploration (`F1`, `F2`, `C1`, `C2`) is closed; those
  arms are retained as analyzed baselines only
- the cutoff grid on C3's added units is frozen at
  `{10240, 15360, 20480, 25600}`; the shipped cutoff is the only open
  tokenizer-side decision
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
- the active memory-footprint reduction checklist for the near-dedup hot path is tracked in:
  - [NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md)
- the active GCP tokenizer worker has been rearmed from the repo tree at:
  - `/home/foivos/Projects/glossapi-tokenizer-extension`
- the repo-backed worker chain currently includes:
  - dedup overlay watcher
  - mix watcher
  - training watcher
  - uploader handoff prep watcher
  - uploader launch watcher
- dedup is now finished on the active worker chain:
  - final dedup outputs exist under `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/runs/exact_stage_20260413T025237Z/final`
  - builder metadata bundle exists under `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/runs/exact_stage_20260413T025237Z/builder_metadata`
- the live full-size downstream chain is no longer blocked by a dead overlay path:
  - overlay publish completed into `/home/foivos/data/glossapi_work/hf_release_publish_working/dedup_metadata/latest.json`
  - uploader handoff prep completed into `/home/foivos/data/glossapi_work/uploader_handoff_20260414`
  - uploader-ready local stage completed with `/home/foivos/data/glossapi_work/uploader_handoff_20260414/launch_summary.json`
  - the active remaining bottleneck is tokenizer mix build under `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413`
- the recovery and repair plans are now tracked explicitly in:
  - [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
  - [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/_archive/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)
- the repair plans now explicitly treat semantic equivalence as the golden rule:
  - same dedup functionality
  - improved efficiency only
- the failed pre-redesign `16`-worker attempt is now historical context, not the current live shape:
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
- the first end-to-end memory reduction is now in the repo-backed near-candidate path:
  - `build_candidate_band_chunk(...)` now streams one bucket at a time from the band shard inputs
  - candidate-pair rows, bucket summaries, and touched-doc outputs are written incrementally instead of being accumulated fully in memory
  - this reduces peak per-worker memory without changing dedup decisions
- the second near-candidate memory reduction is also now in the repo-backed path:
  - `near_candidates` no longer checkpoints only whole bands
  - the stage now partitions each band into bucket-hash prefix member shards and checkpoints `band + prefix` chunks
  - this should reduce time-to-first-durable-progress and lower the amount of work lost on interruption
- uploader handoff no longer has to wait for dedup to exist:
  - the repo now supports `source_only` handoff preparation for the corrected HPLT/source release snapshot
  - uploader staging now respects the manifest `sync_paths` instead of blindly syncing the entire working release root
- because the cheap uploader host is currently unreachable, source-only HF publication is temporarily running from the active GCP worker:
  - staged source-only root:
    - `/home/foivos/data/glossapi_work/hf_upload_stage_source_only/hf_release_publish`
  - live upload log:
    - `/home/foivos/data/glossapi_work/logs/hf_upload_source_only_20260414.log`
  - this is an operational fallback, not the intended permanent uploader topology
- builder replay has one important efficiency guard now in place:
  - when `builder_metadata_v2` exports family membership, builder replay no longer loads `near_candidate_pairs.parquet` unnecessarily
  - `near_candidate_pairs.parquet` is still intentionally retained as an evidence/audit artifact in the exported bundle
- the downstream builder/tokenizer efficiency plan is now tracked explicitly in:
  - [BUILDER_TOKENIZER_EFFICIENCY_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/BUILDER_TOKENIZER_EFFICIENCY_PLAN.md)
- the explicit recovery plan for validating the real worker-side chain after dedup is now tracked in:
  - [PIPELINE_E2E_VERIFICATION_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_PLAN.md)
  - [PIPELINE_E2E_VERIFICATION_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_TODO.md)
  - [PIPELINE_E2E_STAGE_CHAIN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_STAGE_CHAIN.md)
  - [PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)
  - [PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md)
  - [PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md)
- the repo-owned downstream chain has now been truly exercised on the worker through tokenizer training completion on a bounded real-doc smoke run:
  - smoke root:
    - `/home/foivos/data/glossapi_work/smoke_runs/e2e_verify_20260415T085623Z`
  - report:
    - [PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)
- the remaining downstream audit result is now explicit:
  - mix build is the main serial throughput bottleneck
  - tokenizer training is the main progress-transparency gap

## What Is Not Done Yet

- no frozen final HPLT filtering spec beyond the current working defaults
- no frozen HPLT upload-schema mapping writeup at the field-by-field level
- no rerun of the full prepared-source dataset with the corrected HPLT slice included as the finalized tokenizer/CPT input view
- no frozen downstream manifests for `GlossAPI-only` vs `GlossAPI + HPLT`
- no frozen held-out eval manifests derived from the refreshed upstream dataset
- no true Greek `BPE` discovery tokenizer
- no continuous-`BPE` comparison run from the Apertus base tokenizer yet
- no implemented merge-rule extension
- no model adaptation plan beyond high-level constraints
- no live armed cheap-instance uploader service yet for publishing the full updated dataset snapshot plus refreshed dedup metadata
- no completed worker-side source-only HF upload run yet for the corrected HPLT/source snapshot
- no frozen production answer yet for whether the full-size live training launcher should stay inline or return to the canonical dual-service path
- no downstream per-stage structured progress JSON or trace contracts yet for:
  - mix build
  - tokenizer training
  - uploader local stage / launch

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

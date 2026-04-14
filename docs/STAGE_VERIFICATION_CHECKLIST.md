# Stage Verification Checklist

This checklist records what has actually been verified for each active stage of the tokenizer pipeline.

Legend:
- `[x]` verified to the current working bar
- `[ ]` not yet verified to the required bar

Important rule:
- small-sample correctness and contract coverage do not imply large-scale operational validation
- a stage is only "throughput verified" once it has been exercised under realistic CPU/RAM pressure for that stage

## Pipeline Whole

- [x] Tiny real-document end-to-end run exists.
  - Covered path:
    - HPLT build
    - HPLT integration
    - dedup
    - dedup overlay publish
    - mix build
    - uploader handoff
    - tiny discovery tokenizer training
  - Main harness:
    - `ops/smoke/run_real_docs_e2e_smoke.py`
  - Repo-local artifact used during recent validation:
    - `/home/foivos/data/glossapi_work/smoke_runs/dedup_resume_fix_20260414/smoke_summary.json`

- [ ] Full live-scale end-to-end run is verified.
  - Current blocker:
    - live run is still in `stage_02_near` / `near_candidates`
  - Missing proof:
    - `latest_success.json`
    - dedup overlay refresh
    - full tokenizer mix generation
    - full 50k tokenizer training
    - actual uploader launch completion

## Stage 1: HPLT Filtering / Canonical Slice Build

- [x] Small test that the stage works.
  - Contract smoke:
    - `tests/test_pipeline_contracts.py::test_build_hplt_slice_contract_smoke`
  - Tiny real-doc smoke:
    - `ops/smoke/run_real_docs_e2e_smoke.py`

- [x] Previous/next-stage contracts are verified.
  - Input contract:
    - raw `jsonl.zst` HPLT shard input
  - Output contract:
    - canonical source-parquet schema expected by the working release and downstream builder
  - Verified properties:
    - dataset name
    - `source_doc_id`
    - `source_metadata_json`
    - MT exclusion
    - empty-row exclusion

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - shard fanout
    - `--workers`
    - `--batch-size`
    - `--rows-per-part`
    - `--clean-num-threads`
    - `Corpus.clean` RSS per batch
  - Current state:
    - small real-doc smoke exists
    - live large run existed
    - no dedicated medium/high-throughput benchmark has been run from the repo-local harness

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - shard-level concurrency
    - cleaner thread count vs worker count
    - CPU saturation without memory blow-up
  - Current state:
    - known earlier underutilization on tail shards
    - no explicit throughput sweep proving the highest safe setting

## Stage 2: HPLT Integration Into Working Release

- [x] Small test that the stage works.
  - Contract smoke:
    - `tests/test_pipeline_contracts.py::test_integrate_hplt_slice_refreshes_working_release`

- [x] Previous/next-stage contracts are verified.
  - Input contract:
    - corrected HPLT source-parquet slice
  - Output contract:
    - working release with stale HPLT slice replaced
    - refreshed `row_counts.csv`
    - integration summary JSON

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - replacement of multiple parquet parts
    - metadata refresh cost
    - optional validation / manifest rebuild cost
  - Current state:
    - correctness verified
    - no dedicated large-snapshot benchmark

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - copy speed
    - metadata rebuild parallelism
  - Current state:
    - stage is mostly serial and I/O-light
    - no throughput tuning or benchmark recorded

## Stage 3: Dedup Exact Stage

- [x] Small test that the stage works.
  - Exact-stage regression:
    - `tests/test_text_dedup.py::test_exact_dedup_run_strict_and_relaxed`

- [x] Previous/next-stage contracts are verified.
  - Verified contracts:
    - exact memberships
    - exact survivor outputs
    - resume behavior for completed exact stage
  - Relevant tests:
    - `tests/test_text_dedup.py::test_resume_reuses_completed_exact_stage_even_if_progress_marker_was_overwritten`
    - `tests/test_text_dedup.py::test_full_pipeline_resume_reuses_completed_stage_outputs`

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - parquet-backed exact memberships
    - `run_docs_inventory` export
    - strict exact materialization
    - relaxed exact partitioning by hash prefix
  - Current state:
    - design repaired to remove the worst SQLite/WAL failure mode
    - small-sample correctness verified
    - no fresh medium/high-throughput benchmark on the repaired exact path

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - exact export parallelism
    - DuckDB thread count
    - serial tail behavior
  - Current state:
    - exact still has serial segments
    - no measured "max safe throughput" sweep

## Stage 4: Dedup Near Candidates

- [x] Small test that the stage works.
  - Small efficiency smoke:
    - `ops/perf/run_efficiency_smoke.py`
    - `tests/test_efficiency_smoke.py`

- [x] Previous/next-stage contracts are verified on small runs.
  - Verified contracts:
    - exact survivors feed near-candidate generation
    - candidate outputs exist in the form needed by near clustering
  - Relevant tests:
    - `tests/test_text_dedup.py::test_full_dedup_pipeline_runs_stage_2_only_on_exact_survivors`
    - `tests/test_text_dedup.py::test_near_cluster_stage_can_resume_from_partial_component_chunks`

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - `bands`
    - `rows_per_band`
    - `max_bucket_size`
    - candidate-pair shard aggregation
    - per-worker RSS during band processing
  - Current state:
    - conservative worker cap (`8`) exists
    - live run is currently exercising this stage
    - no completed high-throughput benchmark proving the stage shape is good enough

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - worker cap vs available CPU
    - first-band completion latency
    - whether `8` workers is too conservative
  - Current state:
    - live run shows 8 hot workers
    - no proof yet that this is the fastest safe setting
    - current progress instrumentation is too coarse to prove throughput quality

## Stage 5: Dedup Near Clusters / Final Dedup Outputs

- [x] Small test that the stage works.
  - Resume and partial-cluster test:
    - `tests/test_text_dedup.py::test_near_cluster_stage_can_resume_from_partial_component_chunks`

- [x] Previous/next-stage contracts are verified on small runs.
  - Verified contracts:
    - near-candidate outputs feed near-cluster stage
    - final dedup outputs feed overlay export and builder metadata
  - Relevant tests:
    - `tests/test_text_dedup.py::test_full_pipeline_resume_reuses_completed_stage_outputs`

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - component graph size
    - touched-doc filtering
    - candidate-pair database/parquet aggregation
  - Current state:
    - only small-sample verification exists
    - no medium/high-throughput benchmark recorded

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - cluster chunk fanout
    - component scheduling
    - aggregation tail
  - Current state:
    - no explicit utilization benchmark

## Stage 6: Dedup Overlay Publish Into Working Release

- [x] Small test that the stage works.
  - Contract smoke:
    - `tests/test_pipeline_contracts.py::test_publish_dedup_overlay_contract`

- [x] Previous/next-stage contracts are verified.
  - Input contract:
    - dedup `latest_success.json`
    - builder metadata root
  - Output contract:
    - refreshed `dedup_metadata/latest.json`
    - code bundle / metadata bundle where expected by later stages

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - builder metadata bundle size
    - code bundle copy cost
  - Current state:
    - correctness verified
    - no large-bundle benchmark recorded

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - mostly I/O throughput, not CPU
  - Current state:
    - no throughput benchmark recorded

## Stage 7: Tokenizer Mix Build

- [x] Small test that the stage works.
  - Tiny end-to-end smoke covers mix generation
  - Small efficiency smoke covers streaming mix path:
    - `ops/perf/run_efficiency_smoke.py`
    - `tests/test_efficiency_smoke.py`

- [x] Previous/next-stage contracts are verified on small runs.
  - Verified contracts:
    - dedup metadata is consumed correctly
    - `openarchives.gr needs_ocr == true` exclusion survives
    - resulting `mix.parquet` feeds tokenizer training
  - Relevant coverage:
    - `tests/test_pipeline_contracts.py`
    - tiny real-doc smoke

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - streaming vs full in-memory path
    - source-mix char-budget enforcement
    - dedup replay on duplicate-family subset
  - Current state:
    - streaming path exists and passes small smoke
    - no medium/high-throughput benchmark on the worker-class dataset

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - whether DuckDB / parquet scan parallelism is tuned well
    - whether the chosen path saturates available CPU without excessive RAM
  - Current state:
    - no worker-class throughput sweep recorded

## Stage 8: Discovery Tokenizer Training

- [x] Small test that the stage works.
  - Tiny discovery training is covered by the real-doc smoke harness

- [x] Previous/next-stage contracts are verified on small runs.
  - Verified contracts:
    - `mix.parquet` input is consumable by training script
    - training output directories and summaries are produced
    - front-end compatibility check is exercised in the tiny smoke path

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - iterator-driven training
    - parquet read throughput
    - tokenizer trainer memory footprint at `50k`
  - Current state:
    - small training smoke only
    - no medium/high-throughput benchmark from the canonical repo on the GCP worker

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - `RAYON_NUM_THREADS`
    - `TOKENIZERS_PARALLELISM`
    - parallel two-run scheduling on the worker
  - Current state:
    - planned values exist
    - full throughput validation not done

## Stage 9: Uploader Handoff Preparation

- [x] Small test that the stage works.
  - Contract smoke and tiny end-to-end smoke cover handoff manifest creation

- [x] Previous/next-stage contracts are verified on small runs.
  - Verified contracts:
    - working release snapshot is validated
    - handoff manifest contains correct remote upload command and payload paths

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - large working-release manifest validation cost
    - handoff payload size assumptions
  - Current state:
    - no large-snapshot benchmark recorded

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - mostly I/O and manifest generation
  - Current state:
    - no throughput benchmark recorded

## Stage 10: Uploader Launch / HF Upload

- [x] Small launch-path test exists.
  - Local staging / detached-launch path has been smoke tested
  - Cheap uploader instance has been provisioned and basic command path checked

- [ ] Previous/next-stage contracts are fully verified.
  - Verified so far:
    - handoff manifest contract
    - remote-launch command construction
  - Missing:
    - full remote HF upload completion from the cheap uploader instance as part of the current repo-owned flow

- [ ] Scalable-work design is fully verified under CPU/RAM constraints.
  - Internals that matter:
    - rsync/sync cost for large release snapshot
    - HF `upload_large_folder(...)` behavior
    - Xet/non-Xet memory and bandwidth behavior
  - Current state:
    - no large real upload benchmark in the canonical repo flow

- [ ] Maximum safe compute utilization is verified.
  - Internals that matter:
    - uploader CPU count
    - HF worker count
    - sync/upload overlap policy
  - Current state:
    - no throughput tuning benchmark recorded

## Current Bottom Line

What is genuinely verified right now:
- small correctness
- small contract compatibility
- small resumability
- small real-doc end-to-end behavior
- small efficiency smokes for mix build and near-candidate execution

What is not yet genuinely verified right now:
- high-throughput behavior for the active live stages
- full live-scale end-to-end completion of the repo-backed pipeline
- maximum safe compute utilization for each heavy stage

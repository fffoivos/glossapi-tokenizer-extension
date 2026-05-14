# Pipeline E2E Stage Chain

## Purpose

This file is the canonical worker-side stage chain for true end-to-end verification.

It exists to prevent a repeat of the recent failure where:
- dedup finished
- but the downstream chain was still pointing at a dead overlay path
- so tokenizer work never started

The commands here are the real repo-owned entrypoints that should be used for worker validation.

## Worker Context

Current worker-side roots used for verification:

- working release root:
  - `/home/foivos/data/glossapi_work/hf_release_publish_working`
- dedup state root:
  - `/home/foivos/data/glossapi_work/analysis/dedup/text_publish/state/gcp_refresh_20260413`
- tokenizer mix root:
  - `/home/foivos/data/glossapi_work/tokenizer_mixes_20260413`
- tokenizer training root:
  - `/home/foivos/data/glossapi_work/tokenizer_training_runs_20260413`
- uploader handoff root:
  - `/home/foivos/data/glossapi_work/uploader_handoff_20260414`

## Canonical Chain

### 1. Dedup

- wrapper:
  - [wait_for_hplt_integration_and_run_dedup.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/wait_for_hplt_integration_and_run_dedup.sh)
- required input:
  - `${working_release_root}/hplt_integration_summary.json`
- first progress signal:
  - dedup process exists
  - progress files appear under `${run_root}/progress/`
- completion marker:
  - `${state_root}/latest_success.json`
- stall threshold:
  - stage-specific; must use dedup progress DB, progress JSON, and trace files
- restart rule:
  - resume from the same `run_root` and `state_root`

### 2. Dedup Overlay Publish

- wrapper:
  - [wait_for_dedup_and_publish_overlay.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/wait_for_dedup_and_publish_overlay.sh)
- underlying script:
  - [publish_dedup_overlay_into_working_release.py](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/publish_dedup_overlay_into_working_release.py)
- required input:
  - `${state_root}/latest_success.json`
- first progress signal:
  - overlay publish process exists
- completion artifacts:
  - `${working_release_root}/dedup_metadata/latest.json`
  - `${working_release_root}/dedup_metadata/<run_id>/publish_summary.json`
- stall threshold:
  - no output artifact and no active overlay process after expected short publish window
- restart rule:
  - rerun the wrapper directly; it is idempotent for the latest successful run

### 3. Tokenizer Mix Build

- wrapper:
  - [wait_for_dedup_overlay_and_build_tokenizer_mixes.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh)
- required inputs:
  - `${working_release_root}/dedup_metadata/latest.json`
  - `${state_root}/latest_success.json`
  - `${working_release_root}/hplt_integration_summary.json`
- first progress signal:
  - `glossapi_corpus_cli.cli mix` process exists
  - first temp parquet appears under `${mix_root}/...`
- completion artifacts:
  - `${mix_root}/glossapi_only/mix.parquet`
  - `${mix_root}/glossapi_plus_hplt_70_30/mix.parquet`
- stall threshold:
  - no active mix process and no advancing temp/output files
- restart rule:
  - rerun the wrapper; it rebuilds the two mix outputs from the published overlay

### 4. Tokenizer Training

- wrapper:
  - [wait_for_tokenizer_mixes_and_launch_training.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/wait_for_tokenizer_mixes_and_launch_training.sh)
- required inputs:
  - `${mix_root}/glossapi_only/mix.parquet`
  - `${mix_root}/glossapi_plus_hplt_70_30/mix.parquet`
- first progress signal:
  - training process exists or training service exists
  - output directory is created
- caveat:
  - the current training log is not a trustworthy mid-run progress signal because `training_summary.json` is only written at the end
- completion artifacts:
  - `${training_root}/glossapi_only_50k/training_summary.json`
  - `${training_root}/glossapi_plus_hplt_70_30_50k/training_summary.json`
- default launch mode:
  - `systemd`
- verification launch mode used for direct worker checks when needed:
  - `inline`
- stall threshold:
  - no active training process and no advancing logs or summary files
- restart rule:
  - rerun the wrapper from the same mix root and training root

### 5. Uploader Handoff Prep

- wrapper:
  - [wait_for_dedup_overlay_and_prepare_handoff.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_dedup_overlay_and_prepare_handoff.sh)
- required inputs:
  - `${working_release_root}/dedup_metadata/latest.json`
  - `${state_root}/latest_success.json`
  - `${working_release_root}/hplt_integration_summary.json`
- first progress signal:
  - handoff-prep process exists
- completion artifacts:
  - `${handoff_root}/uploader_handoff.json`
  - `${handoff_root}/handoff_summary.json`
  - `${handoff_root}/remote_upload_command.txt`
- stall threshold:
  - no handoff artifact and no active prep process
- restart rule:
  - rerun the wrapper directly

### 6. Uploader Launch Or Uploader-Ready Local Stage

- wrapper:
  - [wait_for_uploader_handoff_and_launch.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_uploader_handoff_and_launch.sh)
- underlying launcher:
  - [launch_hf_uploader_handoff.py](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/launch_hf_uploader_handoff.py)
- required input:
  - `${handoff_root}/uploader_handoff.json`
- first progress signal:
  - launcher process exists
  - local stage root or remote sync begins to materialize
- caveat:
  - the current launcher has no structured progress file; mid-run visibility comes from staged-file growth and process existence
- completion artifact for uploader-ready local verification:
  - `${handoff_root}/launch_summary.json`
- completion artifact for local staged subset verification:
  - `${local_stage_root}/<remote_release_root_basename>/...`
- stall threshold:
  - no launch summary and no active launcher process
- restart rule:
  - rerun the wrapper directly

## Verification Rule

The chain is not considered end-to-end validated unless:

1. each stage above is started from the real previous artifact
2. each stage shows real first progress
3. each stage produces its real completion artifact
4. tokenizer training actually launches after mix completion
5. uploader handoff prep actually runs from the published overlay, not from assumptions

## Verified On 2026-04-15

The worker-side chain is now verified in two layers:

1. live large-chain verification from the real dedup-complete worker state:
- overlay publish completed
- uploader handoff prep completed
- uploader-ready local stage completed
- full-size mix build started and produced durable output

2. bounded real-doc worker smoke verification:
- completed through mix build
- completed through tokenizer training
- completed through uploader-ready local stage

Reference:

- [PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)

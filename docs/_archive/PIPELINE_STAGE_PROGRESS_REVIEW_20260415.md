# Pipeline Stage Progress Review 2026-04-15

## Scope

This review checks whether the remaining post-dedup worker stages are transparent enough to answer:

- is it running
- is it progressing
- is it stalled
- is it complete

## Findings By Stage

### 1. Dedup Overlay Publish

Code paths:

- [wait_for_dedup_and_publish_overlay.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/wait_for_dedup_and_publish_overlay.sh)
- [publish_dedup_overlay_into_working_release.py](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/publish_dedup_overlay_into_working_release.py)

Current progress signals:

- process existence
- completion artifacts:
  - `dedup_metadata/latest.json`
  - `publish_summary.json`

Missing:

- no progress file
- no phase field
- no trace log
- no intermediate milestone output

Assessment:

- weak transparency
- acceptable only because the stage is short

### 2. Tokenizer Mix Build

Code paths:

- [wait_for_dedup_overlay_and_build_tokenizer_mixes.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh)
- [pipeline.py](/home/foivos/Projects/glossapi-tokenizer-extension/glossapi_corpus_cli/pipeline.py)

Current progress signals:

- process existence
- temp parquet growth:
  - `.filtered_input.parquet.tmp`
- final artifacts:
  - `mix.parquet`
  - `mix.summary.csv`
  - `mix.source_mix_summary.json`

Missing:

- no progress JSON
- no phase field for:
  - filtered-input materialization
  - dedup replay
  - source-mix selection
  - final summary write
- no total/completed units
- no trace log
- no stall heartbeat beyond file-size inference

Assessment:

- fails transparency review
- the live stage is observable only by process sampling and temp-file growth

### 3. Tokenizer Training

Code paths:

- [wait_for_tokenizer_mixes_and_launch_training.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/wait_for_tokenizer_mixes_and_launch_training.sh)
- [train_discovery_tokenizer.py](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py)

Current progress signals:

- process or service existence
- output directory creation
- completion artifact:
  - `training_summary.json`

Important limitation:

- the script prints JSON only at the end
- there is no periodic training heartbeat
- the training log can remain silent for most or all of the run

Assessment:

- fails transparency review
- current watcher logic can detect start and finish, but not meaningful mid-run progress

### 4. Uploader Handoff Prep

Code paths:

- [wait_for_dedup_overlay_and_prepare_handoff.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_dedup_overlay_and_prepare_handoff.sh)

Current progress signals:

- process existence
- completion artifacts:
  - `uploader_handoff.json`
  - `handoff_summary.json`
  - `remote_upload_command.txt`

Missing:

- no progress file
- no trace file
- no intermediate state marker while the shell is waiting on inputs

Assessment:

- weak transparency
- acceptable only because the actual preparation step is short

### 5. Uploader Launch / Uploader-Ready Local Stage

Code paths:

- [wait_for_uploader_handoff_and_launch.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_uploader_handoff_and_launch.sh)
- [launch_hf_uploader_handoff.py](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/launch_hf_uploader_handoff.py)

Current progress signals:

- process existence
- staged files appearing under the local stage root or remote release root
- completion artifact:
  - `launch_summary.json`

Missing:

- no explicit progress file
- no phase field for:
  - local staging
  - remote mkdir
  - rsync
  - remote launch
- no unit counts or byte totals

Assessment:

- weak transparency
- large stages can only be inferred from filesystem growth

## Summary

Progress verdicts:

- overlay publish:
  - weak but tolerable because short
- tokenizer mix build:
  - fails transparency review
- tokenizer training:
  - fails transparency review
- uploader handoff prep:
  - weak but tolerable because short
- uploader launch / local stage:
  - weak and needs explicit progress if used for large payloads

## Priority

Progress instrumentation should be added in this order:

1. tokenizer mix build
2. tokenizer training
3. uploader launch / local stage
4. overlay publish and handoff prep only if they grow enough to justify it

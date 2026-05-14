# Pipeline Stage Parallelism Review 2026-04-15

## Scope

This review covers the remaining post-dedup worker stages:

1. dedup overlay publish
2. tokenizer mix build
3. tokenizer training
4. uploader handoff prep
5. uploader launch or uploader-ready local stage

The goal is to identify:

- hidden serial sections
- fake parallelism
- underfed workers
- oversized finalization tails

## Findings

### 1. Dedup Overlay Publish

Code paths:

- [wait_for_dedup_and_publish_overlay.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/wait_for_dedup_and_publish_overlay.sh)
- [publish_dedup_overlay_into_working_release.py](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/publish_dedup_overlay_into_working_release.py)

Current behavior:

- fully serial
- copies builder metadata and run summary into the working release snapshot
- no worker pool and no chunking

Assessment:

- acceptable serial stage
- this is a short bounded publication step, not a throughput hotspot

Verdict:

- acceptable

### 2. Tokenizer Mix Build

Code paths:

- [wait_for_dedup_overlay_and_build_tokenizer_mixes.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/scripts/wait_for_dedup_overlay_and_build_tokenizer_mixes.sh)
- [pipeline.py](/home/foivos/Projects/glossapi-tokenizer-extension/glossapi_corpus_cli/pipeline.py)

Key structural issues:

- the wrapper builds the two mixes strictly one after another:
  - `glossapi_only`
  - `glossapi_plus_hplt_70_30`
- the filtering path is front-loaded by `materialize_filtered_mix_input(...)`
- `iter_filtered_mix_frames(...)` scans `data/*.parquet` serially in Python batch by batch
- only after that filtered parquet is materialized do the downstream dedup-replay and source-mix steps run

Live evidence from the active worker process:

- process:
  - `566412`
- observed shape:
  - one active process
  - `150` threads present, but only about `102%` CPU
- measured progress sample:
  - `.filtered_input.parquet.tmp` grew from `8,260,301,129` to `8,685,521,420` bytes over `15s`
  - process consumed about `15.35` CPU-seconds over the same `15s`

Interpretation:

- this is not meaningfully parallel at the stage level
- thread count is misleading here; runtime throughput still looks like roughly one hot core
- the wrapper also serializes the two mixes, which doubles the end-to-end wall clock before training can even start

Verdict:

- needs parallelization
- main remaining throughput bottleneck

### 3. Tokenizer Training

Code paths:

- [wait_for_tokenizer_mixes_and_launch_training.sh](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/wait_for_tokenizer_mixes_and_launch_training.sh)
- [train_discovery_tokenizer.py](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py)

Canonical wrapper behavior:

- default launch mode is `systemd`
- in that mode, the two training runs are launched as separate user services

Verification-run nuance:

- the live downstream rearm used `TOKENIZER_TRAINING_LAUNCH_MODE=inline`
- that verification override serializes the two training runs in a single shell

Trainer behavior:

- the training script is a single Python process
- the heavy tokenizer work is delegated to the fast tokenizer backend
- parallelism is internal via:
  - `TOKENIZERS_PARALLELISM`
  - `RAYON_NUM_THREADS`

Assessment:

- the canonical wrapper can run the two corpus views in parallel
- the core training loop is not obviously the next redesign target
- the main issue here is observability, not raw parallelism

Verdict:

- acceptable in canonical mode
- verification-inline mode is intentionally serial
- needs instrumentation more than redesign

### 4. Uploader Handoff Prep

Code paths:

- [wait_for_dedup_overlay_and_prepare_handoff.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_dedup_overlay_and_prepare_handoff.sh)

Current behavior:

- serial wait-and-prepare step
- validates overlay/dedup agreement and writes handoff artifacts

Assessment:

- short bounded control-plane stage
- not a throughput hotspot

Verdict:

- acceptable

### 5. Uploader Launch / Uploader-Ready Local Stage

Code paths:

- [wait_for_uploader_handoff_and_launch.sh](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/wait_for_uploader_handoff_and_launch.sh)
- [launch_hf_uploader_handoff.py](/home/foivos/Projects/glossapi-tokenizer-extension/ops/upload/launch_hf_uploader_handoff.py)

Current behavior:

- local staging is serial filesystem work
- `stage_release_subset(...)` walks the manifest `sync_paths` and hardlinks or copies the staged files
- remote launch path is one serial rsync + remote detach command

Assessment:

- the stage can still be long for a large `data/` tree
- but the main cost is I/O, not missing worker parallelism
- the more important gap is progress visibility

Verdict:

- acceptable for now
- needs better progress reporting before it needs a parallel redesign

## Summary

Stage verdicts:

- dedup overlay publish:
  - acceptable
- tokenizer mix build:
  - needs parallelization
- tokenizer training:
  - acceptable in canonical mode, needs instrumentation
- uploader handoff prep:
  - acceptable
- uploader launch / local stage:
  - acceptable, needs instrumentation

## Priority

The next remaining throughput work should target tokenizer mix build first.

Why:

- it is the longest clearly serial post-dedup stage
- it blocks tokenizer launch entirely
- the live worker evidence shows one hot process instead of meaningful stage-level parallelism

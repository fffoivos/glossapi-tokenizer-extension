# Project Index

This repo is intentionally split into smaller subprojects.

## Parallel Tracks

### Tokenizer Critical Path

1. [01_1_corpus_dedup](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/README.md)
2. [01_2_training_dataset_mix](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/README.md)
3. [02_apertus_tokenizer_spec](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_apertus_tokenizer_spec/README.md)
4. [02_1_tokenizer_experiments](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/README.md)
5. [02_2_tokenizer_implementation](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_2_tokenizer_implementation/README.md)
6. [03_apertus_extension_and_embedding_adaptation](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/README.md)

### Dataset Operational Sidetrack

1. [01_hplt_filtering](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/README.md)
2. [01_1_corpus_dedup](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/README.md)

## Canonical Sources Of Truth

- global decisions:
  - [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- current status:
  - [CURRENT_STATUS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/CURRENT_STATUS.md)
- active backlog:
  - [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- stage verification checklist:
  - [STAGE_VERIFICATION_CHECKLIST.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/STAGE_VERIFICATION_CHECKLIST.md)
- HF dedup comparison and diversion note:
  - [HF_DEDUP_INVESTIGATION.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/HF_DEDUP_INVESTIGATION.md)
- functional issues TODO:
  - [FUNCTIONAL_ISSUES_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/FUNCTIONAL_ISSUES_TODO.md)
- near dedup memory-footprint TODO:
  - [NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md)
- near dedup redesign plan:
  - [NEAR_DEDUP_REDESIGN_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/NEAR_DEDUP_REDESIGN_PLAN.md)
- builder/tokenizer efficiency plan:
  - [BUILDER_TOKENIZER_EFFICIENCY_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/BUILDER_TOKENIZER_EFFICIENCY_PLAN.md)
- pipeline e2e verification plan:
  - [PIPELINE_E2E_VERIFICATION_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_PLAN.md)
- pipeline e2e verification todo:
  - [PIPELINE_E2E_VERIFICATION_TODO.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_VERIFICATION_TODO.md)
- pipeline e2e stage chain:
  - [PIPELINE_E2E_STAGE_CHAIN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_STAGE_CHAIN.md)
- pipeline e2e worker run report:
  - [PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md)
- pipeline stage parallelism review:
  - [PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md)
- pipeline stage progress review:
  - [PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md)
- machine-readable config:
  - [apertus_greek_extension.yaml](/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml)
- legacy material that should not drive new execution:
  - [legacy/README.md](/home/foivos/Projects/glossapi-tokenizer-extension/legacy/README.md)

## Current Status

- the filtered HPLT local slice already exists in the canonical source-parquet tree on `home`
- the HF upload is an operational sidetrack, not the critical path for tokenizer work
- dedup is complete and the active blocker is the long serial tokenizer mix stage
- no true Greek `BPE` tokenizer has been trained yet
- no merge-rule Apertus extension has been built yet
- the worker-side downstream chain is now truly verified through tokenizer training on a bounded real-doc smoke run
- the next hard gate on the tokenizer path is improving mix throughput and stage transparency on the full-size live chain

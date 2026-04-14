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
- machine-readable config:
  - [apertus_greek_extension.yaml](/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml)
- legacy material that should not drive new execution:
  - [legacy/README.md](/home/foivos/Projects/glossapi-tokenizer-extension/legacy/README.md)

## Current Status

- the filtered HPLT local slice already exists in the canonical source-parquet tree on `home`
- the HF upload is an operational sidetrack, not the critical path for tokenizer work
- the active blocker is dedup exact-stage repair and salvage, not tokenizer training logic
- no true Greek `BPE` tokenizer has been trained yet
- no merge-rule Apertus extension has been built yet
- the next hard gate on the tokenizer path is resuming dedup with unchanged semantics, then freezing local manifests and the literal Apertus tokenizer-replication checklist

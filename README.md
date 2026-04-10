# GlossAPI Tokenizer Extension

Clean workspace for extending `swiss-ai/Apertus-8B-2509` for Greek.

The active method is:
- discover Greek morphology-respecting units through true `BPE` training
- replicate Apertus tokenization behavior as exactly as possible
- build a diverse, deduplicated training corpus before any tokenizer training
- extend Apertus through `model.vocab` and `model.merges`, not `add_tokens(...)`

The old whole-word `add_tokens(...)` sweep is retained only as a legacy baseline and has been moved out of the active planning path.

## Canonical Files

- project index:
  - [PROJECT_INDEX.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PROJECT_INDEX.md)
- global decisions:
  - [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- current status:
  - [CURRENT_STATUS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/CURRENT_STATUS.md)
- active backlog:
  - [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- machine-readable config:
  - [apertus_greek_extension.yaml](/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml)

## Subprojects

- [01_hplt_filtering](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/README.md)
- [01_1_corpus_dedup](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/README.md)
- [01_2_training_dataset_mix](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_2_training_dataset_mix/README.md)
- [02_apertus_tokenizer_spec](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_apertus_tokenizer_spec/README.md)
- [02_1_tokenizer_experiments](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/README.md)
- [02_2_tokenizer_implementation](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_2_tokenizer_implementation/README.md)
- [03_apertus_extension_and_embedding_adaptation](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/README.md)

## Repo Policy

- `artifacts/` and local virtualenvs are ignored by git
- legacy baseline material is preserved under `legacy/`
- unresolved research decisions should be frozen only after explicit user confirmation

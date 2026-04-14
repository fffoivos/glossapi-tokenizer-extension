# GlossAPI Tokenizer Extension

Clean workspace for extending `swiss-ai/Apertus-8B-2509` for Greek.

The active method is:
- discover Greek morphology-respecting units through true `BPE` training
- replicate Apertus tokenization behavior as exactly as possible
- filter HPLT and normalize it into the existing canonical GlossAPI source-parquet schema
- consume the same CPT-ready dataset for both continued pretraining and tokenizer experiments
- extend Apertus through `model.vocab` and `model.merges`, not `add_tokens(...)`
- preserve dedup functionality while improving dedup efficiency and scalability

The old whole-word `add_tokens(...)` sweep is retained only as a legacy baseline and has been moved out of the active planning path.

## Canonical Code Root

This repo is now the canonical source for the active tokenizer pipeline code.

That includes:
- `glossapi_corpus_cli/`
- stage orchestration scripts under `subprojects/`
- upload handoff scripts under `ops/upload/`
- smoke and efficiency harnesses under `ops/`
- the active test suite under `tests/`

The workspace copy under `/home/foivos/data/glossapi_work/` is no longer the development source of truth for pipeline code. It remains a data/workspace root and deployment target only.

## Test Matrix

The active repo-local verification matrix includes:
- script proof tests
- stage-to-stage contract tests
- resumability regressions
- tiny real-document end-to-end smoke runs
- efficiency smokes for streaming mix build and near-candidate execution

## Execution Shape

There are two parallel tracks:
- tokenizer critical path: repair and resume dedup without changing its decisions, freeze local manifests, freeze the literal Apertus tokenizer spec, export local BPE-training text, run discovery tokenizers, and implement the merge-rule extension
- dataset operational sidetrack: finish HPLT filtering and integration work, refresh published dedup metadata, and publish the updated upstream dataset from a separate cheap uploader instance using the official large-folder HF upload path

The tokenizer critical path does not need to wait for the HF upload once the filtered HPLT slice exists locally on `home`.

The active dedup recovery and scale plans are:
- [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
- [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)

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

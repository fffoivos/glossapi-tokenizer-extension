# Current Status

## Active Phase

HPLT filtering and upload-schema freeze.

## What Is Settled

- the shipping method is merge-rule tokenizer extension, not whole-word `add_tokens(...)`
- Apertus compatibility is a hard constraint
- machine-translated HPLT content is excluded from the final training dataset
- HPLT is being prepared for the upstream HF corpus dataset, not as a separate tokenizer-only corpus
- the downstream CPT/tokenizer builder is expected to stay lightweight after HF download
- the workspace has now been split into smaller subprojects

## What Is Not Done Yet

- no frozen HPLT filtering spec
- no frozen HPLT upload-schema mapping
- no uploaded HPLT parquet integrated into the canonical HF dataset
- no frozen training/eval manifests derived from the refreshed upstream dataset
- no true Greek `BPE` discovery tokenizer
- no implemented merge-rule extension
- no model adaptation plan beyond high-level constraints

## Current Trust Boundary

Active planning and execution should use:
- [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- the relevant subproject folders

Legacy baseline and exploratory material has been moved under:
- [legacy/](/home/foivos/Projects/glossapi-tokenizer-extension/legacy/README.md)

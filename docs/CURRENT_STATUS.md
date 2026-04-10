# Current Status

## Active Phase

Corpus design and workspace cleanup.

## What Is Settled

- the shipping method is merge-rule tokenizer extension, not whole-word `add_tokens(...)`
- Apertus compatibility is a hard constraint
- a diverse deduplicated corpus is a hard constraint
- the workspace has now been split into smaller subprojects

## What Is Not Done Yet

- no frozen HPLT filtering spec
- no frozen dedup pipeline
- no frozen training/eval manifests
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


# 01.1 Corpus Dedup

## Status

**DONE for the C3 shipping path** (as of 2026-05-11).

Dedup metadata bundle frozen; C3 trained on the deduped corpus.

This subproject does not need further work to ship C3. See
[../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md) for the
active scope.


## Scope

Document and verify the dedup contract inherited from the upstream GlossAPI corpus dataset pipeline.

## Already Decided

- the training corpus must still be deduplicated before proper `BPE` training
- for this project, HPLT prep should not reimplement a separate local dedup builder stage
- the upstream corpus dataset pipeline is the owner of cleaning and dedup for the uploaded source parquets
- downstream tokenizer/CPT builders should consume that prepared result lightly
- HPLT `cluster_size` remains a useful audit hint, not a guarantee
- published `dedup_metadata` refresh is a later operational step and does not block tokenizer progress

## Role In This Project

This subproject now has two roles:
- document the dedup contract inherited by downstream builder consumers
- own the scaling repair for the exact dedup implementation used to refresh `dedup_metadata`

The active repair plans are:
- [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
- [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)

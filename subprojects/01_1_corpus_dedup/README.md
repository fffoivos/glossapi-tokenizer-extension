# 01.1 Corpus Dedup

## Scope

Document and verify the dedup contract inherited from the upstream GlossAPI corpus dataset pipeline.

## Already Decided

- the training corpus must still be deduplicated before proper `BPE` training
- for this project, HPLT prep should not reimplement a separate local dedup builder stage
- the upstream corpus dataset pipeline is the owner of cleaning and dedup for the uploaded source parquets
- downstream tokenizer/CPT builders should consume that prepared result lightly
- HPLT `cluster_size` remains a useful audit hint, not a guarantee

## Role In This Project

This subproject is now mostly a verification boundary:
- confirm what dedup guarantees the upstream dataset pipeline provides
- confirm what builder-time dedup, if any, still remains lightweight downstream
- avoid drifting into a second independent dedup workflow here

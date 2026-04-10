# 01.1 Corpus Dedup

## Scope

Deduplicate HPLT and GlossAPI candidate data before tokenizer training.

## Already Decided

- dedup is mandatory before any proper `BPE` training
- same-source overlap reduction comes before text-level dedup
- canonical URL dedup should be included
- exact-text and near-duplicate dedup should both be included
- HPLT `cluster_size` is only a hint, not a guarantee

## Required Order

1. same-source overlap reduction
2. canonical URL dedup
3. exact-text dedup
4. near-duplicate dedup
5. freeze training manifest


# 01 HPLT Filtering

## Scope

Build a clean, source-aware, quality-aware filtered view of Greek HPLT that is suitable to feed downstream corpus assembly.

## Already Decided

- do not sample contiguous prefixes from sorted HPLT shards
- use HPLT metadata for stratification
- treat quality bins `8-10` as the current default keep range
- use `web-register` information for content diversity analysis
- do not trust released HPLT dedup as sufficient by itself

## Relevant Existing Assets

- metadata probe script:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/inspect_hplt_greek_metadata.py`
- lightweight manifest builder:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/build_hplt_greek_manifest.py`
- quality-bin analysis:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/analyze_hplt_quality_bins.py`
- register mapping:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/hplt_web_register.py`

## Deliverable

A frozen filtered HPLT manifest with:
- shard
- document id
- URL or host
- content type
- quality bin
- register labels
- character count

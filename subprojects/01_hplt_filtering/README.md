# 01 HPLT Filtering

## Scope

Build a clean, source-aware, quality-aware filtered view of Greek HPLT that is suitable for upload into the upstream canonical GlossAPI source-parquet dataset.

## Already Decided

- do not sample contiguous prefixes from sorted HPLT shards
- use HPLT metadata for stratification
- treat quality bins `8-10` as the current default keep range
- exclude `Machine translated or generated` content from final training data
- use `web-register` information for content diversity analysis
- HPLT-specific provenance should be preserved in `source_metadata_json`
- HPLT should be normalized into the existing canonical source-parquet schema before upload

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

Upload-ready HPLT parquet file(s) in the canonical schema used by:
- `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
- `data/*.parquet`

The shipped HPLT slice must:
- exclude `Machine translated or generated`
- preserve the canonical top-level columns
- map HPLT-specific provenance into `source_metadata_json`

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
- the tokenizer path may proceed from the local canonical source-parquet tree without waiting for the HF upload to finish

## Relevant Existing Assets

- metadata probe script:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/inspect_hplt_greek_metadata.py`
- lightweight manifest builder:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/build_hplt_greek_manifest.py`
- quality-bin analysis:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/analyze_hplt_quality_bins.py`
- register mapping:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/hplt_web_register.py`
- upload builder:
  - `/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/build_hplt_hf_slice.py`

## Deliverables

Upload-ready HPLT parquet file(s) in the canonical schema used by:
- `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
- `data/*.parquet`

The shipped HPLT slice must:
- exclude `Machine translated or generated`
- run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
- drop rows with `greek_badness_score > 60`
- preserve the canonical top-level columns
- map HPLT-specific provenance into `source_metadata_json`

## Operational Note

This subproject is the first step of the dataset sidetrack:
1. build and upload the filtered HPLT slice
2. rerun the full local prepared-source dataset with HPLT included using the existing dataset scripts
3. hand the resulting local prepared dataset to the tokenizer track without waiting for HF upload completion

# 01.2 Training Dataset Mix

## Scope

Freeze the training and eval corpus views that will feed tokenizer discovery, using the same CPT-ready dataset that will be used for Apertus continued pretraining.

## Already Decided

- compare two tokenizer-training views:
  - `GlossAPI-only`
  - `GlossAPI + HPLT`
- HPLT should initially be matched roughly to nanochat scale
- held-out evaluation documents must not overlap with training documents
- the `200`-document human review should come from the filtered upload-candidate pool or the refreshed CPT-ready dataset
- downstream mixing should happen after HPLT has already been normalized into the canonical source-parquet schema
- the downstream builder should stay lightweight after HF download

## Deliverables

- `nanochat_train` manifest
- `hplt_matched_sample` manifest
- mixed training manifest
- held-out manifests:
  - `nanochat_eval`
  - `hplt_eval`
  - `modern_greek_eval`

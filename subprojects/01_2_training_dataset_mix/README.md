# 01.2 Training Dataset Mix

## Scope

Freeze the training and eval corpus views that will feed tokenizer discovery.

## Already Decided

- compare two tokenizer-training views:
  - `GlossAPI-only`
  - `GlossAPI + HPLT`
- HPLT should initially be matched roughly to nanochat scale
- held-out evaluation documents must not overlap with training documents
- the `200`-document human review should come from the frozen deduplicated training pool

## Deliverables

- `nanochat_train` manifest
- `hplt_matched_sample` manifest
- mixed training manifest
- held-out manifests:
  - `nanochat_eval`
  - `hplt_eval`
  - `modern_greek_eval`


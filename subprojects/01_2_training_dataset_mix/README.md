# 01.2 Training Dataset Mix

## Scope

Freeze the training and eval corpus views that will feed tokenizer discovery, using the same CPT-ready dataset that will be used for Apertus continued pretraining.

## Already Decided

- compare two tokenizer-training views:
  - `GlossAPI-only`
  - `GlossAPI + HPLT`
- the mixed `GlossAPI + HPLT` tokenizer-training view should use a `70/30` split by training-token mass
- held-out evaluation documents must not overlap with training documents
- the `200`-document human review should come from the filtered upload-candidate pool or the refreshed CPT-ready dataset
- downstream mixing should happen after HPLT has already been normalized into the canonical source-parquet schema
- downstream builder work should stay lightweight after HF download
- local mix freezing can proceed from the local prepared dataset without waiting for the HF upload to complete
- `openarchives.gr` rows with `needs_ocr == true` must remain excluded in the dataset fed into tokenizer work

## Deliverables

- `nanochat_train` manifest
- `hplt_matched_sample` manifest
- mixed training manifest
- held-out manifests:
  - `nanochat_eval`
  - `hplt_eval`
  - `modern_greek_eval`

## Operational Input

The intended input here is the refreshed local prepared-source dataset that already includes the filtered HPLT slice, not the exploratory review samples.

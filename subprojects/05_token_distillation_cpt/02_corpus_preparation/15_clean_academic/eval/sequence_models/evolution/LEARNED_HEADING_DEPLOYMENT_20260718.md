# Learned three-role heading deployment

## Outcome

The existing grouped-OOF heading artifacts are deployable without retraining.
The five fold pickles each contain their fitted character and word TF-IDF
transforms, numeric scaler, any-heading classifier, and conditional three-role
classifier.  Fold-specific replay on the original held-out rows reproduced all
24,616 by 4 stored float32 probabilities exactly:

- maximum absolute difference: `0.0`;
- differing float32 cells: `0`;
- in-memory probability array SHA-256:
  `4819357172ac5f76eb3dc1cc781d3be89d561f6af265d2ce1bb1369641c926f9`;
- pinned runtime: scikit-learn `1.9.0`.

The historical job serialized `HeadingBundle` and `HeadingTransform` under the
module name `__main__`.  `bibliography_heading_deployment.py` handles this with
a narrow two-class compatibility mapping.  It does not treat arbitrary
`__main__` globals as trusted classes.

## Immutable Clariden sources

Model root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_role_pipeline_20260715/heading_oof_6b28e4c_r1`

- receipt SHA-256:
  `bc26867ba770f3724b3305e767630ed9645dea086d80d52f6f3fad10f75261e8`;
- stored OOF NPY SHA-256:
  `6b7dc8e9f2bf8986e508792745a961fd71c49ac0c4d039b0c622dfff0426b4ba`;
- fold models: exactly `fold0.pkl` through `fold4.pkl`.

Heading candidate table:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_role_pipeline_20260715/heading_table_6b28e4c_r1`

- receipt SHA-256:
  `ebd4792e087674e3854f470f8aedce42594598f558c62132a249db7e0af7d886`;
- manifest SHA-256:
  `5db5f4e1c35f2099195b81e5c5ef23396feb96332c758a0a60dfc719f03ab03a`.

Training base table:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/table`

- manifest SHA-256:
  `be8e146647470384fca9edcb601369f22fa4964f826977f8a9af5ae09571f498`;
- 1,118 documents and 1,100 work groups.

The evolution validation table has 274 documents and 274 work groups.  Direct
comparison found zero overlapping document IDs and zero overlapping work IDs
with the heading training table.

## Two disjoint inference modes

`receipt_bound_grouped_oof_replay` exists only as a deployment verification
gate.  It accepts the exact stored heading candidate row inventory and uses the
model corresponding to each row's held-out fold.  It cannot score arbitrary
rows.

`unseen_ensemble_mean` is the only operational mode for evolution validation
and the sealed test.  Before scoring, it compares each input work ID against a
candidate-owned copy of the 1,100 training work IDs and requires zero overlap.
It then applies all five fold models and takes their arithmetic mean.

Both modes use the original high-recall candidate predicate, D1 threshold
`0.25`, physical radius `30`, and 43 numeric heading features.  Operational
roles sweep the predeclared any-heading thresholds `0.30`, `0.40`, `0.50`,
`0.60`, and `0.70` (with `0.50` as the control) across header windows 1 through
4.  The exact threshold is part of the candidate identity, backend receipt,
and sealed replay.  Accepted candidates take the argmax of `BIB_HEADER`,
`BIB_SUBHEADER`, and `NON_BIB_HEADER`; non-candidates and rows below threshold
remain `NONE`.

## Candidate ownership

Every learned G2 candidate materializes `backend/heading_deployment/` before
using the model.  The directory contains:

- all five copied fold models;
- `feature_contract.json`, including feature names, candidate policy,
  assignment policy, and code hashes;
- `training_work_ids.json`, used for the zero-overlap gate;
- `receipt.json`, including the byte-identical OOF replay proof and both mode
  contracts.

The normal evolution finalizer recursively inventories these files, so they
are owned by the finalized candidate receipt.  Sealed inference refuses a
learned G2 candidate unless every consumed package file is in that verified
candidate inventory.  The candidate derivation records their byte hashes.

## Launch bindings

The learned G2 template now requires these runner inputs, each represented by
an exact `G2_INPUT_RECEIPTS` row:

- `HEADING_OOF_DIR`;
- `HEADING_TRAINING_TABLE_DIR`;
- `HEADING_TRAINING_BASE_TABLE_DIR`;
- `VALIDATION_DOCUMENTS`;
- `VALIDATION_LINE_PROBABILITY`.

`VALIDATION_DOCUMENTS` must not point at the mixed-split STRUCT-2K source.
First run the prediction-document materializer, which aligns the validation
rows to the validation table and writes only identity, work/source, physical
coordinates, and text (no labels and no train/test documents):

```bash
python -m sequence_models.bibliography_heading_deployment \
  materialize-prediction-documents \
  --source /capstor/REPLACE/struct2k.LLM_silver.jsonl \
  --table-dir /capstor/REPLACE/validation_r3/validation_table \
  --split validation \
  --output-dir /capstor/REPLACE/heading_validation_prediction_documents
```

Bind the resulting `documents.jsonl` and its SHA-256 in `G2_INPUT_RECEIPTS` as
a label-free validation artifact.  The mixed-split source is never a candidate
runner input.

This branch does not launch or train anything.  A learned G2 candidate should
only be instantiated after this commit is rebased/cherry-picked onto the
audited evolution/sealed-inference bridge and its exact test receipt is bound
into the candidate specification.

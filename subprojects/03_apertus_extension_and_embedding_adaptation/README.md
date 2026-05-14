# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Sub-subprojects

- [03_1_greek_embedding_diagnostic/](03_1_greek_embedding_diagnostic/README.md)
  — pre-extension diagnostic characterising how Apertus-8B-2509 represents
  Greek on its E + U matrices (centroid geometry, MP-edge spectrum, hull
  occupancy / infiltrators, morphological clustering, binary Greek-vs-¬Greek
  classifier, cross-language semantic-cluster baseline). Diagnostic only —
  no new-token init, no CPT.
  - canonical plan: [../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md](../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md)
  - results report: [03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md](03_1_greek_embedding_diagnostic/artifacts/results/report_v2.md)
  - session review: [03_1_greek_embedding_diagnostic/artifacts/results/REVIEW.md](03_1_greek_embedding_diagnostic/artifacts/results/REVIEW.md)

## Already Decided

- this comes after tokenizer and corpus work, not before
- embeddings and `lm_head` both matter because `tie_word_embeddings = false`
- only the new rows need explicit initialization
- the intended schedule is:
  - frozen-base warmup
  - then full continued pretraining

## Still Open

- exact initialization method
- exact warmup schedule
- exact multilingual replay ratio
- exact acceptance criteria for model-side success


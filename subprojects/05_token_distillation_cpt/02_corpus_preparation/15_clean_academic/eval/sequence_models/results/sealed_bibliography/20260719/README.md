# Sealed bibliography receipts — 2026-07-19

This directory is the repository-local metadata archive for the completed
annotation workflow. It intentionally contains no sealed document text, line
key, or line labels.

- `original-150-consensus.blocked.receipt.json` preserves the original
  150-document merge and its failed 0.98 raw A/B agreement gate.
- `consensus-silver.materialization.receipt.json` binds the user-directed
  143-document agreement-only cohort and all large Clariden output hashes.
- `consensus-silver.audit.receipt.json` is the independent reconstruction and
  hash audit.
- `consensus-silver.exclusions.json` records the seven explicit document
  exclusions without document text.
- `consensus-silver.FROZEN.receipt.json` is the terminal seal from Clariden job
  `2799088` and code commit `b9f27cd`.

The large mode-`0440` sealed artifacts remain only on Clariden at:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/sealed_tests/bibliography_150_20260718/48_consensus_silver/run-4256753`

The successor is a post-hoc consensus-silver evaluation cohort, not a rewrite
of the failed original prediction-blind 150-document protocol. See
`../../../CONSENSUS_SILVER_20260719.md` for the full interpretation.

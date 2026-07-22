# Bibliography training extraction review — 2026-07-14

The prediction- and label-blind Clariden screen in job `2755131` examined all
1,118 training documents using the GlossAPI Rust Greek-noise score plus text
extraction checks.  It produced 24 review candidates.  Nothing was excluded
automatically.

For every candidate, five evenly distributed text samples and several lines
containing the triggering artifact were inspected.  Two ambiguous
character-spacing cases were inspected at twelve evenly distributed points.
The review considered only whether the extracted text retained usable words
and lines.  Silver labels and classifier predictions were not used.

## Decision

- **Keep:** 19 candidates.  These contain localized formula, bullet, table,
  polytonic-character, or table-of-contents artifacts, while their prose and
  reference-like text remain readable.
- **Exclude from classifier fitting and train-OOF scoring:** 5 candidates.
  These are unusable symbol-font output, literal byte escapes, pervasive GLYPH
  placeholders, or pervasive character spacing.

The five excluded document IDs are:

1. `ce097160089c20a95f138155fb2942fbd90abfe0ad4f2f894a33eb3bd277abf4`
2. `cab64a69a86eee21b403088272252c96daefc7c85641fa6353f94709e5d52442`
3. `9e0f741f3702a5ad6fe6899b51d49d688c4b0df621e44ca7ec26ba2862d9ef8f`
4. `1de32df6634a009c512f2a5eefdf231b2c400cb975bdb37e1739a2d58556b5e2`
5. `4887aad0f891a0d93167d6ae7018cbd026f2ecfc0867d2034e2b58dfdfa7f217`

The complete 24-document decision record is
`bibliography_training_quality_decisions_20260714.json`.  Its source packet is
the immutable Clariden artifact:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/experiments/bib_entry_oof_20260713t204926z/train_quality_r1
```

The packet SHA-256 is
`e05aadda461019f2fff9fe25dfff79c98c1e43de9809a98090670d11ca8de7d7`.
The exclusions are diagnostic classifier-training exclusions only.  They do
not silently redefine the frozen validation set or make a dataset-publication
decision.

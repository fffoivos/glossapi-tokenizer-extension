# Bibliography line-feature explorer

This is a deliberately unweighted inspection tool. It is separate from both
the bibliography-v2 weighted scorer and the document-level block decoder.

- It selects 20 source-balanced, `train` / `full_document` documents by a
  deterministic SHA-256 rank.
- It evaluates every nonblank line with the 38-field
  `BibliographyFeatures` extractor.
- `token_count` is displayed as line context rather than treated as a detected
  bibliography event. The remaining 37 nonzero feature counts each contribute
  exactly one point, regardless of magnitude.
- The page retains all evaluated lines so disabling a feature reranks the whole
  inventory before rendering the top 100; it does not merely filter the
  already-rendered list.
- It never includes silver labels, model predictions, weighted scores, or
  block-decoder decisions.

Build on a worker with the current STRUCT-2K JSONL:

```bash
python3 -m sequence_models.bibliography_feature_explorer \
  --input /absolute/struct2k.LLM_silver.jsonl \
  --output /new/output/index.html \
  --receipt /new/output/build.receipt.json
```

Serve the resulting directory locally with any static HTTP server. The page is
self-contained and requires no backend or external assets.

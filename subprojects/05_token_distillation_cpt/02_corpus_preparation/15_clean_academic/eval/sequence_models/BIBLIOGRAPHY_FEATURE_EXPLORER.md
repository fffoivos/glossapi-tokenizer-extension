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

The tracked Clariden wrapper is
`clariden/build_bibliography_feature_explorer.sbatch`. It binds a clean commit,
hides accelerators, refuses to replace an existing output, and verifies the
expected 7/7/6 source split before publishing the site path.

Serve the resulting directory locally with any static HTTP server. The page is
self-contained and requires no backend or external assets.

## Completed build

Clariden job `2749470` completed on 2026-07-13 from exact worker commit
`e6052c9243e2ed7a18e50d355ddbfb91b20629f5`:

- 20 documents: 7 Greek PhD, 7 Kallipos, and 6 OpenArchives;
- 14,815 nonblank lines;
- 37 scored detector fields plus unscored token-count context;
- 8,434,904-byte self-contained HTML;
- HTML SHA-256
  `fd6e7ba1c3fbcb24b02d48047a0b5962dce0485e2d6cbab99c6efb91b33b4bb6`.

The immutable remote site is at:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/bibliography_feature_explorers/bib_features_train20_e6052c9/index.html
```

The exact receipt is archived at
`results/bibliography_feature_explorer/build.receipt.json`. The downloaded
local presentation is intentionally ignored by git at
`outputs/bibliography-feature-explorer/index.html`.

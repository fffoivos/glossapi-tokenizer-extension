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
- The review-only span extractor emits exact `[start:end]` Unicode-character
  offsets into the NFKC-normalized text displayed by the page. Every raw feature
  count must equal the number of emitted spans for that feature or the build
  fails. These spans are not used by the weighted scorer or block decoder.
- The target line overlays each active match with its feature colour and prints
  the same offsets in the feature badge. Overlapping features are displayed as
  stacked colour bands.
- Besides the original distinct-feature count, the page computes active match
  occurrences per 100 characters, distinct feature points per 100 characters,
  and union matched-character coverage. A menu can rerank the entire inventory
  by any of these diagnostics.
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

## Current span-aware build

Clariden job `2749627` completed on 2026-07-13 from exact worker commit
`aa40bb0ad50986fa6a036043e6a713c8f748ebc0`. It uses the same 20 documents
and 14,815 lines as the initial build, now with count-parity-checked character
offsets and length-normalized ranking options. The HTML SHA-256 is
`64a42a52629f18fb3dcaea326275a12cf7c52a830c80d84df6776443990bc18b`.

The immutable remote site is at:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/bibliography_feature_explorers/bib_features_train20_spans_aa40bb0/index.html
```

The exact receipt is archived at
`results/bibliography_feature_explorer/span_review_build.receipt.json`. The
current downloaded presentation remains at
`outputs/bibliography-feature-explorer/index.html`.

The original rank-3 false positive illustrates why both diagnostics are
retained: 16 distinct features and 240 overlapping matches fire, but they occur
inside a 3,015-character line. Only 26.0% of its characters are covered and its
match density is 8.0 per 100 characters, versus roughly 25–31 per 100 for the
neighbouring full bibliography entries.

## Initial build

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

The initial receipt is archived at
`results/bibliography_feature_explorer/build.receipt.json`. The downloaded
presentation is intentionally ignored by git.

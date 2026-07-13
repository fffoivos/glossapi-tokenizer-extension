# Bibliography line-feature explorer

This is a deliberately unweighted inspection tool. It is separate from both
the bibliography-v2 weighted scorer and the document-level block decoder.

- It selects 20 source-balanced, `train` / `full_document` documents by a
  deterministic SHA-256 rank.
- It evaluates every nonblank line with the current
  `BibliographyFeatures` extractor.
- `token_count` is displayed as line context rather than treated as a detected
  bibliography event. The remaining 35 nonzero feature counts each contribute
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

For feature/UI iterations, the exact label-blind sample embedded in an existing
site can be rebuilt locally without rereading or transferring STRUCT-2K:

```bash
python3 -m sequence_models.bibliography_feature_explorer \
  --input-site /absolute/previous/index.html \
  --output /new/output/index.html \
  --receipt /new/output/build.receipt.json
```

The tracked Clariden wrapper is
`clariden/build_bibliography_feature_explorer.sbatch`. It binds a clean commit,
hides accelerators, refuses to replace an existing output, and verifies the
expected 7/7/6 source split before publishing the site path.

Serve the resulting directory locally with any static HTTP server. The page is
self-contained and requires no backend or external assets.

## Current corrected local build

The current presentation was rebuilt locally on 2026-07-13 from the exact
20-document, 14,815-line label-blind packet embedded in the prior hover build.
No Clariden compute job was required. It has 35 scored features after these
corrections:

- `inverted_author_count` captures every inverted author and all adjacent
  initials, including `Lewis, M.A.`;
- `initial_count` covers one- or two-letter forms such as `I.` and `Ph.`, while
  dotted words require at least three letters, so their spans do not overlap;
- the redundant `initial_sequence_count` and `author_joiner_count` fields were
  removed;
- `numbered_entry_count` marks a line when its first non-decoration character
  is numeric, using a bounded linear scan.

The current HTML SHA-256 is
`84e7266a577a5ec27b66378fa35726854220447e91a5e4998f320bfd32e63bdd`.
The exact local build receipt is archived at
`results/bibliography_feature_explorer/corrected_local_build.receipt.json`.
Its output path records the staging location before the SHA-identical file was
promoted to `outputs/bibliography-feature-explorer/index.html`.

The earlier bibliography-v2 metric reports predate these feature-definition
changes and are historical; the scorer must be re-evaluated before those
metrics are treated as current.

## Current hover-spotlight build

Clariden job `2749714` completed on 2026-07-13 from exact worker commit
`a559e6139e6f88fec5e6612fb1e3a426c4051228`. Hovering either a feature badge
or its sidebar label now redraws all visible target lines with only that
feature's boxes; leaving restores every active feature without changing scores
or ordering. The HTML SHA-256 is
`72afc7208532d5e242dc4bc68dd51b7f721054e537cfe09a8fbd483b0f9fe424`.

The immutable remote site is at:

```text
/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/bibliography_feature_explorers/bib_features_train20_hover_a559e61/index.html
```

The exact receipt is archived at
`results/bibliography_feature_explorer/hover_build.receipt.json`.

## Initial span-aware build

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

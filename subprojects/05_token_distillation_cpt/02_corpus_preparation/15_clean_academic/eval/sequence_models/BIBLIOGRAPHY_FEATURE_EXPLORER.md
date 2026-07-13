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
  inventory. It displays every line in descending order through an infinite
  feed, appending 100 lines at a time rather than truncating the result.
- The review-only span extractor emits exact `[start:end]` Unicode-character
  offsets into the NFKC-normalized text displayed by the page. Every resolved
  feature count must equal the number of emitted spans for that feature or the build
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

## Current multilingual-format v5 build

The current presentation keeps the complete 14,815-line inventory in every
ranking view. It renders the first 100 lines immediately and appends subsequent
100-line batches as the reviewer scrolls, preserving descending order under the
selected metric. Changing a detector or ranking metric starts a newly ranked
feed from rank 1.

The author/name patterns now derive their case-sensitive character classes from
Unicode categories across Latin Extended, Greek and Greek Extended, and
Cyrillic blocks. This covers Romance and Latin-script Slavic diacritics,
Cyrillic Slavic names, and monotonic/polytonic Greek, including remaining
combining marks after NFKC normalization. Citation surname particles and the
specific PDF-extraction splits seen in the review corpus are supported without
making broad proper-name words consume ordinary whitespace.

Verified corpus corrections include:

- document `47416e0142ff...`, line 1142: `Č iarnien ė , R.` and
  `Vienažindien ė , M.` are two inverted authors;
- document `cf9352e1db72...`, line 2394: `Enzymology.`, `(eds)`, undotted
  `pp`, and all five undotted surname/initial pairs have owners;
- line 2411: the complete `34:27-39` volume/page coordinate is owned rather
  than only its numeric suffix;
- line 2425: `5 th edn.`, `(ed)`, undotted `pp`, and `622-642` have distinct
  owners.

The unrequested `author_year_count` composite remains absent. Years and author
shapes are separate evidence. `article_page_range_count`, displayed as
“Volume/article page range”, owns both `44:1-44:14` and `34:27-39` forms.

This remains a 35-feature presentation. The current HTML SHA-256 is
`9d052e449b96bfc1bc95649487ba15a2f68e1ab20a4636aab6b236f5cc954c2d`.
The exact build receipt is archived at
`results/bibliography_feature_explorer/multilingual_v5_build.receipt.json`. Its
output path records the staging location before promotion to
`outputs/bibliography-feature-explorer/index.html`.

## Previous infinite-scroll v4 build

The v4 build introduced the all-line, descending infinite feed, removed the
author-year composite, and added the repeated article/page form
`44:1-44:14`. Its HTML SHA-256 is
`3dfdb8d94a1de1f2980ca9a938dece041add91f930ebad49878f357f68964279`.
Its receipt remains at
`results/bibliography_feature_explorer/infinite_v4_build.receipt.json`.

## Previous ownership-resolved v3 build

This presentation was rebuilt locally on 2026-07-13 from the exact
20-document, 14,815-line label-blind packet embedded in the prior hover build.
No Clariden compute job was required. The 35 scored features now use explicit
ownership: specific lexical/numeric detectors claim a span before a broad
fallback detector may count it. Structural composites may still contain their
atomic evidence; for example an inverted-author span contains its initials.

The principal corrections are:

- dotted words exclude editor/translator, edition, volume, page, publisher,
  place, title-quote, identifier, and dotted-sequence spans;
- proper-name words cannot end immediately before a dot and exclude specific
  place, publisher, date, title, and structural-term spans;
- DOI owns DOI URLs; composite dates and publication coordinates own their
  years; place–publisher shapes own their shared lexical span;
- page ranges reject decimals, thousands-separated values, full dates, and
  year ranges; decimal-leading lines are not numbered entries;
- `ed.` inside `2nd ed.` belongs to edition, and bare `trans` inside a
  hyphenated word is not a translator abbreviation;
- direct-order and inverted-order author hypotheses are evaluated across the
  line and only the stronger eligible orientation is retained. An eligible
  hypothesis must begin after a legal bullet/list prefix.

On the same packet, the audited accidental-overlap policy fell from 6,924
events across 46 feature pairs to zero. Proper-name matches immediately before
a dot fell from 1,814 to zero. For document
`a32563c98868101bfde4b1942897ca3d6c867b1ca882b116070772b0902c6235`, line
2421 now has seven direct-order authors and zero inverted-order authors.

The v3 HTML SHA-256 is
`a00a8ac935adcd785f21db6669190b3aa174b824d4f83c6a33e0b4dfcf62cd4c`.
The exact local build receipt is archived at
`results/bibliography_feature_explorer/ownership_v3_build.receipt.json`, and
the overlap evidence is archived at
`results/bibliography_feature_explorer/overlap_audit_v3.json`.
Its output path records the staging location before the SHA-identical file was
promoted to `outputs/bibliography-feature-explorer/index.html`.

The earlier bibliography-v2 metric reports predate these feature-definition
changes and are historical; the scorer must be re-evaluated before those
metrics are treated as current.

## Previous corrected v2 local build

The preceding 35-feature local build had HTML SHA-256
`84e7266a577a5ec27b66378fa35726854220447e91a5e4998f320bfd32e63bdd`.
Its receipt remains at
`results/bibliography_feature_explorer/corrected_local_build.receipt.json`.

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

# Bibliography FILLER feature research — 2026-07-16

## Conclusion

`FILLER` is not a broad semantic text class in the reviewed corpus. It is
mostly extraction/layout debris that belongs inside an already-supported
bibliography region. The current 177-feature connector table is therefore
over-complete for the conditional `FILLER` versus `CONTINUATION` decision.

The evidence supports treating `FILLER` like the `O-REF` state in a sequence
model: deterministic artifact recognition for obvious cases, followed by a
contextual gap/block decision. It should not be expected to work as a global
standalone line classifier.

## What the labelled FILLER class contains

The trusted train overlay contains 278 `FILLER` and 535 `CONTINUATION` lines.
All 813 lines were rejoined to their blinded review text.

| observed FILLER form | lines | share |
|---|---:|---:|
| tiny fragment of at most three characters | 141 | 50.7% |
| repeated rule or separator | 53 | 19.1% |
| Markdown table row | 46 | 16.5% |
| image/HTML extraction marker | 13 | 4.7% |
| all other forms | 25 | 9.0% |

The residual 25 lines are mostly repeated page headers/footers, acronym-list
material, OCR fragments, and a few long extraction artifacts. They are not a
single lexical class.

The class is also document-concentrated:

- one malformed OpenArchives document contributes 132/278 fillers (47.5%);
- the five largest documents contribute 181/278 (65.1%);
- 130/278 fillers (46.8%) repeat elsewhere in the same document after simple
  whitespace/case normalization, versus 35/535 continuations (6.5%);
- 116/278 fillers repeat at least five times, versus 23/535 continuations.

This means line-weighted training can learn the extraction pathology of one
document instead of a general filler concept. Document-balanced fitting and
per-document evaluation are required.

## Context is part of the definition

Within the reviewed sequences:

- 108/278 fillers sit directly between two `ENTRY_ANCHOR` labels;
- 232/278 (83.5%) are within two physical-line indices of an entry or
  continuation;
- 264/278 (95.0%) are within five;
- all 278 are within ten;
- the median contiguous filler run has one line; the longest has ten.

These observations are consistent with filler being an interior bridge state,
not a self-supporting bibliography signal.

## Empirical feature audit

Clariden job `2773901` ran out-of-fold permutation and univariate audits over
the frozen connector models. It did not open validation. The full receipt is
under:

`results/bibliography_role_pipeline/20260716/filler_feature_audit_9152d6a_r2/`

The fold-weighted OOF FILLER PR-AUC was 0.933635. This is a development result
and is likely made easier by the large population of trivial and repeated
artifacts.

| feature group | features | FILLER PR-AUC drop when permuted |
|---|---:|---:|
| current-line shape | 34 | 0.327554 |
| unmatched-character geometry | 7 | 0.038082 |
| deterministic bibliography counts | 35 | 0.015520 |
| adjacent-line shape pairs | 18 | 0.007780 |
| entry-probability neighbourhoods | 30 | 0.006955 |
| deterministic bibliography presence | 35 | 0.005702 |
| block-relative position | 2 | 0.004497 |
| joined-line entry gains | 8 | 0.004450 |
| heading probabilities | 4 | -0.000015 |
| nearest-entry-anchor fields | 4 | -0.001907 |

Permutation groups are correlated, so their drops are not additive. However,
the scale difference is decisive: line shape dominates, unmatched-character
coverage is useful, and most of the remaining 136 features add little in the
already-fitted trees. Heading probabilities and the nearest-anchor fields are
irrelevant or redundant for this conditional decision.

The strongest univariate signals are equally simple:

- token count: oriented ROC-AUC 0.9286; median FILLER 0 versus CONTINUATION 9;
- maximum token length: 0.9223;
- mean token length: 0.8915;
- letter fraction: 0.8419;
- unmatched-prefix fraction: 0.8405; median FILLER 1.0 versus CONTINUATION
  0.0059;
- unmatched-suffix fraction: 0.8151;
- joined-previous distinct-feature gain: 0.7947;
- proper-name presence: 0.7833;
- punctuation fraction: 0.7650.

## What the external work says

Körner's line-based CRF formulation defines `O-REF` as a line inside a
reference string that is not itself part of the reference. Its concrete
example is a page number inserted between two continuation lines. The paper
emphasizes layout, indentation, line spacing, start/end patterns, punctuation,
and sequence transitions rather than treating such lines as a lexical topic:
[Reference String Extraction Using Line-Based Conditional Random Fields](https://arxiv.org/abs/1705.08154).

ParsCit likewise uses a cascade of reference-section heuristics and CRF
parsing, with line length, ending punctuation, entry markers, and author-like
patterns used in reference-string segmentation:
[ParsCit](https://aclanthology.org/L08-1291/).

CERMINE separates reference-zone detection from reference-string grouping and
uses layout-aware line grouping before field parsing:
[CERMINE](https://link.springer.com/article/10.1007/s10032-015-0249-8).

GROBID's current reference segmenter is explicitly a sequence model. Its
documentation reports that a `BidLSTM_ChainCRF_FEATURES` model, including a
layout-feature channel, improves over the CRF baseline even on very long
reference sections:
[GROBID deep-learning models](https://grobid.readthedocs.io/en/latest/Deep-Learning-models/).

The literature therefore agrees with the corpus audit: the relevant signal is
line shape plus ordered context and layout, not a large bag of independent
bibliographic markers.

## Recommended compact feature contract

Keep or derive:

1. **Current-line artifact shape**
   - token and character length;
   - letter, digit, punctuation, symbol, and whitespace fractions;
   - rule, table-rule, page-number, image/HTML, glyph-only, and blank flags;
   - opening/closing punctuation and bracket/quote balance.
2. **Bibliographic coverage aggregates**
   - matched-character and unmatched-prefix/suffix fractions;
   - distinct bibliography-feature count;
   - compact flags for year/date, author/name, URL/DOI, and page/volume
     evidence.
3. **Continuation evidence**
   - previous/next join probability gain;
   - previous/next distinct-feature gain;
   - length ratio, indentation change, starts-lowercase, and whether the
     previous line ends open.
4. **Ordered block context**
   - entry/continuation probabilities for the immediate neighbours;
   - distance to supported evidence on each side;
   - relative position inside the proposed gap and whether evidence exists on
     both sides.
5. **New extraction features missing today**
   - normalized line repetition count within the document;
   - repeated header/footer position pattern;
   - page-boundary and page-number progression;
   - empty-table-cell ratio and isolated OCR-glyph pattern.

Remove from the conditional subtype head:

- heading probabilities;
- the redundant nearest-anchor feature group;
- most of the 70 separate presence/count bibliography fields;
- most duplicated multi-radius entry summaries.

The detailed bibliography features remain useful for the entry classifier;
they need not be copied wholesale into the filler subtype model.

## Experiments required before changing the pipeline

Compare, with identical document-grouped outer folds:

1. current 177-feature baseline;
2. line shape plus unmatched-character geometry;
3. the above plus compact join/context features;
4. the compact contract plus new repetition/page-layout features;
5. a small CRF/TCN sequence head over the compact line vectors.

Use document-balanced sample weights and report both micro and document-macro
metrics. Report a separate slice for the dominant malformed OpenArchives
document and a second result excluding it; do not allow it to determine model
selection. Keep the final validation set closed until this feature contract and
the proposed between-block connect/not-connect model are frozen.

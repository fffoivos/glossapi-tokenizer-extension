# Greek bibliography focus audit

## Selected document

- source: `openarchives`
- document ID: `b0b38a0ca63f3ac7ccfbab2326e75f3745dbd98b293d8bae498debb11ec62cc7`
- work ID: `10d1893ab125ea8530a821c8596d6d26d717427036bb8cfedb9e366bed418f44`
- physical lines: 2,108
- nonblank review lines: 1,197
- explicit section boundaries:
  - `## Βιβλιογραφία`: line 1,825
  - `## Ελληνόγλωσση`: line 1,827
  - 33 Greek-language entries: lines 1,829–1,893
  - `## Ξενόγλωσση`: line 1,895

This document was selected because it provides Greek body prose, an explicitly
headed Greek-language bibliography, and a foreign-language bibliography in the
same source document. The headings provide a useful section-boundary audit,
but they are not human-gold line annotations.

## Unweighted feature scores

The score is the current explorer score: one point for every enabled detector
feature that fires at least once on a line. It is not the weighted classifier
score and it does not include the block decoder.

| Region | Lines | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pre-bibliography body | 1,055 | 0 | 1 | 2 | 3 | 8 | 2.14 |
| Greek bibliography | 33 | 5 | 7 | 7 | 8 | 12 | 7.48 |
| Foreign bibliography | 106 | 1 | 6 | 7 | 8 | 12 | 7.23 |

The Greek references score slightly higher on average than the foreign
references. All 33 Greek references score at least 5; 31 of 33 score at least
7.

## Threshold contrast against the preceding body

This table treats the explicit Greek section only as a section-derived audit
target. “Precision” therefore means contrast against the 1,055 preceding body
lines in this document, not corpus precision.

| Score threshold | Greek entries recovered | Greek-section recall | Earlier body lines admitted | Section-contrast precision |
|---|---:|---:|---:|---:|
| ≥5 | 33/33 | 100.0% | 47 | 41.3% |
| ≥6 | 32/33 | 97.0% | 7 | 82.1% |
| ≥7 | 31/33 | 93.9% | 1 | 96.9% |
| ≥8 | 12/33 | 36.4% | 1 | 92.3% |
| ≥9 | 4/33 | 12.1% | 0 | 100.0% |

At the exploratory threshold of 7, the only earlier-body line admitted is line
359: a long prose paragraph ending in an inline citation. That is exactly the
kind of isolated false positive that a bibliography block/coherence decoder
should veto. The only Greek entry scoring below 6 is line 1,857, which lacks a
year and conventional publication coordinates.

## Strong examples

- line 1,885: score 12, rank 1 in the document
- lines 1,865, 1,873, and 1,887: score 9, ranks 7–9
- line 1,829: score 8, rank 24

The focused explorer preserves all 1,197 nonblank lines so feature switches and
alternative density rankings can be reviewed without losing the body
counterexamples. Build provenance is recorded in
`greek_bibliography_focus_v7_build.receipt.json`.


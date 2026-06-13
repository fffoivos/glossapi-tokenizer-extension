# Eval B at scale — β-gate confusion matrix (1985 sections, 596 held-out test)

annotations matched=1985 (missing 15) · hallucination-flagged quotes=79

## Held-out TEST, prevalence-weighted to the 94,282-section corpus

| | truth: bib | truth: NOT |
|---|---:|---:|
| gate: bib | 13940 | 4358 |
| gate: kept | 1305 | 8568 |

**precision = 0.762  (95% CI 0.709–0.820)**
**recall = 0.914  (95% CI 0.888–0.940)**
**F1 = 0.831 · accuracy = 0.799**

(unweighted on the balanced test sample: precision 0.807, recall 0.824)

## Per-stratum error rates (all units)

| stratum | n | error rate |
|---|---:|---:|
| bib_strong | 694 | FP 0.274 |
| bib_weak | 391 | FP 0.069 |
| kept_hasyear | 500 | FN 0.308 |
| kept_noyear | 400 | FN 0.043 |

## False-positive kinds (gate=bib, truly not): {'colophon_citation': 4, 'not_reference': 116, 'footnote_reference': 5, 'cv_publication_list': 90, 'web_sources': 2}

## False-negative language (gate=kept, truly bib): {'greek': 120, 'mixed_greek_foreign': 40, 'latin_foreign': 11}

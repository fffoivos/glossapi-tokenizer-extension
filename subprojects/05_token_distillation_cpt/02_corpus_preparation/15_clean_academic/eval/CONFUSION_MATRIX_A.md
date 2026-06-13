# Eval A — end-matter bibliography boundary (497 doc tails)

## (1) Found / not-found confusion (detector vs Opus)

| | truth: has bib | truth: no bib |
|---|---:|---:|
| detector: found | 205 (TP) | 3 (FP) |
| detector: not found | 126 (FN) | 123 (TN) |

**precision = 0.986  recall = 0.619  F1 = 0.761**  (n usable = 457)

## (2) Boundary localisation (docs both found)

- n = 227; median |error| = 0 lines; within 5 lines = 75%; within 20 = 76%; signed median = 0 (─ = detector starts too early)

## By source
| source | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| openarchives | 112 | 0 | 32 | 92 |
| greek_phd | 93 | 3 | 94 | 31 |
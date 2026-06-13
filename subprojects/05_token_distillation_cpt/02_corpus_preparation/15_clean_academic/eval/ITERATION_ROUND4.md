# β-gate Round 4 — document position + neighbour context (held-out test, weighted)

| model | precision | recall | F1 |
|---|---:|---:|---:|
| section-internal (round-3 lock) | 0.852 | 0.854 | 0.853 |
| **+ position/neighbour** | 0.939 | 0.861 | 0.898 |

**Paired bootstrap ΔF1(+pos − internal) = +0.046  [95% +0.017, +0.078]  → REAL break (CI excludes 0)**

## CV-FP before/after position

- CV-list FP (internal): 26  → (with position): 5

## position-model weights
{"log_entry": 0.87, "author": 1.02, "year_paren": 0.56, "place": 0.15, "header_cv": -0.38, "header_app": -0.08, "pos": 0.36, "dist_end": -0.36, "is_last": 0.49, "doc_maxpos": -0.2, "front_cluster": -0.58, "log_nsib": 0.13, "doc_has_end_bib": 0.0}
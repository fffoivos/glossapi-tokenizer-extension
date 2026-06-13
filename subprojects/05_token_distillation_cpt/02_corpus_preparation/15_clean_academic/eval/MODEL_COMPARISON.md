# β-gate: current vs tuned-rule vs logistic-regression (held-out test, weighted to corpus)

train=1389 · test=596

| predictor | precision | recall | F1 | accuracy |
|---|---:|---:|---:|---:|
| current gate | 0.762 | 0.914 | 0.831 | 0.799 |
| tuned rule (Y≥0.6, deny CV/colophon) | 0.757 | 0.927 | 0.833 | 0.799 |
| logistic regression (13 feats) | 0.882 | 0.889 | 0.886 | 0.876 |

## Logistic-regression weights (standardised — interpretable, Rust-portable as a dot-product)

- `year_line`: +1.34
- `url`: +1.14
- `author_init`: +0.82
- `year_paren`: +0.43
- `cv_marker`: -0.39
- `digit`: -0.34
- `comma_ano`: +0.30
- `stopword`: -0.19
- `locator`: +0.12
- `place_pub`: +0.11
- `midcap`: +0.11
- `period`: +0.07
- `colophon`: +0.00
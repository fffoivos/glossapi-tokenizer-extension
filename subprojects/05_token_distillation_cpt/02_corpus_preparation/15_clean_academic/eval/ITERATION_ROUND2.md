# β-gate iteration — round 2 (held-out test, weighted to corpus)

train=1389 test=596

| model | size | precision | recall | F1 | acc |
|---|---:|---:|---:|---:|---:|
| M0 gate | 0 | 0.762 | 0.914 | 0.831 | 0.799 |
| LR6 (round1, has url) | 6 | 0.826 | 0.896 | 0.860 | 0.842 |
| LR6* (entry-count, no url) | 6 | 0.888 | 0.860 | 0.874 | 0.866 |
| LR4 (count+author+imprint+cv-deny) | 4 | 0.869 | 0.852 | 0.860 | 0.850 |
| RULE (≥4 entries & author/year, header-veto) | 1 | 0.944 | 0.658 | 0.775 | 0.794 |

## Paired bootstrap of F1 deltas on the test (does the win exceed noise?)

- LR6* (entry-count, no url) − LR6 (round1, has url): ΔF1 = +0.015  [95% -0.022, +0.050]  → noise (spans 0)
- LR4 (count+author+imprint+cv-deny) − LR6 (round1, has url): ΔF1 = +0.001  [95% -0.038, +0.041]  → noise (spans 0)
- RULE (≥4 entries & author/year, header-veto) − LR6 (round1, has url): ΔF1 = -0.084  [95% -0.133, -0.036]  → REAL (excludes 0)

## Round-2 weights (justified model)

`LR6* (entry-count, no url)`: {"log_entry": 1.02, "author_init": 0.92, "year_paren": 0.48, "place_pub": 0.12, "header_cv": -0.59, "header_app": -0.05}
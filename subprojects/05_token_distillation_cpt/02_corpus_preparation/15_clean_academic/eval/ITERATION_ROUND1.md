# β-gate iteration — round 1 (held-out test, weighted to corpus)

train=1389 test=596

| candidate | #feats/rules | precision | recall | F1 | acc |
|---|---:|---:|---:|---:|---:|
| M0 current gate | 0 | 0.762 | 0.914 | 0.831 | 0.799 |
| M-idea: entry_density≥0.5 + header deny (2 rules) | 1 | 0.728 | 0.891 | 0.801 | 0.761 |
| M-LR3: entry+header_cv+header_app | 3 | 0.797 | 0.886 | 0.839 | 0.816 |
| M-LR6 | 6 | 0.822 | 0.895 | 0.857 | 0.838 |
| M-LR10 (all feats) | 10 | 0.903 | 0.867 | 0.885 | 0.878 |
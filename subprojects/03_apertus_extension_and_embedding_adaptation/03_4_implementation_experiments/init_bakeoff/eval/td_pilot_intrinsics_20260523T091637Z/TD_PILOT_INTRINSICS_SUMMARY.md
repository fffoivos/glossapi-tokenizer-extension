# TD Pilot Intrinsic Eval Summary

- Output root: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_pilot_intrinsics_20260523T091637Z`
- Eval JSONL: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
- Best BPC arm: `td_layer11`

## Heldout Greek Metrics

| arm | BPC | delta vs ReTok | NLL/char | tokens/word | STRR | docs |
|---|---:|---:|---:|---:|---:|---:|
| retok | 2.9503 | 0.0000 | 3.4896 | 1.7352 | 0.4458 | 500 |
| td_last | 2.7830 | -0.1673 | 3.2917 | 1.7352 | 0.4458 | 500 |
| td_layer11 | 2.7753 | -0.1750 | 3.2825 | 1.7352 | 0.4458 | 500 |

## New-Token Diagnostics

| arm | new targets | D1 mean rank | D1 top1 | D1 top5 | D2 new mass | D4 top1-new | D5 greedy-new | E norm ratio | U norm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retok | 276,332 | 3868.27 | 0.0065 | 0.0231 | 0.9821 | 0.9992 | 1.0000 | 1.0578 | 1.0140 |
| td_last | 276,332 | 3294.05 | 0.0146 | 0.0383 | 0.9796 | 0.9991 | 1.0000 | 1.0578 | 1.0142 |
| td_layer11 | 276,332 | 3263.69 | 0.0148 | 0.0385 | 0.9795 | 0.9991 | 1.0000 | 1.0579 | 1.0142 |

## Interpretation Notes

- Lower BPC/NLL is better.
- For D1, lower mean rank and higher top-k rates are better.
- D2/D4/D5 should move toward healthy use of new IDs without exploding relative to ReTok.
- If layer-11 improves BPC but shows unstable D-rank or output-norm behavior, prefer last-layer TD for the full run.


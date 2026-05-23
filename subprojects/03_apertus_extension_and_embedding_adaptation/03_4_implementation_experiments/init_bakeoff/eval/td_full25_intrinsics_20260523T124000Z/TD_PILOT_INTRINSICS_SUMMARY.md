# TD Pilot Intrinsic Eval Summary

- Output root: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_intrinsics_20260523T124000Z`
- Eval JSONL: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl`
- Best BPC arm: `td_full25_layer11`

## Heldout Greek Metrics

| arm | BPC | delta vs ReTok | NLL/char | tokens/word | STRR | docs |
|---|---:|---:|---:|---:|---:|---:|
| retok | 2.9503 | 0.0000 | 3.4896 | 1.7352 | 0.4458 | 500 |
| td_full25_last | 1.4249 | -1.5254 | 1.6853 | 1.7352 | 0.4458 | 500 |
| td_full25_layer11 | 1.3846 | -1.5657 | 1.6376 | 1.7352 | 0.4458 | 500 |

## New-Token Diagnostics

| arm | new targets | D1 mean rank | D1 top1 | D1 top5 | D2 new mass | D4 top1-new | D5 greedy-new | E norm ratio | U norm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retok | 276,332 | 3868.27 | 0.0065 | 0.0231 | 0.9821 | 0.9992 | 1.0000 | 1.0578 | 1.0140 |
| td_full25_last | 276,332 | 1756.04 | 0.0381 | 0.1596 | 0.8194 | 0.9266 | 0.9980 | 1.0585 | 1.0195 |
| td_full25_layer11 | 276,332 | 1617.48 | 0.0415 | 0.1722 | 0.8094 | 0.9153 | 0.9960 | 1.0590 | 1.0195 |

## Interpretation Notes

- Lower BPC/NLL is better.
- For D1, lower mean rank and higher top-k rates are better.
- D2/D4/D5 should move toward healthy use of new IDs without exploding relative to ReTok.
- If layer-11 improves BPC but shows unstable D-rank or output-norm behavior, prefer last-layer TD for the full run.


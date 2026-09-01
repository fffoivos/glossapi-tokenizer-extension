# Native-Greek peak-window evaluation (clean subset)

Status: **completed and receipt-verified** on 2026-08-17.

This evaluates checkpoints at approximately 30B, 35B, 40B, 45B and 50B
token slots. The 40B row reuses the authoritative existing evaluation; the
other four rows were evaluated here. Only benchmark examples retained by the
frozen contamination audit are scored.

## Population and authority

- Training dataset audited: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`
  at revision `987b8955fcd395c6219e39df9e64715457f69065`.
- Source benchmark examples: 83,970.
- Strong-match exclusions: 10,076.
- Retained clean examples per checkpoint: 73,894.
- Final result:
  `/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_peak_window_20260817/matrix_v1_issue94/peak_window_results.json`
- Result SHA-256:
  `9d618e7325b5b751b2c748f6f1152ad0d7b553c6141fb9f33eff70e91c546f81`.
- Final segment receipt SHA-256:
  `d9c2cf1f3a06efafe0dd1c67cbc5a8f547d6b82ca3f333b2671056f8fda2fef8`.
- Integrity audit: 84 unique shard receipts; 5 checkpoint rows; 9 reported
  benchmark views; all referenced receipt hashes reverified.

## Accuracy (%)

| Benchmark | 30B | 35B | 40B | 45B | 50B |
| --- | ---: | ---: | ---: | ---: | ---: |
| DemosQA | **47.75** | 46.41 | 46.41 | 46.41 | 46.91 |
| Medical MCQA | **41.77** | 39.62 | 40.81 | 41.53 | 41.53 |
| ASEP MCQA | 54.32 | 53.98 | 54.83 | 55.08 | **55.51** |
| GPCR | 58.25 | 59.79 | 61.34 | **63.40** | 62.89 |
| OYXOY NLI, raw | 63.44 | **63.96** | 62.22 | 62.36 | 59.94 |
| OYXOY NLI, exact set | 2.23 | 3.66 | 18.31 | 25.34 | **26.49** |
| OYXOY WSD-definition | 38.63 | 37.95 | 38.67 | 38.32 | **38.90** |
| OYXOY WiC | 56.43 | 42.11 | 39.31 | **67.90** | 33.77 |
| OYXOY metaphor | **49.17** | 39.32 | 35.46 | 34.08 | 33.89 |

## Choice NLL (lower is better)

| Benchmark | 30B | 35B | 40B | 45B | 50B |
| --- | ---: | ---: | ---: | ---: | ---: |
| DemosQA | 1.2900 | 1.2900 | 1.2855 | 1.2878 | **1.2855** |
| Medical MCQA | 1.5088 | 1.5139 | 1.5030 | 1.4992 | **1.4863** |
| ASEP MCQA | 1.2560 | 1.2602 | 1.2548 | 1.2522 | **1.2464** |
| GPCR | 0.6765 | 0.6774 | 0.6698 | 0.6683 | **0.6588** |
| OYXOY NLI | 0.6676 | **0.6575** | 0.6663 | 0.6682 | 0.6796 |
| OYXOY WSD-definition | 1.3046 | 1.3242 | **1.2974** | 1.3124 | 1.3096 |
| OYXOY WiC | 0.6767 | 0.7309 | 0.7444 | **0.6335** | 0.7823 |
| OYXOY metaphor | **0.7187** | 0.7446 | 0.8420 | 0.9889 | 1.3553 |

OYXOY NLI exact-set accuracy has no corresponding choice NLL.

## Main reading

- There is no universal best checkpoint. The 40B GreekMMLU-selected
  checkpoint is not the optimum for most of these tasks.
- ASEP and GPCR generally improve later. GPCR accuracy peaks at 45B, while its
  NLL continues improving at 50B.
- OYXOY metaphor regresses strongly and almost monotonically after 30B, in
  both accuracy and NLL.
- OYXOY WiC is highly unstable: it peaks sharply at 45B and collapses at 50B.
  Balanced accuracy changes much less than raw accuracy, indicating strong
  checkpoint-dependent answer-label bias rather than a simple capability
  trajectory.
- OYXOY NLI raw accuracy is also label-distribution-sensitive. Its exact-set
  score rises from 2.23% to 26.49%, while raw item accuracy falls late; report
  balanced accuracy and macro F1 with the raw score.
- DemosQA and Medical MCQA accuracy are comparatively flat/noisy, although
  Medical choice NLL improves clearly by 50B.

## Execution note

The original four-node, 22-minute segment profile timed out with two shards
unfinished. The 12 completed receipts were preserved and only the two missing
shards were resumed. Remaining work used a receipt-verified two-node profile
with a longer wall-time inside the same 88 node-minute QoS ceiling. A separate
post-processing defect counted four `combined/receipt.json` files as shard
receipts; the filter was corrected and finalization was rerun with zero model
inference. Neither correction changed examples, scorer settings or predictions.

# Iter-195 Bakeoff Digest

- run tag: `bakeoff_1node_chain_20260522_005620`
- checkpoint: `iter_0000195` (~0.818B tokens at checkpoint, 195 / 476 planned steps)
- job status: conversion, Greek-only lm-eval, tokenizer-fair metrics, and new-token diagnostics all completed with Slurm exit `0:0` for all three arms.
- reading: still pre-decision. Vanilla remains strongest on Greek downstream and tokenizer-fair BPC. ReTok has narrowed the gap and remains clearly better than Centroid on Greek downstream and new-token integration. Centroid is still weak on Greek use.

| arm | BPC | NLL/char | el ARC | el Belebele | el XNLI | el XQuAD F1 | el MMLU | el Base44 | el PIQA |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.5293 | 0.6262 | 0.4164 [0.3882, 0.4454] | 0.5478 [0.5155, 0.5800] | 0.4028 [0.3839, 0.4221] | 0.3219 [0.3020, 0.3409] | 0.4381 [0.4299, 0.4460] | 0.4493 [0.4076, 0.4891] | 0.6400 [0.5500, 0.7400] |
| retok | 0.6827 | 0.8075 | 0.3259 [0.2986, 0.3524] | 0.4467 [0.4133, 0.4811] | 0.3803 [0.3610, 0.3984] | 0.3077 [0.2875, 0.3275] | 0.3858 [0.3765, 0.3935] | 0.3786 [0.3370, 0.4185] | 0.5800 [0.4898, 0.6700] |
| centroid | 1.0396 | 1.2296 | 0.2594 [0.2346, 0.2850] | 0.3244 [0.2956, 0.3567] | 0.3627 [0.3446, 0.3815] | 0.0261 [0.0189, 0.0338] | 0.2841 [0.2765, 0.2915] | 0.3043 [0.2645, 0.3424] | 0.5300 [0.4300, 0.6300] |

New-token diagnostics at iter-195:

| arm | n_new | D1 top1 new-target | D2 avg mass on new vocab | D4 top1-new rate | D5 generation new-token use |
|---|---:|---:|---:|---:|---:|
| vanilla | 0 | n/a | 0.0000 | n/a | 0.000 |
| retok | 17,408 | 0.2526 | 0.3398 | 0.4829 | 0.102 |
| centroid | 17,408 | 0.0403 | 0.3361 | 0.2066 | 0.036 |

CI brackets are 95% bootstrap intervals from per-sample lm-eval outputs. `BPC` and `NLL/char` come from the 500-document tokenizer-fair heldout.

# Iter-260 Bakeoff Digest

- run tag: `bakeoff_1node_chain_20260522_005620`
- checkpoint: `iter_0000260` (~1.091B tokens at checkpoint, 260 / 476 planned steps)
- job status: conversion, full lm-eval, tokenizer-fair metrics, and new-token diagnostics all completed with Slurm exit `0:0` for all three arms.
- reading: still pre-decision. Vanilla remains strongest on Greek BPC and most Greek downstream metrics. ReTok is still behind vanilla but keeps narrowing and has the only healthy extended-token integration profile. Centroid remains a poor Greek-init candidate.

| arm | BPC | NLL/char | el ARC | el Belebele | el XNLI | el XQuAD F1 | el MMLU | el Base44 | el PIQA |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.5173 | 0.6120 | 0.4061 [0.3797, 0.4326] | 0.5067 [0.4744, 0.5400] | 0.4092 [0.3912, 0.4273] | 0.3022 [0.2825, 0.3225] | 0.4285 [0.4199, 0.4367] | 0.4239 [0.3786, 0.4656] | 0.6200 [0.5300, 0.7200] |
| retok | 0.6370 | 0.7535 | 0.3439 [0.3183, 0.3678] | 0.4600 [0.4311, 0.4922] | 0.3735 [0.3566, 0.3908] | 0.3261 [0.3056, 0.3463] | 0.3829 [0.3739, 0.3913] | 0.4112 [0.3659, 0.4511] | 0.5800 [0.4800, 0.6800] |
| centroid | 0.9875 | 1.1680 | 0.2551 [0.2304, 0.2782] | 0.3378 [0.3078, 0.3700] | 0.3398 [0.3217, 0.3574] | 0.0239 [0.0169, 0.0317] | 0.2834 [0.2761, 0.2909] | 0.3098 [0.2681, 0.3478] | 0.5100 [0.4100, 0.6100] |

New-token diagnostics at iter-260:

| arm | n_new | D1 top1 new-target | D2 avg mass on new vocab | D4 top1-new rate | D5 generation new-token use |
|---|---:|---:|---:|---:|---:|
| vanilla | 0 | n/a | 0.0000 | n/a | 0.000 |
| retok | 17,408 | 0.2915 | 0.3388 | 0.5268 | 0.150 |
| centroid | 17,408 | 0.0606 | 0.3406 | 0.2625 | 0.076 |

Retention/general-task snapshot from the aligned full eval:

| arm | ARC-C | HellaSwag | WinoGrande | PIQA | MMLU |
|---|---:|---:|---:|---:|---:|
| vanilla | 0.536 | 0.761 | 0.689 | 0.786 | 0.540 |
| retok | 0.509 | 0.738 | 0.680 | 0.784 | 0.545 |
| centroid | 0.546 | 0.757 | 0.691 | 0.801 | 0.549 |

CI brackets are 95% bootstrap intervals from per-sample lm-eval outputs. `BPC` and `NLL/char` come from the 500-document tokenizer-fair heldout.

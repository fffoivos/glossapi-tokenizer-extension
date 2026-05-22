# Iter-325 Bakeoff Digest

- run tag: `bakeoff_1node_chain_20260522_005620`
- checkpoint: `iter_0000325` (~1.363B tokens at checkpoint, 325 / 476 planned steps)
- job status: conversion, tokenizer-fair metrics, and new-token diagnostics completed with Slurm exit `0:0`; the original three per-arm full evals were cancelled during CPU/dataset setup and replaced by packed full-eval job `2345516`, which completed with Slurm exit `0:0` in `00:43:39`.
- reading: still pre-decision. Vanilla remains strongest on Greek BPC and most Greek downstream metrics. ReTok keeps closing the BPC gap and has the healthiest extended-token integration profile. Centroid remains a poor Greek-init candidate for the Greek objective.

| arm | BPC | NLL/char | el ARC | el Belebele | el XNLI | el XQuAD F1 | el MMLU | el Base44 | el PIQA |
|---|---:|---:|---|---|---|---|---|---|---|
| vanilla | 0.5045 | 0.5968 | 0.4121 [0.3831, 0.4411] | 0.5144 [0.4822, 0.5500] | 0.4028 [0.3835, 0.4213] | 0.3193 [0.2996, 0.3388] | 0.4240 [0.4151, 0.4322] | 0.4275 [0.3822, 0.4692] | 0.6300 [0.5300, 0.7200] |
| retok | 0.6070 | 0.7179 | 0.3626 [0.3353, 0.3874] | 0.4767 [0.4456, 0.5078] | 0.3699 [0.3526, 0.3888] | 0.3134 [0.2923, 0.3335] | 0.3891 [0.3800, 0.3970] | 0.3877 [0.3442, 0.4294] | 0.5600 [0.4700, 0.6600] |
| centroid | 0.9525 | 1.1266 | 0.2491 [0.2244, 0.2722] | 0.3233 [0.2933, 0.3545] | 0.3482 [0.3305, 0.3659] | 0.0257 [0.0185, 0.0334] | 0.2827 [0.2752, 0.2904] | 0.3007 [0.2627, 0.3388] | 0.5100 [0.4100, 0.6100] |

New-token diagnostics at iter-325:

| arm | n_new | D1 top1 new-target | D2 avg mass on new vocab | D4 top1-new rate | D5 generation new-token use |
|---|---:|---:|---:|---:|---:|
| vanilla | 0 | n/a | 0.0000 | n/a | 0.000 |
| retok | 17,408 | 0.3183 | 0.3455 | 0.5612 | 0.161 |
| centroid | 17,408 | 0.0757 | 0.3409 | 0.2939 | 0.068 |

Retention/general-task snapshot from the aligned full eval:

| arm | ARC-C | HellaSwag | WinoGrande | PIQA | MMLU |
|---|---:|---:|---:|---:|---:|
| vanilla | 0.536 | 0.763 | 0.705 | 0.791 | 0.534 |
| retok | 0.510 | 0.740 | 0.681 | 0.785 | 0.547 |
| centroid | 0.538 | 0.757 | 0.695 | 0.792 | 0.546 |

Operational note: the packed eval path is now validated on a real checkpoint. It used one 4-GPU node instead of three whole-node one-GPU eval allocations, with GPUs 0/1/2 doing the arm evals and GPU3 left as spare.

CI brackets are 95% bootstrap intervals from per-sample lm-eval outputs. `BPC` and `NLL/char` come from the 500-document tokenizer-fair heldout.

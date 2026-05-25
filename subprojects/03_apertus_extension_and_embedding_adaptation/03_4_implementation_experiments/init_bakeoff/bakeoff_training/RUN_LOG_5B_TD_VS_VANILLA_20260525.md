# 5B TD vs Vanilla Continuation Run Log

Date: 2026-05-25.

Goal: continue only `TokenDistil`/`td_layer11` and `Vanilla` from the 3.5B
checkpoints to ~5B tokens, while evaluating saved checkpoints in parallel with
training.

Decision target:

- Does TD beat the matched Vanilla control at the same CPT point?
- Does TD close enough of the gap to the original Apertus / Vanilla-init
  baseline to justify longer production training?

Run shape:

```text
source checkpoints:
  continuation_3p5b_20260524T143012Z_vanilla/checkpoints/iter_0000834
  continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834

target checkpoints:
  iter_0001013  ~4.249B tokens
  iter_0001192  ~4.9996B tokens

arms:
  vanilla
  td_layer11

training:
  2 arms in parallel, each with two chained segments

eval:
  sidecar submitter on xfer
  conversion/intrinsic/packed lm-eval jobs depend on checkpoint-producing
  segment only; later training segment keeps running
```

Files added for this run:

```text
bakeoff_training/submit_5b_td_vs_vanilla_chain.sh
bakeoff_training/RUN_LOG_5B_TD_VS_VANILLA_20260525.md
eval/submit_3p5b_eval_sidecars_incremental.py
```

The eval submitter was generalized with `EVAL_ARMS`, `DIAG_ARMS`, and
`PACKED_JOB_PREFIX`; default behavior remains the original 3.5B three-arm
configuration.

## Launch Record

Submitted live on Clariden at 2026-05-25 14:25 UTC.

Run tag:

```text
continuation_5b_td_vs_vanilla_20260525T142522Z
```

State paths:

```text
/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_5b_td_vs_vanilla_20260525T142522Z_submit_state
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_sidecar_eval_incremental
```

Training jobs:

```text
2382982  5b_vanilla_1013       no dependency
2382983  5b_vanilla_1192       afterok:2382982
2382984  5b_td_layer11_1013    no dependency
2382985  5b_td_layer11_1192    afterok:2382984
```

Eval submitter:

```text
2382986  eval_submit_5b        xfer
```

Initial sidecars submitted by the eval watcher:

```text
2382998  tohf_vanilla_1013     afterok:2382982
2382999  bpc_vanilla_1013      afterok:2382998
2383000  tohf_td_layer11_1013  afterok:2382984
2383001  bpc_td_layer11_1013   afterok:2383000
2383002  diag_td_layer11_1013  afterok:2383000
2383003  eval_5b_1013_full     afterok:2382998:2383000
```

The 1192 / ~5B sidecars are intentionally left to the running xfer submitter,
which trickles jobs under the account submit limit while training continues.

## First Health Check

Checked at 2026-05-25 14:46 UTC.

Both first-segment training jobs were running and had completed checkpoint load,
dataset build, forward/backward, and optimizer step. First observed iteration
lines:

```text
vanilla    job 2382982  iter 837/1013  3.511B tokens  loss 1.631706  8020.6 tokens/sec/GPU  eta ~6:23
td_layer11 job 2382984  iter 837/1013  3.511B tokens  loss 2.411584  7851.2 tokens/sec/GPU  eta ~6:32
```

Both logs reported:

```text
number of skipped iterations: 0
number of nan iterations: 0
```

The xIELU optimizer audit reported `missing=0` for both arms during checkpoint
load. The stderr content at this point was warnings only.

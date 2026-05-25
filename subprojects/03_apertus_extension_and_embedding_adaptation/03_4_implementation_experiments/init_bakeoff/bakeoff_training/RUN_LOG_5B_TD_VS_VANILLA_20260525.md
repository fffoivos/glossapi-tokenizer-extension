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

## Steady-State Monitor

Checked at 2026-05-25 14:56 UTC.

Both first-segment jobs remained healthy:

```text
vanilla    job 2382982  iter 842/1013  3.532B tokens  loss 1.640331  8020.7 tokens/sec/GPU
td_layer11 job 2382984  iter 841/1013  3.527B tokens  loss 2.391435  7851.0 tokens/sec/GPU
```

The 4.25B sidecar eval jobs are dependency-staged. The xfer submitter is still
running and retrying the 5B sidecar DAG; at this point only the six `iter_1013`
sidecars are recorded in the incremental state file. This does not block
training, because the second training segment already depends directly on the
first training segment and not on eval completion.

A lightweight home-side status logger was started for breadcrumbs only:

```text
systemd user unit: codex-5b-td-monitor-20260525.service
main pid at start: 495253
log: /home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/monitor.log
script: /home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/monitor_status.sh
repo copy: bakeoff_training/monitor_5b_td_vs_vanilla_status.sh
```

## Intermediate Checkpoint Check

Checked at 2026-05-25 15:07 UTC.

Both arms completed the first intermediate async save and continued training:

```text
vanilla    job 2382982  iter 846/1013  3.548B tokens  loss 1.628929
td_layer11 job 2382984  iter 846/1013  3.548B tokens  loss 2.382900
```

The `iter_0000845` directories contain the expected `common.pt` plus eight
`*.distcp` shard files per arm, and both logs contain:

```text
successfully saved checkpoint from iteration     845
```

The previous 3.5B continuation logs show that final target iterations are saved
at job end even when they are not regular `SAVE_INTERVAL` multiples; for
example, the repaired Vanilla 834 segment saved `iter_0000834` after training
was done. That keeps the 1013 conversion dependency plausible without changing
the live jobs.

## Eval Submitter Repair

Checked at 2026-05-25 15:11 UTC.

The original xfer eval submitter (`2382986`) was alive but repeatedly failed to
stage the first 5B sidecar with:

```text
sbatch: error: QOSMaxSubmitJobPerUserLimit
allocation failure: Job violates accounting/QOS policy
```

Root cause: the launcher used `MAX_SUBMITTED_JOBS=14`, while Clariden refused
the 12th active job for this user. This only affected future 5B sidecar
submission; the 4.25B eval DAG and both training chains were already intact.

Action taken:

```text
scancel 2382986
2383700  eval_submit_5b_fix   failed immediately due bad wrapped script path
2383705  eval_submit_5b_fix2  running with MAX_SUBMITTED_JOBS=11 and python -u
```

The repaired submitter log now reports:

```text
state: submitted=6 missing=6 active_jobs=11
next_missing: convert:1192:vanilla, bpc:1192:vanilla, convert:1192:td_layer11, bpc:1192:td_layer11, diag:1192:td_layer11, packed:1192:full
```

The launcher was updated to use `MAX_SUBMITTED_JOBS=11`, `PYTHONUNBUFFERED=1`,
and an explicit eval working directory for future launches.

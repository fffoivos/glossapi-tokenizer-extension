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

## Monitor Hardening

Checked at 2026-05-25 15:14 UTC.

The home-side systemd monitor was restarted with a name-pattern stop condition
instead of only the original job-id list. This matters because the repaired
eval submitter (`2383705`) and the future 1192 sidecars were not known when
the monitor was first launched.

The monitor now keeps running until:

```text
tracked_active_count=0
tracked_sidecar_rows=12
```

The fresh monitor sample showed:

```text
vanilla    job 2382982  iter 850/1013  3.565B tokens  loss 1.647988
td_layer11 job 2382984  iter 849/1013  3.561B tokens  loss 2.378793
eval_submit_5b_fix2 job 2383705 running with sidecar_rows=6
```

## Post-Resume Health Check

Checked at 2026-05-25 15:22 UTC.

The two 4.25B training jobs are still running normally, the two 5B continuation
jobs remain dependency-pending, and the repaired eval submitter is still alive.

Latest matched training lines:

```text
vanilla    job 2382982  iter 853/1013  3.578B tokens  loss 1.647105  skipped=0 nan=0
td_layer11 job 2382984  iter 853/1013  3.578B tokens  loss 2.377223  skipped=0 nan=0
```

Current checkpoint state:

```text
vanilla    latest_checkpointed_iteration.txt = 845
td_layer11 latest_checkpointed_iteration.txt = 845
```

The eval sidecar state remains at 6/12 rows, all for the 1013 checkpoint. This
is expected while the user job cap is full: the missing 1192 sidecars should be
submitted by `eval_submit_5b_fix2` once the 1013 training jobs finish and the
active job count drops below the cap.

## Continued First-Leg Health Check

Checked at 2026-05-25 16:48 UTC.

Both first-leg jobs continued cleanly toward the 4.25B checkpoint boundary:

```text
vanilla    job 2382982  iter 893/1013  3.746B tokens  loss 1.641109  skipped=0 nan=0
td_layer11 job 2382984  iter 891/1013  3.737B tokens  loss 2.362208  skipped=0 nan=0
```

Current checkpoint state remains:

```text
vanilla    latest_checkpointed_iteration.txt = 845
td_layer11 latest_checkpointed_iteration.txt = 845
```

The 1013 sidecars are still dependency-staged, the 1192 training jobs are still
dependency-pending, and `eval_submit_5b_fix2` is still running with
`submitted=6 missing=6 active_jobs=11`. This is the expected state before the
1013 checkpoint save and handoff.

## Second Intermediate Checkpoint Check

Checked at 2026-05-25 17:49 UTC.

Both arms saved the second intermediate checkpoint and continued training:

```text
vanilla    job 2382982  iter 921/1013  3.863B tokens  loss 1.616130  skipped=0 nan=0
td_layer11 job 2382984  iter 919/1013  3.855B tokens  loss 2.365760  skipped=0 nan=0
```

Current checkpoint state:

```text
vanilla    latest_checkpointed_iteration.txt = 910
td_layer11 latest_checkpointed_iteration.txt = 910

vanilla    iter_0000910 timestamp 2026-05-25 19:27
td_layer11 iter_0000910 timestamp 2026-05-25 19:31
```

The 1013 eval sidecars remain dependency-staged, the 1192 training jobs remain
dependency-pending, and `eval_submit_5b_fix2` is still waiting under the active
job cap with `submitted=6 missing=6 active_jobs=11`.

## Third Intermediate Checkpoint Check

Checked at 2026-05-25 20:00 UTC.

Both arms saved the last regular intermediate checkpoint before the 1013 target
handoff and continued training:

```text
vanilla    job 2382982  iter 981/1013  4.115B tokens  loss 1.608867  skipped=0 nan=0
td_layer11 job 2382984  iter 978/1013  4.102B tokens  loss 2.355478  skipped=0 nan=0
```

Current checkpoint state:

```text
vanilla    latest_checkpointed_iteration.txt = 975
td_layer11 latest_checkpointed_iteration.txt = 975

vanilla    iter_0000975 timestamp 2026-05-25 21:49
td_layer11 iter_0000975 timestamp 2026-05-25 21:55
```

The run is now close enough to the 1013 boundary that monitoring should switch
from hourly to shorter checks. The expected next transition is:

```text
iter_0001013 appears for both arms
2382983 / 2382985 start the 1192 training legs
2382998 / 2383000 start HF conversion for 1013
```

## 4.25B Checkpoint Handoff

Checked at 2026-05-25 21:19 UTC.

Both first-leg training jobs reached the target `iter_0001013` checkpoint and
exited cleanly:

```text
vanilla    job 2382982  COMPLETED 0:0  iter 1013/1013  4.249B tokens  final lm loss 1.611620  skipped=0 nan=0
td_layer11 job 2382984  COMPLETED 0:0  iter 1013/1013  4.249B tokens  final lm loss 2.314037  skipped=0 nan=0
```

Checkpoint state:

```text
vanilla    latest_checkpointed_iteration.txt = 1013
td_layer11 latest_checkpointed_iteration.txt = 1013

vanilla    iter_0001013 timestamp 2026-05-25 23:10
td_layer11 iter_0001013 timestamp 2026-05-25 23:18
```

The first 1013 conversion jobs also handed off correctly:

```text
tohf_vanilla_1013     job 2382998  COMPLETED 0:0  elapsed 00:01:12
tohf_td_layer11_1013  job 2383000  COMPLETED 0:0  elapsed 00:01:09
```

Current queue state after the handoff:

```text
2382983  5b_vanilla_1192       PENDING (Priority)
2382985  5b_td_layer11_1192    PENDING (Priority)
2382999  bpc_vanilla_1013      PENDING (Priority)
2383001  bpc_td_layer11_1013   PENDING (Dependency/Priority transition after conversion)
2383002  diag_td_layer11_1013  PENDING (Dependency/Priority transition after conversion)
2383003  eval_5b_1013_full     PENDING (Dependency/Priority transition after conversion)
```

The incremental eval submitter is still alive as `2383705`. It successfully
added the first two 1192 sidecar rows once active job count dropped:

```text
convert:1192:vanilla -> 2388813
bpc:1192:vanilla     -> 2388814
```

It also logged transient `QOSMaxSubmitJobPerUserLimit` failures while trying to
add the remaining 1192 sidecars. This is expected under the user job cap as
long as the submitter keeps retrying; next check should confirm whether it adds
`convert/bpc/diag/packed` rows for `td_layer11` and `1192` after active jobs
clear.

Follow-up at 2026-05-25 21:23 UTC: the submitter did keep retrying and the
sidecar table advanced to 10/12 rows:

```text
convert:1192:td_layer11 -> 2388835
bpc:1192:td_layer11     -> 2388836
```

The only missing future sidecars are now:

```text
diag:1192:td_layer11
packed:1192:full
```

Those remain intentionally unsubmitted while the job cap is full. Current
training/eval work is dependency-clear and queued on Slurm priority:

```text
2382983  5b_vanilla_1192       PENDING (Priority)
2382985  5b_td_layer11_1192    PENDING (Priority)
2382999  bpc_vanilla_1013      PENDING (Priority)
2383001  bpc_td_layer11_1013   PENDING (Priority)
2383002  diag_td_layer11_1013  PENDING (Priority)
2383003  eval_5b_1013_full     PENDING (Priority)
```

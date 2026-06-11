# 5B TD vs Vanilla Continuation Run Log

Date: 2026-05-25.

## 1192 Poll: Iter 1071/1072, Still Running

Checked at 2026-05-25 23:38 UTC.

Both final 1192 training jobs remain running. The local monitor and finalizer
services are active; no restart or manual intervention was needed. The final
conversion/BPB/diagnostic/downstream sidecars are still dependency-pending
behind the 1192 training jobs, and no final `iter_0001192` eval files exist yet.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 02:12:48  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 02:12:48  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1072/1192  4.496B tokens  loss 1.614770  skipped=0 nan=0  eta 4:23:50
td_layer11 iter 1071/1192  4.492B tokens  loss 2.309794  skipped=0 nan=0  eta 4:29:13
```

Checkpoint status remains `latest=1040` for both arms. The finalizer summary
JSON is still interim: latest summarized iteration `1013`, missing iteration
`1192`.

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

## 1192 Leg Started; 1013 Intrinsics Complete

Checked at 2026-05-25 21:40 UTC.

The scheduler started the 5B continuation legs and the 1013 sidecars:

```text
2382983  5b_vanilla_1192     RUNNING  since 23:25:03 UTC
2382985  5b_td_layer11_1192  RUNNING  since 23:25:03 UTC
2383003  eval_5b_1013_full   RUNNING  since 23:25:12 UTC
```

The 1013 intrinsic sidecars completed cleanly:

```text
2382999  bpc_vanilla_1013      COMPLETED 0:0  elapsed 00:01:56
2383001  bpc_td_layer11_1013   COMPLETED 0:0  elapsed 00:01:41
2383002  diag_td_layer11_1013  COMPLETED 0:0  elapsed 00:02:08
```

The incremental submitter completed after writing all 12 sidecar rows:

```text
2383705  eval_submit_5b_fix2  COMPLETED 0:0
sidecar rows: 12/12
```

Early 1192 training health:

```text
vanilla    job 2382983  iter 1019/1192  4.274B tokens  loss 1.611488  skipped=0 nan=0
td_layer11 job 2382985  iter 1018/1192  4.270B tokens  loss 2.351225  skipped=0 nan=0
```

The 1013 / ~4.25B tokenizer-fair heldout results are now available. Lower BPB
is better; raw `lm loss` is not used for cross-tokenizer selection. The table
renames the historical artifact field `BPC` to `BPB` below, because the
quantity is bits per byte.

```text
stage   arm         BPB       NLL/char  tok/word  chars/tok  STRR
2.0B    vanilla     0.490579  0.580385  2.692984  2.557154   0.270269
2.0B    td_layer11  0.531084  0.628153  1.735161  3.973179   0.445816
3.5B    vanilla     0.472385  0.558861  2.692984  2.557154   0.270269
3.5B    td_layer11  0.505436  0.597817  1.735161  3.973179   0.445816
4.25B   vanilla     0.465681  0.550929  2.692984  2.557154   0.270269
4.25B   td_layer11  0.495345  0.585882  1.735161  3.973179   0.445816
```

Reading so far: TD is still improving on tokenizer-fair heldout BPB
(`0.531084 -> 0.505436 -> 0.495345`), and it preserves the compression win
(`1.735` vs Vanilla `2.693` tokens/word; STRR `0.446` vs `0.270`). But Vanilla
also improves, so TD remains behind on BPB at matched token count. The gap is:

```text
2.0B gap TD - Vanilla:    +0.040505 BPB
3.5B gap TD - Vanilla:    +0.033051 BPB
4.25B gap TD - Vanilla:   +0.029664 BPB
```

TD new-token integration diagnostics remain healthy rather than collapsed:

```text
stage   top1_new_target  top5_new_target  mean_rank  mass_new  greedy_new_util
2.0B    0.386419         0.555687         206.412    0.342453  0.208
3.5B    0.410535         0.581142         174.288    0.342085  0.282
4.25B   0.419633         0.590297         162.710    0.341012  0.260
```

This is encouraging for continued TD learning, but not yet sufficient for the
decision. The 1013 downstream eval is still running, and the final matched 1192
checkpoint/eval is still required.

## 1013 Downstream Eval Complete

Checked at 2026-05-25 22:13 UTC.

The packed 1013 downstream eval finished cleanly:

```text
2383003  eval_5b_1013_full  COMPLETED 0:0  elapsed 00:47:31
```

Result files:

```text
vanilla:
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_vanilla/iter_0001013_full/results_2026-05-26T00-12-33.815872.json

td_layer11:
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_td_layer11/iter_0001013_full/results_2026-05-26T00-11-16.286948.json
```

Matched-task scoreboard at `iter_0001013`:

```text
task                    group           vanilla     td_layer11   TD-Vanilla  winner
MMLU                    EN retention    0.536818    0.561102    +0.024284   TD
HellaSwag               EN retention    0.757917    0.765286    +0.007369   TD
ARC Easy                EN retention    0.784933    0.786616    +0.001684   TD
ARC Challenge           EN retention    0.516212    0.530717    +0.014505   TD
PIQA                    EN retention    0.797606    0.791621    -0.005985   Vanilla
Winogrande              EN retention    0.681137    0.700079    +0.018942   TD
Global MMLU             Multilingual    0.445405    0.461612    +0.016207   TD
XCOPA                   Multilingual    0.615818    0.618545    +0.002727   TD
XNLI                    Multilingual    0.410040    0.411914    +0.001874   TD
Greek MMLU              Greek           0.418245    0.413260    -0.004985   Vanilla
INCLUDE-44 Greek        Greek           0.418478    0.411232    -0.007246   Vanilla
Belebele Greek          Greek           0.516667    0.530000    +0.013333   TD
ARC Challenge MT-el     Greek           0.419795    0.401877    -0.017918   Vanilla
XNLI Greek              Greek           0.392771    0.387952    -0.004819   Vanilla
XQuAD Greek F1          Greek           0.281595    0.354168    +0.072573   TD
PIQA Greek              Greek           0.620000    0.570000    -0.050000   Vanilla
```

Group aggregates:

```text
EN retention:  Vanilla 0.679104  TD 0.689237  TD-Vanilla +0.010133
Multilingual:  Vanilla 0.490421  TD 0.497357  TD-Vanilla +0.006936
Greek:         Vanilla 0.438222  TD 0.438356  TD-Vanilla +0.000134
```

Reading at 4.25B: TD is clearly ahead on EN retention and multilingual
aggregates and essentially tied on Greek downstream mean. The Greek aggregate
is not a broad Greek win yet: Vanilla wins 5/7 Greek tasks, while TD's large
XQuAD F1 gain plus Belebele gain offset those losses. Combined with the
tokenizer-fair BPB gap (TD still worse by `+0.029664` BPB), this remains a
promising-but-not-settled TD trajectory. The final `iter_0001192` matched eval
is still required for the decision.

## 1192 Final Leg Still Running

Checked at 2026-05-25 22:23 UTC.

Both final 1192 training jobs are still healthy and running:

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 00:58:07  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 00:58:07  node nid006211
```

The 1192 sidecars are all correctly submitted and dependency-pending:

```text
2388813  tohf_vanilla_1192       PENDING (Dependency)
2388814  bpc_vanilla_1192        PENDING (Dependency)
2388835  tohf_td_layer11_1192    PENDING (Dependency)
2388836  bpc_td_layer11_1192     PENDING (Dependency)
2388866  diag_td_layer11_1192    PENDING (Dependency)
2388867  eval_5b_1192_full       PENDING (Dependency)
```

No final 1192 files exist yet under the eval roots. The latest checkpoint
tracker for both arms still points to `1013`, as expected before the final save.

Latest visible training lines:

```text
vanilla    iter 1038/1192  4.354B tokens  loss 1.609062  skipped=0 nan=0  eta 5:38:32
td_layer11 iter 1038/1192  4.354B tokens  loss 2.350937  skipped=0 nan=0  eta 5:42:25
```

Reading: no restart or intervention was needed in this poll. Both final jobs
are progressing at roughly 7.9k tokens/sec/GPU with no skipped or NaN
iterations. The decision remains blocked on the final checkpoint plus its
dependent conversion, BPB/diagnostics, and packed downstream eval.

## 1192 Poll: Intermediate 1040 Checkpoint Saved

Checked at 2026-05-25 22:38 UTC.

Both final 1192 jobs are still running cleanly. No restart or intervention was
needed.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:13:03  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:13:03  node nid006211
```

Both arms saved the periodic intermediate checkpoint `iter_0001040`:

```text
vanilla latest_checkpointed_iteration.txt = 1040
td_layer11 latest_checkpointed_iteration.txt = 1040
```

Latest visible training lines:

```text
vanilla    iter 1045/1192  4.383B tokens  loss 1.592180  skipped=0 nan=0  eta 5:22:42
td_layer11 iter 1044/1192  4.379B tokens  loss 2.315261  skipped=0 nan=0  eta 5:29:08
```

The final `iter_0001192` files still do not exist under the eval roots. The
1192 sidecars remain correctly dependency-pending:

```text
2388813  tohf_vanilla_1192       PENDING (Dependency)
2388814  bpc_vanilla_1192        PENDING (Dependency)
2388835  tohf_td_layer11_1192    PENDING (Dependency)
2388836  bpc_td_layer11_1192     PENDING (Dependency)
2388866  diag_td_layer11_1192    PENDING (Dependency)
2388867  eval_5b_1192_full       PENDING (Dependency)
```

The local monitor remains active as
`codex-5b-td-monitor-20260525.service`, polling every 10 minutes and writing
`/home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/monitor.log`.

## 1192 Poll: Still Healthy After Loss-Policy Doc Cleanup

Checked at 2026-05-25 22:44 UTC.

Both final 1192 jobs are still running cleanly. No restart or intervention was
needed.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:19:15  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:19:15  node nid006211
```

Checkpoint status is unchanged from the prior poll: both arms have the
intermediate `iter_0001040` checkpoint and no final `iter_0001192` eval files
exist yet.

Latest visible training lines:

```text
vanilla    iter 1047/1192  4.391B tokens  loss 1.598202  skipped=0 nan=0  eta 5:18:20
td_layer11 iter 1047/1192  4.391B tokens  loss 2.351325  skipped=0 nan=0  eta 5:22:35
```

The final sidecars remain dependency-pending behind the 1192 training jobs:

```text
2388813  tohf_vanilla_1192       PENDING (Dependency)
2388814  bpc_vanilla_1192        PENDING (Dependency)
2388835  tohf_td_layer11_1192    PENDING (Dependency)
2388836  bpc_td_layer11_1192     PENDING (Dependency)
2388866  diag_td_layer11_1192    PENDING (Dependency)
2388867  eval_5b_1192_full       PENDING (Dependency)
```

Note: these already-submitted Slurm jobs still use the historical `bpc_*`
names. New repo submitters now use `bpb_*`, while readers keep accepting legacy
`bpc` state rows and `bpc_bits_per_byte` artifact fields.

## Local 5B Report Tooling Prepared

Checked at 2026-05-25 22:49 UTC.

Final `iter_0001192` artifacts are still not present, but the already-complete
`iter_0001013` / ~4.25B eval artifacts have now been copied into the local
trajectory bundle:

```text
eval/trajectory_analysis_20260524/per_iter_results/vanilla_iter1013.json
eval/trajectory_analysis_20260524/per_iter_results/td_iter1013.json
eval/trajectory_analysis_20260524/per_iter_results/intrinsic/vanilla_iter1013_fair.json
eval/trajectory_analysis_20260524/per_iter_results/intrinsic/td_iter1013_fair.json
eval/trajectory_analysis_20260524/per_iter_results/diagnostics/td_iter1013_new_token_diagnostics.json
```

Added reusable local helpers:

```text
eval/trajectory_analysis_20260524/collect_5b_continuation_artifacts.sh
eval/trajectory_analysis_20260524/summarize_5b_continuation.py
```

The collector copies only lightweight JSON/log artifacts from Clariden, not
checkpoint weights. It already copied 1013 artifacts and current 1192 training
logs; it will copy 1192 eval JSONs once the dependency-pending sidecars finish.

The summarizer now writes an interim report:

```text
eval/trajectory_analysis_20260524/CONTINUATION_5B_RESULTS_20260526.md
eval/trajectory_analysis_20260524/continuation_5b_summary.json
```

Current interim reading at 1013 / ~4.25B: Vanilla still wins heldout BPB
(`0.4657` vs TD `0.4953`), TD leads EN retention and multilingual aggregates,
and the Greek aggregate is effectively tied (`+0.0001` TD minus Vanilla). This
is not the final decision; 1192 downstream eval plus BPB/diagnostics remain
required.

## 1192 Poll: Iter 1050, Still Running

Checked at 2026-05-25 22:50 UTC.

Both final 1192 jobs remain healthy. No restart or intervention was needed.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:25:34  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:25:34  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1050/1192  4.404B tokens  loss 1.610176  skipped=0 nan=0  eta 5:11:44
td_layer11 iter 1050/1192  4.404B tokens  loss 2.326564  skipped=0 nan=0  eta 5:15:41
```

Checkpoint status remains `latest=1040` for both arms. No final `iter_0001192`
eval files exist yet. The conversion, BPB, diagnostics, and packed full-eval
sidecars are still dependency-pending behind the training jobs.

## Loss-Measurement Docs Synced Globally

Checked at 2026-05-25 23:05 UTC.

Updated the repo and release docs so the loss policy is consistent everywhere
new readers are likely to start:

```text
README.md
docs/PROJECT_INDEX.md
docs/CURRENT_STATUS.md
release/apertus-tokenizer-extension/README.md
release/apertus-tokenizer-extension/benchmark-evals/3.5B-comparison/README.md
release/apertus-tokenizer-extension/supporting-material/provenance/evals/*
release/apertus-tokenizer-extension/supporting-material/provenance/token-distillation/TOKEN_DISTILLATION_PLAN.md
```

The public wording now says: heldout BPB plus downstream evals are the
cross-tokenizer evidence; raw Megatron `lm loss` is health/within-arm telemetry;
older `BPC` labels are legacy bits-per-byte aliases. The copied provenance docs
in the HF release tree were synced from the canonical source docs. The 5B
monitor pattern now accepts both already-submitted `bpc_*` jobs and future
`bpb_*` jobs.

## 1192 Poll: Iter 1053, Still Running

Checked at 2026-05-25 22:55 UTC.

Both final 1192 training jobs remain healthy; no restart or intervention was
needed. Final conversion/BPB/diagnostic/eval sidecars remain dependency-pending.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:30:34  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:30:34  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1053/1192  4.417B tokens  loss 1.607587  skipped=0 nan=0  eta 5:05:12
td_layer11 iter 1052/1192  4.412B tokens  loss 2.288327  skipped=0 nan=0  eta 5:11:13
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## 1192 Dependency Audit: Final Sidecars Ready

Checked at 2026-05-26 00:19 UTC.

The final sidecar DAG is queued correctly:

- `tohf_vanilla_1192` depends on training job `2382983`.
- `tohf_td_layer11_1192` depends on training job `2382985`.
- Vanilla/TD tokenizer-fair BPB jobs depend on their corresponding HF
  conversion jobs.
- TD new-token diagnostics depends on the TD HF conversion job.
- `eval_5b_1192_full` depends on both HF conversion jobs, so full downstream
  eval starts only after both arms are ready.

The `tohf_*` conversion jobs request one GPU on `debug`. This is intentional:
`convert_bakeoff_checkpoint_to_hf.sbatch` documents that Clariden `xfer` nodes
do not expose the needed `uenv`/Torch stack and that Megatron's checkpoint
metadata reader creates a CUDA tensor while loading tracker metadata. The jobs
are bounded to one hour and are not dataset-build CPU work.

## Loss-Measurement Cleanup Follow-Up: Historical Log Labels

Checked the maintained Markdown/script surface again after the BPB policy pass.
The remaining uppercase `BPC` references are now either explicit compatibility
notes, legacy JSON keys such as `bpc_bits_per_byte`, or historical Slurm job
names. I also relabeled the metric-value lines in
`TAKEOVER_LOG_20260521.md` from `BPC` to `BPB`, while preserving immutable
legacy key/job names.

## Loss Measurement Docs/Scripts Cleanup

Checked at 2026-05-25 23:15 UTC.

Repo docs and local release docs now point to the same rule: cross-tokenizer
loss decisions use heldout BPB and downstream evals, while raw Megatron
`lm loss` is per-target-token CE and health/within-arm telemetry only.

The canonical policy lives at:

```text
init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md
```

Updated scripts accept both new BPB names and legacy `BPC` /
`bpc_bits_per_byte` artifacts, and the training-log parsers understand dense
`bpb`, `bpt`, `base_loss`, `new_loss`, and `n_new` fields when present. The
syntax checks passed for the touched Python and shell wrappers, and
`git diff --check` was clean.

## 1192 Poll: No Material Change

Checked at 2026-05-25 23:05 UTC.

No material change from the 23:04 UTC poll: both final training jobs are still
`RUNNING`, all final sidecars are still dependency-pending, checkpoint status
remains `latest=1040` for both arms, and no final `iter_0001192` eval artifacts
exist yet.

## 1192 Poll: Both Arms at Iter 1057

Checked at 2026-05-25 23:05 UTC.

Both final 1192 training jobs remain running. TD has caught up to the same
latest visible iteration as Vanilla. Sidecars remain dependency-pending, and no
final checkpoint or final eval artifacts exist yet.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:40:51  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:40:51  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1057/1192  4.433B tokens  loss 1.605402  skipped=0 nan=0  eta 4:56:31
td_layer11 iter 1057/1192  4.433B tokens  loss 2.315202  skipped=0 nan=0  eta 5:00:22
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## Durable Finalizer Added

Checked at 2026-05-25 23:11 UTC.

The existing local monitor only logs queue/training status; it does not collect
artifacts or regenerate the decision report after the final sidecars finish. I
added and started a separate home-side finalizer:

```text
service: codex-5b-td-finalizer-20260525.service
script:  /home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/finalize_when_ready.sh
log:     /home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/finalizer.log
```

It waits for the required iter `1192` JSON artifacts:

- Vanilla tokenizer-fair metrics
- TD tokenizer-fair metrics
- TD new-token diagnostics
- Vanilla packed downstream `results_*.json`
- TD packed downstream `results_*.json`

When all are present, it runs the local 5B collector and summary generator, then
attempts the trajectory plots. First poll confirmed the current state is still
pre-final: training jobs running, all final sidecars dependency-pending.

## 1192 Poll: Still Pre-Final at Iter 1056/1057

Checked at 2026-05-25 23:04 UTC.

Both final 1192 training jobs remain running. Sidecars are still dependency
pending. No restart, resubmission, or manual intervention was needed.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:39:15  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:39:15  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1057/1192  4.433B tokens  loss 1.605402  skipped=0 nan=0  eta 4:56:31
td_layer11 iter 1056/1192  4.429B tokens  loss 2.337900  skipped=0 nan=0  eta 5:02:25
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## 1192 Poll: Both Arms at Iter 1056

Checked at 2026-05-25 23:03 UTC.

Both final 1192 training jobs remain running. `sacct` agrees with `squeue`:
training jobs are `RUNNING`, and conversion / BPB / diagnostics / full-eval
sidecars are still `PENDING` on dependencies.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:38:22  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:38:22  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1056/1192  4.429B tokens  loss 1.604356  skipped=0 nan=0  eta 4:58:39
td_layer11 iter 1056/1192  4.429B tokens  loss 2.337900  skipped=0 nan=0  eta 5:02:25
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## Loss-Measurement Cleanup Follow-Up

Checked at 2026-05-26 local.

After the global doc/script cleanup, two remaining active-text issues were
patched:

- `TAKEOVER_LOG_20260521.md` now has a top-of-file note that historical `BPC`
  entries mean the byte-normalized `BPB` metric, and that raw `lm loss` entries
  are health telemetry only across different tokenizers.
- `cpt_plan.md` v0.5 changelog now says `BPB (then called BPC in some
  artifacts) / char-NLL`, so the historical note no longer sounds like a
  current metric-name decision.
- Historical Markdown tables under `eval/live_summaries/` now display `BPB`
  instead of `BPC`; their README preserves the compatibility note for legacy
  JSON keys such as `bpc_bits_per_byte`.

Validation rerun:

```text
python3 -m py_compile ... compute_tokenizer_fair_metrics.py summarize_training_logs.py summarize_bakeoff.py summarize_td_pilot_intrinsics.py summarize_3p5b_continuation.py summarize_5b_continuation.py plot_loss_comparison.py plot_intrinsic_van_td.py plot_training_loss.py
bash -n ... run_tokenizer_fair_metrics.sbatch submit_3p5b_eval_sidecars.sh submit_bakeoff_checkpoint_eval*.sh submit_td_checkpoint_eval.sh monitor_5b_td_vs_vanilla_status.sh
git diff --check
```

All three checks passed.

## 1192 Poll: Iter 1055/1056, Still Running

Checked at 2026-05-25 23:02 UTC.

Both final 1192 training jobs remain healthy. Sidecars are still
dependency-pending behind training, and no final 1192 eval artifacts exist yet.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:37:20  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:37:20  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1056/1192  4.429B tokens  loss 1.604356  skipped=0 nan=0  eta 4:58:39
td_layer11 iter 1055/1192  4.425B tokens  loss 2.324546  skipped=0 nan=0  eta 5:04:39
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## 1192 Poll: Both Arms at Iter 1054

Checked at 2026-05-25 22:59 UTC.

Both final 1192 training jobs remain running. Sidecars are still
dependency-pending and no final 1192 artifacts exist yet.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:34:29  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:34:29  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1054/1192  4.421B tokens  loss 1.626766  skipped=0 nan=0  eta 5:03:03
td_layer11 iter 1054/1192  4.421B tokens  loss 2.327818  skipped=0 nan=0  eta 5:06:47
```

Checkpoint status remains `latest=1040` for both arms.

## 1192 Poll: Still Pre-Final Checkpoint

Checked at 2026-05-25 22:58 UTC.

Direct Clariden poll confirms both final training jobs remain running and all
1192 sidecars remain dependency-pending.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:33:39  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:33:39  node nid006211
```

Latest visible training lines are still:

```text
vanilla    iter 1054/1192  4.421B tokens  loss 1.626766  skipped=0 nan=0  eta 5:03:03
td_layer11 iter 1053/1192  4.417B tokens  loss 2.340051  skipped=0 nan=0  eta 5:09:05
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

Monitor health checked on `home`: `codex-5b-td-monitor-20260525.service` is
active and sleeping between 600-second polls. The live monitor script was also
aligned with the repo copy so, if restarted, its job-name pattern accepts both
historical `bpc_*` and future `bpb_*` sidecar names.

## 1192 Poll: Iter 1053/1054, Still Running

Checked at 2026-05-25 22:57 UTC.

Both final 1192 training jobs remain healthy. No restart or intervention was
needed. Sidecars remain dependency-pending behind the training jobs.

```text
2382983  5b_vanilla_1192     RUNNING  elapsed 01:32:54  node nid006171
2382985  5b_td_layer11_1192  RUNNING  elapsed 01:32:54  node nid006211
```

Latest visible training lines:

```text
vanilla    iter 1054/1192  4.421B tokens  loss 1.626766  skipped=0 nan=0  eta 5:03:03
td_layer11 iter 1053/1192  4.417B tokens  loss 2.340051  skipped=0 nan=0  eta 5:09:05
```

Checkpoint status remains `latest=1040` for both arms. No final
`iter_0001192` eval files exist yet.

## 1192 Finalizer

Checked at 2026-05-26 05:05 UTC.

Finalizer detected the required iter `1192` JSON artifacts, copied them locally, regenerated the 5B summary JSON/Markdown, and attempted the trajectory plots. See `/home/foivos/runs/codex_monitors/5b_td_vs_vanilla_20260525/finalizer.log`.

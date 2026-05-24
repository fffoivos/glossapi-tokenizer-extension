# 3.5B continuation dry run — 2026-05-24

Dry-run tag: `continuation_3p5b_dryrun_20260524T020000Z`.

This is a non-submitting validation of the Vanilla/ReTok/TD layer11
continuation plan from iter 476 to iter 834 (~3.5B tokens total).

## What was validated

- Source checkpoints exist on Clariden:
  - Vanilla: `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_vanilla/checkpoints/iter_0000476`
  - ReTok: `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_retok/checkpoints/iter_0000476`
  - TD layer11: `/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/checkpoints/iter_0000476`
- Original bakeoff data prefixes exist and are forced in every training command:
  - base: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_megatron/bulk_mix_text_document`
  - extended: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_ext_megatron/bulk_mix_text_document`
- `LOSS_OBJECTIVE=ntp` is forced in all 9 training commands.
- Training plan contains 9 jobs: 3 arms x 3 segments.
- Eval plan contains 27 jobs: for each of 3 checkpoints, 3 conversions + 3 BPC jobs + 2 new-token diagnostics jobs + 1 packed downstream eval.
- Eval conversion jobs depend on the checkpoint-producing training segment.
- Later training segments do not depend on eval jobs.
- Eval jobs use `--nice=1000` by default.

## Non-submitting Slurm checks

Representative `sbatch --test-only` checks on Clariden returned schedulable
plans but did not submit jobs. A follow-up `squeue -j 2368756,2368757,2368759`
showed no queued jobs.

```text
sbatch test-only: training
sbatch: Job 2368756 to start at 15:40:12 a using 288 processors on nodes nid006102 in partition normal
sbatch test-only: convert
sbatch: Job 2368757 to start at 15:32:13 a using 288 processors on nodes nid007647 in partition debug
sbatch test-only: packed eval
sbatch: Job 2368759 to start at 15:40:13 a using 288 processors on nodes nid006102 in partition normal
```

## Files

| File | Meaning |
|---|---|
| `training_chain.tsv` | planned training segments and per-arm dependencies |
| `training_sbatch_commands.sh` | exact dry-run training `sbatch` commands |
| `eval_sidecar_jobs.tsv` | planned eval sidecars and dependencies |
| `eval_sbatch_commands.sh` | exact dry-run eval `sbatch` commands |

Live launch still requires:

```bash
DRY_RUN=0 CONFIRM_3P5B_LAUNCH=1 RUN_TAG=<real-run-tag> \
  bash submit_3p5b_continuation_chain.sh
```

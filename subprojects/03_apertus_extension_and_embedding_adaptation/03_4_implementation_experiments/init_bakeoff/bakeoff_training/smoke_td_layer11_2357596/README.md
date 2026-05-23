# TD Layer11 Megatron Smoke - Job 2357596

Purpose: prove the selected Token Distillation layer11 checkpoint can load in
Megatron and complete real training iterations with the extended tokenizer/data.

Result:

- job: `2357596`
- state: `COMPLETED`
- elapsed: `00:13:19`
- exit code: `0:0`
- node: `nid007017`
- output dir:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_td_full25_layer11_20260523T162614Z`
- init checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`

Iteration summary:

| Iter | Loss | Grad norm | Tokens/sec/GPU | Skipped | NaN |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.928761 | 153.403 | 7546.1 | 0 | 0 |
| 2 | 9.013231 | 101.992 | 7867.7 | 0 | 0 |
| 3 | 8.947750 | 105.898 | 7907.7 | 0 | 0 |
| 4 | 8.990582 | 106.622 | 7905.1 | 0 | 0 |
| 5 | 8.853868 | 173.612 | 7849.6 | 0 | 0 |

Health notes:

- The patched checkpoint loaded successfully at iteration 0.
- The extended tokenizer and extended Megatron data prefix were selected by
  `ARM=retok`.
- xIELU optimizer audit reported `missing=0` on all four ranks.
- No OOM, skipped iteration, or NaN iteration appeared.
- Rank 0/1 memory after iteration 1 peaked at `61378.95 MB` allocated and
  `61662.0 MB` reserved.

Checkpoint cleanup:

- Megatron saved at `EXIT_INTERVAL=5` even with `SAVE_INTERVAL=999999`.
- The generated smoke checkpoint was `138G`.
- `checkpoint_listing_before_cleanup.txt` records the checkpoint contents.
- The remote checkpoint directory was removed after evidence capture; the
  remaining remote smoke directory is `64K`.

Files:

- `bakeoff_smoke_td_l11-2357596.out`: Slurm stdout.
- `bakeoff_smoke_td_l11-2357596.err`: Slurm stderr.
- `run_metadata.json`: run metadata emitted by `bakeoff_train.sbatch`.
- `training_command.sh`: exact Megatron command.
- `slurm_status.txt`: final Slurm accounting.
- `checkpoint_listing_before_cleanup.txt`: checkpoint size/listing before
  cleanup.

# 16-node CXI timing for two-arm Greek CPT

Date: 2026-06-10. Cluster: Clariden. Account: `a0140`.

All tests used 16 nodes / 64 GPUs, `LAUNCH_MODE=torchrun`, AWS Libfabric/CXI,
and `NCCL_NET_FORCE_FLUSH=0`. No `NET/OFI ... NO_SPACE` failure reproduced.

## Transport gate

- Job `2515665`
- Output:
  `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_mockdata_cxi_noflush_20260610T183943Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:01:48`
- Runtime audit: `WORLD_SIZE=64`, `NCCL_NET=AWS Libfabric`,
  `NCCL_NET_FORCE_FLUSH=0`
- Iteration 1: `15623.3 ms`, `tokens/sec/gpu: 4194.8`

This validates the launch-scale CXI path. Socket/HSN should remain fallback
only.

## Real-data train timing

- Job `2515841`
- Output:
  `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_timing_20260610T191629Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:02:56`
- Setup: real base-tokenized CPT data, extra validation off, saves off,
  `EXIT_INTERVAL=10`
- Iteration 1: `15453.2 ms`
- Iterations 2-10: median `8559.8 ms`, mean `8629.9 ms`, range
  `8553.0-8869.4 ms`

Use iterations 2-10 for ETA. Iteration 1 includes warm-start effects and is
not representative of steady state.

## Validation overhead

- Job `2515891`
- Output:
  `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_eval_20260610T192032Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:02:57`
- Setup: real data, extra validation on, `EVAL_INTERVAL=1`, `EVAL_ITERS=1`,
  saves off, `EXIT_INTERVAL=1`
- Iteration 1 line: `21:30:44`
- Per-set validation lines printed for `hplt`, `openarchives`, `greek_phd`
- Exit line: `21:30:55`

Estimated three-set held-out validation event cost: about `11 s`.

## Checkpoint-save overhead

- Job `2515966`
- Output:
  `/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/smoke16_vanilla_realdata_cxi_noflush_save_20260610T193249Z`
- Result: `COMPLETED`, exit `0:0`, elapsed `00:02:05`
- Setup: real data, extra validation off, `SAVE_INTERVAL=1`, saves on,
  `EXIT_INTERVAL=1`
- Save timer: `save-checkpoint: 22174 ms`
- Checkpoint size from smoke: `136G`

Estimated checkpoint event cost: about `22 s`.

## ETA

Production run constants:

- `TRAIN_ITERS = 3218`
- `EVAL_INTERVAL = 25` -> `128` scheduled eval events
- `SAVE_INTERVAL = 119` -> `27` interval saves, possibly `28` if a final save
  is also emitted
- Recommended 16-node chain: `EXIT_INTERVAL=952`, `N_SEGMENTS=4`

Using mean steady-state train time:

- Training: `3218 * 8.6299 s = 7.71 h`
- Held-out validation: `128 * 11 s = 0.39 h`
- Checkpoint saves: `27 * 22.17 s = 0.17 h`
- Segment startup allowance: `4 * 95 s = 0.11 h`
- Total allocated runtime per arm: about `8.38 h`

Using median steady-state train time gives about `8.31 h`. Adding one final
checkpoint changes the estimate by only about `0.006 h`.

Operational estimate: **8.3-8.5 h per arm of allocated runtime**, excluding
Slurm queue wait and sidecar benchmark jobs. If both arms run concurrently, the
wall time is still about this number but requires 32 nodes / 128 GPUs at once.
If the arms run serially, double it.


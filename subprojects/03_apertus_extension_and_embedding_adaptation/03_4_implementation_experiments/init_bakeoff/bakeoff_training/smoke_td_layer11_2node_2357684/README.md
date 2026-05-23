# TD Layer11 Two-Node Smoke - Job 2357684

Purpose: test whether the TD layer11 arm should use two nodes for the 2B run.

Result:

- job: `2357684`
- state: `FAILED`
- elapsed: `00:02:24`
- exit code: `1:0`
- nodes: `nid006031`, `nid007615`
- launch mode: `torchrun`
- world size: `8`
- output dir:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/smoke_td_full25_layer11_2node_20260523T164424Z`

What passed:

- The selected R17-patched checkpoint loaded successfully.
- The extended tokenizer/data were selected through `ARM=retok`.
- xIELU optimizer audit passed on all eight ranks with `missing=0`.

Failure:

- The run failed before iteration 1 during a data-parallel collective.
- Key error from stderr:
  `NET/OFI Request ... Error: 16 (NO_SPACE)`
- PyTorch surfaced this as:
  `NCCL Error 2: unhandled system error`.

Decision:

- Do not use the two-node `torchrun` path for the TD 2B arm right now.
- Use the proven one-node path and chain a resume job because `normal` is capped
  at 12h while the one-node smoke estimates about 17.5h for 2B.

Files:

- `bakeoff_smoke_td_l11-2357684.out`: Slurm stdout.
- `bakeoff_smoke_td_l11-2357684.err`: Slurm stderr.
- `run_metadata.json`: run metadata emitted before failure.
- `training_command.sh`: exact Megatron command.
- `slurm_status.txt`: final Slurm accounting.

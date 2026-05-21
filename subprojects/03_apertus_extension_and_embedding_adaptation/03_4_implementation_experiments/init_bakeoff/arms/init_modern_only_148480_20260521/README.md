# Modern-only 148480 init checkpoint run - 2026-05-21

Small local audit copy for the Clariden init-checkpoint build/conversion run.
The actual HF and Megatron checkpoint weights remain on Clariden under:

```text
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/
```

Completed jobs:

- `2335382` - `build_init_ckpts`, completed `0:0` in `00:02:43`.
- `2335384` - `convert_init_ckpts`, completed `0:0` in `00:01:41`.

Megatron release checkpoints produced:

```text
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron/release
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok/megatron/release
/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/centroid/megatron/release
```

Files here:

- `init_build_summary.json` - build stats from `build_init_checkpoints.py`.
- `build_init_ckpts-2335382.out/.err` - build Slurm logs.
- `convert_init_ckpts-2335384.out/.err` - conversion Slurm logs.

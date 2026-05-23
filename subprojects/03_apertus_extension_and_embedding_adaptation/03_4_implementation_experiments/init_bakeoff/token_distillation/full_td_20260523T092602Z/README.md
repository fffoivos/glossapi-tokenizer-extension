# Full-Token TD Artifacts - 2026-05-23

Remote output root:

`/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z`

This local directory stores only small audit artifacts, not model shards.

Training job:

- Slurm job: `2353960`
- State: `COMPLETED`
- Elapsed: `05:45:53`
- Exit code: `0:0`

Arms:

| Arm | Target layer | Trained tokens | Skipped tokens | Elapsed |
|---|---:|---:|---:|---:|
| `last` | `-1` | 17,377 | 15 | 20,740.5s |
| `layer11` | `11` | 17,377 | 15 | 16,254.8s |

Preservation jobs:

| Arm | Job | State | Elapsed | Report |
|---|---:|---|---:|---|
| `last` | `2355706` | `COMPLETED` | `00:01:36` | `last/td_preservation_report.json` |
| `layer11` | `2355707` | `COMPLETED` | `00:01:39` | `layer11/td_preservation_report.json` |

Preservation result for both arms:

- No non-embedding tensor changed.
- No xIELU tensor changed.
- No QK-Norm tensor changed.
- No shape or dtype mismatches.
- All 17,377 trained `model.embed_tokens.weight` rows changed.
- All 17,377 trained `lm_head.weight` rows changed.
- All preserved embedding and output rows stayed byte-exact.

Next gate:

- Packed intrinsic eval job `2355714`.
- Remote output root:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_intrinsics_20260523T124000Z`

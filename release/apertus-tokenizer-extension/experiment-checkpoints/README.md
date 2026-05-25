# Experiment Checkpoints

This directory contains one public folder for each experiment checkpoint we care
about. The names intentionally omit run tags, tensor-parallel details, and exact
iteration numbers; those details are in each checkpoint manifest.

| Folder | Meaning |
|---|---|
| `TokenDistil-Init/` | Token Distillation initialization before CPT. |
| `TokenDistil-2B/` | Token Distillation after the 2B bakeoff. |
| `TokenDistil-3.5B/` | Selected Token Distillation checkpoint after continuation. |
| `Vanilla-2B/` | Original-tokenizer control after the 2B bakeoff. |
| `Vanilla-3.5B/` | Original-tokenizer control after continuation. |
| `ReTok-2B/` | ReTok baseline after the 2B bakeoff. |
| `ReTok-3.5B/` | ReTok baseline after continuation. |
| `Centroid-2B/` | Centroid baseline after the 2B bakeoff. |

Weights are uploaded to Hugging Face in these folders. The GitHub source repo
keeps only the metadata files.

Upload verification:

- Clariden Slurm job: `2382635`;
- partition: `xfer`;
- state: `COMPLETED`;
- exit code: `0:0`;
- log: `/users/fffoivos/apertus_hf_upload_checkpoints_20260525_2382635.log`.

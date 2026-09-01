# TD Layer11 R17 Roundtrip Evidence - Job 2357565

This directory is the local evidence copy for the selected full-token Token
Distillation checkpoint:

`retok_td_full25_layers_20260523T092602Z/layer11`

The large model artifacts remain on Clariden scratch:

- input HF checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/layer11`
- raw Megatron TP=2 checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_raw`
- R17-patched Megatron TP=2 checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`
- HF roundtrip checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip`

Slurm result:

- job: `2357565`
- partition: `debug`
- state: `COMPLETED`
- elapsed: `00:02:06`
- exit code: `0:0`

Verification summary from `verification.json`:

- standard tensors max abs diff: `0.0`
- R17 tensors max abs diff: `0.0`
- xIELU max abs diff: `0.0`
- QK-Norm max abs diff: `0.0`
- changed tensors over tolerance: `0`
- shape mismatches: `0`
- logit max abs diff across smoke prompts: `0.0`

Files:

- `td_r17_roundtrip_manifest.json`: Clariden path manifest.
- `verification.json`: tensor and logit roundtrip verifier output.
- `patch_apertus_extras.log`: xIELU/QK-Norm patcher log.
- `verify_hf_roundtrip.log`: verifier stdout.
- `hf_to_megatron.log`: HF -> Megatron conversion log.
- `megatron_to_hf.log`: Megatron -> HF conversion log.
- `megatron_commit.txt`: Megatron-LM commit used by the job.
- `td_r17_roundtrip-2357565.out`: Slurm stdout.
- `td_r17_roundtrip-2357565.err`: Slurm stderr, empty for this job.

# Vanilla R17 Roundtrip Evidence

Source job: `2341182`

Purpose: prove the selected Vanilla/base-tokenizer init path preserves Apertus
xIELU/QK-Norm extras through the accepted HF -> Megatron -> HF route.

Remote source:

- patched Megatron:
  `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`
- roundtrip HF:
  `/capstor/scratch/cscs/fffoivos/runs/r17_patch_roundtrip_vanilla_2341182/vanilla_hf_roundtrip_patched`
- verification:
  `/capstor/scratch/cscs/fffoivos/runs/r17_patch_roundtrip_vanilla_2341182/verification.json`

Result:

- `standard_max_abs_diff`: `0.0`
- `r17_max_abs_diff`: `0.0`
- `xielu_max_abs_diff`: `0.0`
- `qk_norm_max_abs_diff`: `0.0`
- smoke `logit_max_abs_diff`: `0.0`
- smoke prompt top IDs match for all prompts.

This is the production init evidence for the selected 15-20B Vanilla CPT path.

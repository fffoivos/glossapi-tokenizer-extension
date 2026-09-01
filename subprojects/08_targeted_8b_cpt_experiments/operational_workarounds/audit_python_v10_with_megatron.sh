#!/usr/bin/env bash
set -euo pipefail

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
RUN="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
Q="$RUN/qualification/32fd635f36bea663b3a63a8eabb71d013e7a587b533387d92f31481ba48dccd9/promotion_checkpoint_compatible_v105"
RUNNER=/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260818T211000Z-6b7796a-sequence-range
V10=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260821T133000Z-hard-h2g-compatible-checkpoint-permit-v10
MEGA="$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2"
COMPAT="$V10/subprojects/08_targeted_8b_cpt_experiments/runtime_compat"

exec "$RUNNER/bin/apertus-campaign-uenv-exec" "$Q/manifest-proven.json" \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$COMPAT:$MEGA" python3 "$@"

#!/usr/bin/env bash
#SBATCH --job-name=full8b-ckpt-gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_CHECKPOINT_ITERATION:?set exact boundary}"
: "${FULL8_SELECTED_PROFILE:?set frozen selected execution profile receipt}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/train/freeze_checkpoint.py" \
  --checkpoint-root "$FULL8_RUN_ROOT/checkpoints" \
  --iteration "$FULL8_CHECKPOINT_ITERATION" \
  --recipe "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json" \
  --output "$FULL8_RUN_ROOT/checkpoint_receipts/iter_$(printf '%07d' "$FULL8_CHECKPOINT_ITERATION").json"

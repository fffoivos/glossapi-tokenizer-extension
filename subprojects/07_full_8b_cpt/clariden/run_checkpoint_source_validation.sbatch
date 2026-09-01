#!/usr/bin/env bash
#SBATCH --job-name=full8b-source-checkpoint
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --switches=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set training-disjoint validation stage}"
: "${FULL8_SOURCE_TRAIN_RUN:?set production training run root}"
: "${FULL8_SOURCE_ITERATION:?set exact saved checkpoint iteration}"
: "${FULL8_CHECKPOINT_VALIDATION_OUTPUT:?set unique validation output root}"
[[ "$FULL8_SOURCE_ITERATION" =~ ^[1-9][0-9]*$ ]] || { echo "invalid source iteration" >&2; exit 2; }
[[ -f "$FULL8_SOURCE_TRAIN_RUN/checkpoints/iter_$(printf '%07d' "$FULL8_SOURCE_ITERATION")/.metadata" ]] || {
  echo "source checkpoint is incomplete" >&2; exit 2;
}
export FULL8_RUN_ROOT="$FULL8_CHECKPOINT_VALIDATION_OUTPUT/runtime"
export FULL8_LOAD_CHECKPOINT="$FULL8_SOURCE_TRAIN_RUN/checkpoints"
export FULL8_START_ITERATION="$FULL8_SOURCE_ITERATION"
export FULL8_END_ITERATION="$((FULL8_SOURCE_ITERATION + 1))"
export FULL8_EXACT_LOAD_ITERATION="$FULL8_SOURCE_ITERATION"
export FULL8_CHECKPOINT_VALIDATION_ONLY=1
export FULL8_CHECKPOINT_VALIDATION_OUTPUT
exec bash "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"

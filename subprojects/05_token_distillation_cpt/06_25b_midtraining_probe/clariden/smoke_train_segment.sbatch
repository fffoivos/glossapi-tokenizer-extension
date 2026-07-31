#!/usr/bin/env bash
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=01:30:00

set -euo pipefail
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${START_ITERATION:?set START_ITERATION}"
: "${END_ITERATION:?set END_ITERATION}"
: "${CPT_PHASE:?set CPT_PHASE}"
: "${INIT_CKPT:?set INIT_CKPT}"
: "${TRAIN_SCRIPT:?set TRAIN_SCRIPT}"
: "${PROBE_ROOT:?set PROBE_ROOT}"
: "${SEGMENT_OUTPUT_DIR:?set SEGMENT_OUTPUT_DIR}"
: "${SMOKE_CHECKPOINT_RECEIPT:?set SMOKE_CHECKPOINT_RECEIPT}"
[[ "${CPT_SMOKE:-}" == "1" ]] || { echo "ERROR: CPT_SMOKE must be exactly 1" >&2; exit 2; }

preflight=(python3 "$PROBE_ROOT/clariden/preflight_segment.py"
  --smoke --assets "$TRAINING_ASSETS_RECEIPT"
  --start "$START_ITERATION" --end "$END_ITERATION" --phase "$CPT_PHASE"
  --load-checkpoint "$INIT_CKPT")
if [[ -n "${RESUME_CHECKPOINT_RECEIPT:-}" ]]; then
  preflight+=(--resume-receipt "$RESUME_CHECKPOINT_RECEIPT")
fi
uenv run pytorch/v2.9.1:v2 --view=default -- "${preflight[@]}"

export TRAIN_CONFIG_OVERRIDE="$PROBE_ROOT/train/smoke_phase_config.env"
export CPT_PHASE START_ITERATION END_ITERATION
export OUTPUT_DIR="$SEGMENT_OUTPUT_DIR"
export EXIT_INTERVAL="$END_ITERATION"
export RESUME_TRAINING=0
if (( START_ITERATION > 0 )); then
  export RESUME_TRAINING=1
fi

bash "$TRAIN_SCRIPT"

uenv run pytorch/v2.9.1:v2 --view=default -- \
  python3 "$PROBE_ROOT/train/freeze_resume_checkpoint.py" \
    --smoke \
    --checkpoint-dir "$SEGMENT_OUTPUT_DIR/checkpoints" \
    --iteration "$END_ITERATION" \
    --training-assets-receipt "$TRAINING_ASSETS_RECEIPT" \
    --output "$SMOKE_CHECKPOINT_RECEIPT"

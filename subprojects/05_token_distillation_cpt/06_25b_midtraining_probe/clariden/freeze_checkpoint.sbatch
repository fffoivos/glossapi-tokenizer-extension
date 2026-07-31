#!/usr/bin/env bash
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
: "${PROBE_ROOT:?set PROBE_ROOT}"
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
: "${CHECKPOINT_ITERATION:?set CHECKPOINT_ITERATION}"
: "${CHECKPOINT_RECEIPT:?set CHECKPOINT_RECEIPT}"

python3 "$PROBE_ROOT/train/freeze_resume_checkpoint.py" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --iteration "$CHECKPOINT_ITERATION" \
  --training-assets-receipt "$TRAINING_ASSETS_RECEIPT" \
  --output "$CHECKPOINT_RECEIPT"

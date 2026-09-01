#!/usr/bin/env bash
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
python3 "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}/train/freeze_resume_checkpoint.py" \
  --checkpoint-dir "${LR13_CHECKPOINT_DIR:?set LR13_CHECKPOINT_DIR}" \
  --iteration "${LR13_CHECKPOINT_ITERATION:?set LR13_CHECKPOINT_ITERATION}" \
  --training-assets-receipt "${LR13_ASSETS_RECEIPT:?set LR13_ASSETS_RECEIPT}" \
  --output "${LR13_CHECKPOINT_RECEIPT:?set LR13_CHECKPOINT_RECEIPT}"


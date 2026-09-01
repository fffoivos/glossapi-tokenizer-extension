#!/usr/bin/env bash
#SBATCH --job-name=lr13_freeze_data
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}/clariden/paths.env"
mkdir -p "$LR13_DATA_ROOT" "$LR13_RUN_ROOT"
PYTHON_BIN="${LR13_CPU_PYTHON:-python3.11}"
command -v "$PYTHON_BIN" >/dev/null || {
  echo "ERROR: required CPU Python is unavailable: $PYTHON_BIN" >&2
  exit 2
}
"$PYTHON_BIN" "$LR13_CODE_ROOT/scripts_validate_recipe.py" --recipe "$LR13_RECIPE"
"$PYTHON_BIN" "$LR13_CODE_ROOT/dataset/freeze_derived_schedule.py" \
  --recipe "$LR13_RECIPE" --output-root "$LR13_DATA_ROOT"
"$PYTHON_BIN" "$LR13_CODE_ROOT/dataset/freeze_training_assets.py" \
  --recipe "$LR13_RECIPE" \
  --dataset-manifest "$LR13_DATASET_MANIFEST" \
  --training-data-env "$LR13_TRAINING_DATA_ENV" \
  --code-root "$LR13_CODE_ROOT" \
  --output "$LR13_ASSETS_RECEIPT"

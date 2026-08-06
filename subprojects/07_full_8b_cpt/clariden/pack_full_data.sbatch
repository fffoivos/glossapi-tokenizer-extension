#!/usr/bin/env bash
#SBATCH --job-name=full8b-pack
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-511%128
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_STAGE_ROOT:?set immutable data stage root}"
"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/dataset/pack_catalog_bucket.py" \
  --packing-plan "$FULL8_STAGE_ROOT/inventory/packing_plan.json" \
  --input-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID"

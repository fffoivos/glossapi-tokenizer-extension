#!/usr/bin/env bash
#SBATCH --job-name=full8-anon-freeze
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
DATA_PYTHON=${FULL8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
DATA_RUNTIME_RECEIPT=${FULL8_DATA_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/runtime_receipt.json}
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set new anonymized derivative root}"
: "${FULL8_PARENT_SOURCE_ROOT:?set frozen parent binary stage}"
: "${FULL8_VALIDATION_MANIFEST:?set corrected 13-panel validation manifest}"
test -s "$DATA_RUNTIME_RECEIPT"

"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific

uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/anonymization/freeze_anonymization_overlay.py" \
  --parent-input-receipt "$FULL8_PARENT_SOURCE_ROOT/input_receipt.json" \
  --parent-heldout-manifest "$FULL8_PARENT_SOURCE_ROOT/heldouts/heldout_manifest.json" \
  --validation-manifest "$FULL8_VALIDATION_MANIFEST" \
  --stage-root "$FULL8_ANON_STAGE_ROOT"

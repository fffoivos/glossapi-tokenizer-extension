#!/usr/bin/env bash
#SBATCH --job-name=full8-postmask-dedup
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
DATA_PYTHON=${FULL8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
DATA_RUNTIME_RECEIPT=${FULL8_DATA_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/runtime_receipt.json}
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set anonymized derivative root}"
test -s "$DATA_RUNTIME_RECEIPT"
"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
mkdir -p "$FULL8_ANON_STAGE_ROOT/sort-tmp"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/anonymization/finalize_postmask_dedup.py" \
  --overlay "$FULL8_ANON_STAGE_ROOT/anonymization_overlay.json" \
  --stage-root "$FULL8_ANON_STAGE_ROOT" \
  --sort-temp "$FULL8_ANON_STAGE_ROOT/sort-tmp" --sort-parallel 192 --sort-memory 320G

#!/usr/bin/env bash
#SBATCH --job-name=full8b-sync-parity-gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen sanitized stage}"
: "${FULL8_PARITY_ROOT:?set parity smoke root}"
: "${FULL8_PROFILES:?set derived sanitized profiles}"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_checkpoint_parity_smoke.py" \
  --profiles "$FULL8_PROFILES" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --control-root "$FULL8_PARITY_ROOT/control_dp32" \
  --control-repeat-root "$FULL8_PARITY_ROOT/control_dp32_repeat" \
  --output "$FULL8_PARITY_ROOT/checkpoint_parity_receipt.json"

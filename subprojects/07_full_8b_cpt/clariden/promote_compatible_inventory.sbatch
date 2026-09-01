#!/usr/bin/env bash
#SBATCH --job-name=full8-anon-reuse
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set new anonymization stage}"
: "${FULL8_REUSE_INVENTORY_ROOT:?set prior completed inventory stage}"
"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
"$HOST_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/anonymization/promote_compatible_inventory.py" \
  --overlay "$FULL8_ANON_STAGE_ROOT/anonymization_overlay.json" \
  --prior-stage-root "$FULL8_REUSE_INVENTORY_ROOT" \
  --stage-root "$FULL8_ANON_STAGE_ROOT" \
  --output "$FULL8_ANON_STAGE_ROOT/compatible_inventory_promotion.json"

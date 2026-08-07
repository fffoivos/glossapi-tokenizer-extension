#!/usr/bin/env bash
#SBATCH --job-name=full8b-stop-smoke-gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_SMOKE_RUN_ROOT:?set smoke root}"
: "${FULL8_INTERRUPTED_ROOT:?set interrupted attempt root}"
: "${FULL8_RESTART_ROOT:?set restart attempt root}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_graceful_stop_smoke.py" \
  --run-root "$FULL8_SMOKE_RUN_ROOT" --interrupted-root "$FULL8_INTERRUPTED_ROOT" \
  --restart-root "$FULL8_RESTART_ROOT" --output "$FULL8_SMOKE_RUN_ROOT/graceful_stop_smoke_receipt.json"

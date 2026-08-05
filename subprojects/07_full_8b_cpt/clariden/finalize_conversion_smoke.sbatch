#!/usr/bin/env bash
#SBATCH --job-name=full8b-convert-smoke-gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_SELECTED_PROFILE:?set selected profile receipt}"
: "${FULL8_CONVERSION_SMOKE_ROOT:?set conversion smoke root}"
python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_conversion_smoke.py" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --export-receipt "$FULL8_CONVERSION_SMOKE_ROOT/export/checkpoint_eval_export_receipt.json" \
  --greekmmlu-receipt "$FULL8_CONVERSION_SMOKE_ROOT/exact_checkpoint_native_greekmmlu_receipt.json" \
  --output "$FULL8_CONVERSION_SMOKE_ROOT/conversion_greekmmlu_smoke_receipt.json"

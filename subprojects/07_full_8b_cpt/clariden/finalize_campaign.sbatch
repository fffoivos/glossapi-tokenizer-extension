#!/usr/bin/env bash
#SBATCH --job-name=full8b-evidence-finalizer
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_SELECTED_PROFILE:?set selected profile}"
: "${FULL8_TRAINING_RECEIPT:?set training receipt}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
exec python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_campaign.py" \
  --run-root "$FULL8_RUN_ROOT" --prelaunch-root "$FULL8_PRELAUNCH_ROOT" \
  --recipe "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json" \
  --selected-profile "$FULL8_SELECTED_PROFILE" --training-receipt "$FULL8_TRAINING_RECEIPT" \
  --output "$FULL8_RUN_ROOT/campaign_evidence_completion_receipt.json"

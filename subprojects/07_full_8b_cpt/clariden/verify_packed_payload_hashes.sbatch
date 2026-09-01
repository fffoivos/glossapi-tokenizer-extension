#!/usr/bin/env bash
#SBATCH --job-name=full8b-packed-sha
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set corrected stage root}"
: "${FULL8_PACKED_INTEGRITY_OUTPUT:?set new output receipt}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/verify_packed_payload_hashes.py" \
  --packed-receipt "$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json" \
  --workers 32 --output "$FULL8_PACKED_INTEGRITY_OUTPUT"

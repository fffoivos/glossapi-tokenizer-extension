#!/usr/bin/env bash
#SBATCH --job-name=full8b-all-panel-val
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code bundle}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_PARENT_STAGE:?set corrected old-Greek stage}"
: "${FULL8_CORRECTED_STAGE:?set all-panel-disjoint stage}"
tokenizer=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992
content_receipt="$FULL8_PARENT_STAGE/evidence/selected_training_content/selected_training_content_receipt.json"
old_greek_receipt="$FULL8_PARENT_STAGE/validation/clean_replay_retention/clean_replay_validation_manifest.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/prepare_corrected_stage_overlay.py" \
  --parent-stage "$FULL8_PARENT_STAGE" --output-stage "$FULL8_CORRECTED_STAGE"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/build_training_disjoint_validation_manifest.py" \
  --input-manifest "$FULL8_PARENT_STAGE/validation/validation_manifest.json" \
  --training-content-receipt "$content_receipt" --tokenizer-root "$tokenizer" \
  --output-dir "$FULL8_CORRECTED_STAGE/validation/training_disjoint" \
  --output-manifest "$FULL8_CORRECTED_STAGE/validation/validation_manifest.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/finalize_corrected_stage_overlay.py" \
  --parent-stage "$FULL8_PARENT_STAGE" --stage-root "$FULL8_CORRECTED_STAGE" \
  --clean-panel-manifest "$old_greek_receipt" \
  --training-content-receipt "$content_receipt" \
  --all-panel-audit "$FULL8_CORRECTED_STAGE/validation/training_disjoint/all_panel_training_content_audit.json" \
  --output "$FULL8_CORRECTED_STAGE/data_stage_receipt.json"

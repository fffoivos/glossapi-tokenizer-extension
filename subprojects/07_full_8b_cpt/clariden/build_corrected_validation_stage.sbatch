#!/usr/bin/env bash
#SBATCH --job-name=full8b-clean-replay-val
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code bundle}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_PARENT_STAGE:?set frozen v1 stage}"
: "${FULL8_CORRECTED_STAGE:?set new corrected stage}"
: "${FULL8_VALIDATION_PYTHON:?set Python with numpy, pyarrow and tokenizers}"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/prepare_corrected_stage_overlay.py" \
  --parent-stage "$FULL8_PARENT_STAGE" --output-stage "$FULL8_CORRECTED_STAGE"

content_root="$FULL8_CORRECTED_STAGE/evidence/selected_training_content"
uenv run pytorch/v2.9.1:v2 --view=default -- "$FULL8_VALIDATION_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/freeze_selected_training_content.py" \
  --pool-receipt "$FULL8_CORRECTED_STAGE/inventory/pool_corpus_receipt.json" \
  --stage-root "$FULL8_PARENT_STAGE" --output-dir "$content_root"

panel_root="$FULL8_CORRECTED_STAGE/validation/clean_replay_retention"
uenv run pytorch/v2.9.1:v2 --view=default -- "$FULL8_VALIDATION_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/build_clean_replay_validation.py" \
  --pool-receipt "$FULL8_CORRECTED_STAGE/inventory/pool_corpus_receipt.json" \
  --stage-root "$FULL8_PARENT_STAGE" \
  --training-content-receipt "$content_root/selected_training_content_receipt.json" \
  --source-parquet /iopsstor/scratch/cscs/fffoivos/cpt_corpus/greek_replay/greek_replay.parquet \
  --tokenizer-root /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992 \
  --original-panel /capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1/heldouts/sets/val_forget_old_greek.jsonl \
  --output-dir "$panel_root" --target-tokens 10000000

neutral_manifest="$FULL8_PARENT_STAGE/validation/neutral_external_modern_greek.manifest.json"
neutral_receipt=$(python3 - "$neutral_manifest" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["corpus_receipt"]["path"])
PY
)
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/freeze_validation_manifest.py" \
  --pool-receipt "$FULL8_CORRECTED_STAGE/inventory/pool_corpus_receipt.json" \
  --neutral-manifest "$neutral_manifest" --neutral-corpus-receipt "$neutral_receipt" \
  --replacement-old-greek-manifest "$panel_root/clean_replay_validation_manifest.json" \
  --output "$FULL8_CORRECTED_STAGE/validation/validation_manifest.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/finalize_corrected_stage_overlay.py" \
  --parent-stage "$FULL8_PARENT_STAGE" --stage-root "$FULL8_CORRECTED_STAGE" \
  --clean-panel-manifest "$panel_root/clean_replay_validation_manifest.json" \
  --training-content-receipt "$content_root/selected_training_content_receipt.json" \
  --output "$FULL8_CORRECTED_STAGE/data_stage_receipt.json"

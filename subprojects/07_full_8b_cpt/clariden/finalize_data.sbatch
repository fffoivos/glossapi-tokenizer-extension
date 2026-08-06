#!/usr/bin/env bash
#SBATCH --job-name=full8b-data-final
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set immutable data stage root}"
: "${FULL8_VALIDATION_PYTHON:?set Python with numpy, pyarrow and tokenizers}"
DATASET_CODE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/dataset"
NEUTRAL_CORPUS=${FULL8_NEUTRAL_CORPUS_RECEIPT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/20260802T221000Z-neutral-gpp-v4/heldout-corrected-v1/neutral_external_corpus.json}
TOKENIZER_ROOT=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$DATASET_CODE/finalize_packed_corpus.py" \
  --packing-plan "$FULL8_STAGE_ROOT/inventory/packing_plan.json" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --output "$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$DATASET_CODE/build_five_schedules.py" \
  --packed-receipt "$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json" \
  --output-dir "$FULL8_STAGE_ROOT/schedules" \
  --global-batch-sequences 1024

mkdir -p "$FULL8_STAGE_ROOT/contracts"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/derive_sanitized_contracts.py" \
  --base-recipe "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json" \
  --base-profiles "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/execution_profiles.json" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --recipe-output "$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json" \
  --profiles-output "$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json"

content_root="$FULL8_STAGE_ROOT/evidence/selected_training_content"
uenv run pytorch/v2.9.1:v2 --view=default -- "$FULL8_VALIDATION_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/freeze_selected_training_content.py" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --stage-root "$FULL8_STAGE_ROOT" --output-dir "$content_root"

original_old_greek=$(uenv run pytorch/v2.9.1:v2 --view=default -- python3 - \
  "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(next(row["raw_jsonl"]["path"] for row in x["heldouts"] if row["name"]=="old_greek"))
PY
)
panel_root="$FULL8_STAGE_ROOT/validation/clean_replay_retention"
uenv run pytorch/v2.9.1:v2 --view=default -- "$FULL8_VALIDATION_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/build_clean_replay_validation.py" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --training-content-receipt "$content_root/selected_training_content_receipt.json" \
  --source-parquet /iopsstor/scratch/cscs/fffoivos/cpt_corpus/greek_replay/greek_replay.parquet \
  --tokenizer-root "$TOKENIZER_ROOT" --original-panel "$original_old_greek" \
  --output-dir "$panel_root" --target-tokens 10000000

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/evaluation/build_neutral_external_binary.py" \
  --corpus-receipt "$NEUTRAL_CORPUS" --tokenizer-dir "$TOKENIZER_ROOT" \
  --output-prefix "$FULL8_STAGE_ROOT/validation/neutral_external_modern_greek"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/freeze_validation_manifest.py" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --neutral-manifest "$FULL8_STAGE_ROOT/validation/neutral_external_modern_greek.manifest.json" \
  --neutral-corpus-receipt "$NEUTRAL_CORPUS" \
  --replacement-old-greek-manifest "$panel_root/clean_replay_validation_manifest.json" \
  --output "$FULL8_STAGE_ROOT/validation/validation_manifest.pre_content_audit.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/build_training_disjoint_validation_manifest.py" \
  --input-manifest "$FULL8_STAGE_ROOT/validation/validation_manifest.pre_content_audit.json" \
  --training-content-receipt "$content_root/selected_training_content_receipt.json" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --output-dir "$FULL8_STAGE_ROOT/validation/training_disjoint" \
  --output-manifest "$FULL8_STAGE_ROOT/validation/validation_manifest.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/initialization/build_token_byte_lengths.py" \
  --tokenizer-dir "$TOKENIZER_ROOT" \
  --output-npy "$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.npy" \
  --output-receipt "$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.receipt.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json"

#!/usr/bin/env bash
#SBATCH --job-name=t8b-freeze-train
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_EXPERIMENT:?set A or B}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
PARENT_STAGE=${TARGET8_PARENT_STAGE_ROOT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3}
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "training-asset freeze must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$TARGET8_STAGE_ROOT/experiment_$name"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_training_assets.py" \
  --experiment "$TARGET8_EXPERIMENT" --base-recipe "$PARENT_STAGE/contracts/recipe_8b_full_mixed.sanitized.json" \
  --pool-receipt "$stage/inventory/pool_corpus_receipt.json" --schedule-manifest "$stage/schedules/schedule_manifest.json" \
  --validation-manifest "$PARENT_STAGE/validation/validation_manifest.json" \
  --token-byte-lengths "$PARENT_STAGE/validation/token_utf8_byte_lengths.npy" --stage-root "$stage" \
  --receipt "$stage/contracts/training_assets_receipt.json"

uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$stage/contracts/recipe_8b_full_mixed.sanitized.json" \
  --pool-receipt "$stage/inventory/pool_corpus_receipt.json" --schedule-manifest "$stage/schedules/schedule_manifest.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_execution_profile.py" \
  --recipe "$stage/contracts/recipe_8b_full_mixed.sanitized.json" \
  --profiles "$stage/contracts/execution_profiles.sanitized.json" --profile-id dp32_16node \
  --output "$stage/contracts/selected_execution_profile.json"

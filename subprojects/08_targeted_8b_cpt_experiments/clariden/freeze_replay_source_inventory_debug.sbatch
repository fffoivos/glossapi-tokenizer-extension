#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-assets
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "replay source inventory requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

bulk="$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build/bulk_13b.json"
maker="$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/03_training_experiments/curriculum_sweeps_v2/dataset/make_phase_recipes.py"
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
recipe_root="$H2G_STAGE_ROOT/configs/historical_phase_recipes"
derived_recipe="$H2G_STAGE_ROOT/configs/replay_only_reused_validation.json"
validation_receipt="$H2G_STAGE_ROOT/receipts/reused_validation_panels.json"
acquisition_receipt="/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/20260731T124000Z-cpt25b-v1/receipts/replay_acquisition_receipt.json"
inventory_receipt="$H2G_STAGE_ROOT/receipts/replay_source_inventory.json"
[[ -s "$validation_receipt" ]] || { echo "reused validation receipt missing: $validation_receipt" >&2; exit 2; }
[[ -s "$acquisition_receipt" ]] || { echo "replay acquisition receipt missing: $acquisition_receipt" >&2; exit 2; }
for path in "$recipe_root" "$derived_recipe" "$inventory_receipt"; do
  [[ ! -e "$path" ]] || { echo "immutable replay-inventory output exists: $path" >&2; exit 2; }
done
temporary_root="$H2G_STAGE_ROOT/tmp/replay-source-inventory-$SLURM_JOB_ID"
[[ ! -e "$temporary_root" ]] || { echo "temporary replay-inventory root exists: $temporary_root" >&2; exit 2; }
temporary_recipes="$temporary_root/historical_phase_recipes"
temporary_derived="$temporary_root/replay_only_reused_validation.json"
temporary_receipt="$temporary_root/replay_source_inventory.json"
mkdir -p "$H2G_STAGE_ROOT/configs" "$H2G_STAGE_ROOT/receipts" "$temporary_root"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$maker" "$bulk" "$temporary_recipes" "$H2G_STAGE_ROOT"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/derive_replay_recipe_for_reused_panels.py" \
    --historical-recipe "$temporary_recipes/replay_only.json" \
    --published-historical-recipe "$recipe_root/replay_only.json" \
    --validation-receipt "$validation_receipt" \
    --replay-acquisition-receipt "$acquisition_receipt" \
    --output "$temporary_derived"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/freeze_replay_source_inventory.py" \
    --recipe "$temporary_derived" \
    --published-recipe "$derived_recipe" \
    --bulk-recipe "$bulk" \
    --replay-acquisition-receipt "$acquisition_receipt" \
    --output "$temporary_receipt" \
    --workers 16

# Publish the authority receipt last. Downstream jobs treat its absence as a
# hard failure, so an interrupted preflight cannot make partial recipes valid.
mv "$temporary_recipes" "$recipe_root"
mv "$temporary_derived" "$derived_recipe"
mv "$temporary_receipt" "$inventory_receipt"
rmdir "$temporary_root"

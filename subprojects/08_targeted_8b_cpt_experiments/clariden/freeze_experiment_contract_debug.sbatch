#!/usr/bin/env bash
#SBATCH --job-name=t8b-freeze-contract
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_EXPERIMENT:?set A or B}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "contract freeze must run on debug" >&2; exit 2; }
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

subproject="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$TARGET8_STAGE_ROOT/experiment_$name"
args=()
if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  args+=(
    --poly-token-receipt "$stage/poly_release_token_receipt.json"
    --a-decontamination-receipt "$stage/receipts/decontamination_summary.json"
    --a-selected-pools-receipt "$stage/receipts/selected_pools_receipt.json"
  )
else
  args+=(--b-schedule-manifest "$stage/schedules/schedule_manifest.json")
fi
mkdir -p "$stage/contracts"
/usr/bin/python3.11 "$subproject/scripts/freeze_experiment_contract.py" \
  --experiment-a-config "$subproject/configs/experiment_a_recipe.json" \
  --experiment-b-config "$subproject/configs/experiment_b_recipe.json" \
  --allocation-config "$subproject/configs/allocation_plan.json" \
  --mode launch --experiment "$TARGET8_EXPERIMENT" "${args[@]}" \
  --output "$stage/contracts/frozen_experiment_contract.json"

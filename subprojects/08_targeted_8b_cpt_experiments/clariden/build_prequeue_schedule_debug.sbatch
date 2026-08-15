#!/usr/bin/env bash
#SBATCH --job-name=t8b-freeze-prequeue
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_EXPERIMENT:?set A or B}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "prequeue schedule freeze must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$TARGET8_STAGE_ROOT/experiment_$name"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_prequeue_schedule.py" \
  --experiment "$TARGET8_EXPERIMENT" --recipe "$stage/contracts/recipe_8b_full_mixed.sanitized.json" \
  --allocation-plan "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/allocation_plan.json" \
  --output "$stage/contracts/prequeue_schedule.json"

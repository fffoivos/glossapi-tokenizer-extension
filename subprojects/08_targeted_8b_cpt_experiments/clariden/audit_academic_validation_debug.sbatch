#!/usr/bin/env bash
#SBATCH --job-name=t8b-audit-val
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=350G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "validation audit must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
root="$TARGET8_STAGE_ROOT/experiment_a/academic_training_eligible"
[[ -s "$root/exclusion_manifest.json" ]] || { echo "validation exclusion manifest missing" >&2; exit 2; }
[[ ! -e "$root/audit_receipt.json" ]] || { echo "immutable validation audit receipt exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/audit_validation_exclusion.py" \
  --exclusion-manifest "$root/exclusion_manifest.json" \
  --receipt "$root/audit_receipt.json"

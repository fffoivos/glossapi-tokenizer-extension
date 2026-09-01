#!/usr/bin/env bash
#SBATCH --job-name=t8b-find-renamed-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "renamed artifact discovery must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

output=${TARGET8_POLY_RENAMED_SEARCH_OUTPUT:-$TARGET8_STAGE_ROOT/prelaunch/exact_poly_artifact_content_search_v3.json}
[[ ! -e "$output" ]] || { echo "immutable renamed-artifact receipt exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/search_renamed_poly_artifact.py" \
  --root /users/fffoivos \
  --root /iopsstor/scratch/cscs/fffoivos \
  --root /capstor/scratch/cscs/fffoivos \
  --output "$output"

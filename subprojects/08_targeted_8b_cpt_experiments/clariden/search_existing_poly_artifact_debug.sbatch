#!/usr/bin/env bash
#SBATCH --job-name=t8b-find-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/usr/bin/python3.11}

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "artifact discovery must run on debug" >&2
  exit 2
}

output=${TARGET8_POLY_SEARCH_OUTPUT:-$TARGET8_STAGE_ROOT/prelaunch/exact_poly_artifact_search_v1.json}
[[ ! -e "$output" ]] || { echo "immutable search receipt already exists" >&2; exit 2; }

"$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/search_existing_poly_artifact.py" \
  --root /users/fffoivos \
  --root /iopsstor/scratch/cscs/fffoivos \
  --root /capstor/scratch/cscs/fffoivos \
  --output "$output"

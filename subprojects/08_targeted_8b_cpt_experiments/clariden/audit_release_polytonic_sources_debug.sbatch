#!/usr/bin/env bash
#SBATCH --job-name=t8b-audit-release-poly
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_RELEASE_ROOT:?set anonymized release root}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "release polytonic audit must run on debug" >&2; exit 2; }

/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
config="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/experiment_a_recipe.json"
output="$TARGET8_STAGE_ROOT/experiment_a/polytonic_release_source_audit.json"
[[ ! -e "$output" ]] || { echo "immutable release-polytonic source audit exists" >&2; exit 2; }
mapfile -t sources < <(jq -r '.modern_data.polytonic.source_datasets[]' "$config")
(( ${#sources[@]} > 0 )) || { echo "no release polytonic sources configured" >&2; exit 2; }
args=()
for source in "${sources[@]}"; do args+=(--source "$source"); done
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/audit_release_polytonic_sources.py" \
  --release-root "$TARGET8_RELEASE_ROOT" "${args[@]}" --workers 32 --output "$output"

#!/usr/bin/env bash
#SBATCH --job-name=t8b-extract-hplt
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
: "${TARGET8_RELEASE_ROOT:?set anonymized release root}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "HPLT selection must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
output="$TARGET8_STAGE_ROOT/experiment_a/hplt_candidates_pre_decontamination"
[[ ! -e "$output" ]] || { echo "immutable HPLT candidate output exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/extract_release_sources.py" \
  --release-root "$TARGET8_RELEASE_ROOT" \
  --source HPLT/ell_Grek_ge8_no_mt_clean60 \
  --fraction-numerator 1 --fraction-denominator 4 --seed 20260811 \
  --output-root "$output" --workers 32

#!/usr/bin/env bash
#SBATCH --job-name=t8b-exclude-hval
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
VALIDATION_MANIFEST=${TARGET8_VALIDATION_MANIFEST:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3/validation/validation_manifest.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "HPLT validation exclusion must run on debug" >&2; exit 2; }

/usr/bin/python3.11 \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
decontam="$TARGET8_STAGE_ROOT/experiment_a/hplt_candidates_decontaminated"
output="$TARGET8_STAGE_ROOT/experiment_a/hplt_candidates_training_eligible"
[[ -s "$decontam/audit_receipt.json" ]] || { echo "HPLT decontamination audit missing" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "immutable HPLT eligibility output exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/exclude_frozen_validation_content.py" \
  --input "$decontam/data" --validation-manifest "$VALIDATION_MANIFEST" \
  --output-root "$output" --workers 32

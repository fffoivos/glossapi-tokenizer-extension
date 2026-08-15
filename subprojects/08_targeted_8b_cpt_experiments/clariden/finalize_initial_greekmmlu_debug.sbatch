#!/usr/bin/env bash
#SBATCH --job-name=t8b-finalize-init-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable finalizer code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable finalizer code receipt}"
: "${TARGET8_EVALUATION_CODE_ROOT:?set immutable bundle which ran GreekMMLU}"
: "${TARGET8_EVALUATION_CODE_RECEIPT:?set evaluation bundle receipt}"
: "${TARGET8_INITIAL_HF:?set corrected initial HF model}"
: "${TARGET8_GREEKMMLU_EVALUATION_ROOT:?set completed raw GreekMMLU directory}"
: "${TARGET8_GREEKMMLU_LEGACY_RECEIPT:?set inherited aggregate GreekMMLU receipt}"
: "${TARGET8_GREEKMMLU_CLEAN_SUBSET:?set frozen decontaminated GreekMMLU subset}"
: "${TARGET8_GREEKMMLU_TARGETED_RECEIPT:?set new immutable targeted GreekMMLU receipt}"
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "GreekMMLU finalization must run on debug" >&2; exit 2; }
[[ ! -e "$TARGET8_GREEKMMLU_TARGETED_RECEIPT" ]] || { echo "refusing to overwrite targeted GreekMMLU receipt" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_targeted_initial_greekmmlu.py" \
  --evaluation-code-root "$TARGET8_EVALUATION_CODE_ROOT" \
  --evaluation-code-receipt "$TARGET8_EVALUATION_CODE_RECEIPT" \
  --model "$TARGET8_INITIAL_HF" \
  --evaluation-root "$TARGET8_GREEKMMLU_EVALUATION_ROOT" \
  --legacy-receipt "$TARGET8_GREEKMMLU_LEGACY_RECEIPT" \
  --clean-subset-manifest "$TARGET8_GREEKMMLU_CLEAN_SUBSET" \
  --output "$TARGET8_GREEKMMLU_TARGETED_RECEIPT"

#!/usr/bin/env bash
#SBATCH --job-name=mini_checkpoint_eval_plan
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${SCHEDULE_MANIFEST:?set completed five-arm schedule manifest}"
: "${EXPERIMENT_MATRIX:?set frozen experiment matrix}"
: "${CHECKPOINT_EVALUATION_PLAN:?set unique output plan path}"

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$EVALUATION_BUNDLE/evaluation/build_checkpoint_evaluation_plan.py" \
  --schedule-manifest "$SCHEDULE_MANIFEST" \
  --experiment-matrix "$EXPERIMENT_MATRIX" \
  --output "$CHECKPOINT_EVALUATION_PLAN"

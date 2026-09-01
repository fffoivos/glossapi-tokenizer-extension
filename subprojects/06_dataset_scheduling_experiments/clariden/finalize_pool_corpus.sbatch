#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_finalize
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail

: "${SCHEDULE_VALIDATION_BUNDLE:?set immutable validation code bundle}"
: "${SCHEDULE_STAGE_ROOT:?set Mini corpus stage}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCHEDULE_VALIDATION_BUNDLE/dataset/finalize_pool_corpus.py" \
  --stage-root "$SCHEDULE_STAGE_ROOT" \
  --expected-groups 513 \
  --expected-training-tasks 1457 \
  --expected-heldout-tasks 12

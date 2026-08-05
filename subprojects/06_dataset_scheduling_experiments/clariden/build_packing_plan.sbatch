#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_packplan
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
: "${SCHEDULE_PACKING_BUNDLE:?set immutable packing code bundle}"
: "${SCHEDULE_STAGE_ROOT:?set Mini corpus stage}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
PACKING_PLAN_PATH=${PACKING_PLAN_PATH:-$SCHEDULE_STAGE_ROOT/packing_plan.json}
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCHEDULE_PACKING_BUNDLE/dataset/build_packing_plan.py" \
  --pool-receipt "$SCHEDULE_STAGE_ROOT/pool_corpus_receipt.json" \
  --output "$PACKING_PLAN_PATH"

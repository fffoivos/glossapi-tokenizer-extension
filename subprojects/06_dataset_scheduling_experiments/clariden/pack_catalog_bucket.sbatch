#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_pack
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
: "${SCHEDULE_PACKING_BUNDLE:?set immutable packing code bundle}"
: "${SOURCE_REPO_ROOT:?set frozen corpus builder checkout}"
: "${SCHEDULE_STAGE_ROOT:?set Mini corpus stage}"
: "${SLURM_ARRAY_TASK_ID:?array job required}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
BRIDGE_COMMON="$SOURCE_REPO_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/bridge_common.py"
PACKING_PLAN_PATH=${PACKING_PLAN_PATH:-$SCHEDULE_STAGE_ROOT/packing_plan.json}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCHEDULE_PACKING_BUNDLE/dataset/pack_catalog_bucket.py" \
  --packing-plan "$PACKING_PLAN_PATH" \
  --input-receipt "$SCHEDULE_STAGE_ROOT/input_receipt.json" \
  --stage-root "$SCHEDULE_STAGE_ROOT" \
  --bridge-common "$BRIDGE_COMMON" \
  --task-index "$SLURM_ARRAY_TASK_ID"

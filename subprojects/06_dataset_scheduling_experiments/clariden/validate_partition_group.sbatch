#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_verify
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
set -euo pipefail

: "${SCHEDULE_VALIDATION_BUNDLE:?set immutable validation code bundle}"
: "${SOURCE_REPO_ROOT:?set frozen corpus builder checkout}"
: "${SCHEDULE_STAGE_ROOT:?set Mini corpus stage}"
: "${SLURM_ARRAY_TASK_ID:?array job required}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
BRIDGE_COMMON="$SOURCE_REPO_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/bridge_common.py"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCHEDULE_VALIDATION_BUNDLE/dataset/validate_partition_group.py" \
  --input-receipt "$SCHEDULE_STAGE_ROOT/input_receipt.json" \
  --heldout-manifest "$SCHEDULE_STAGE_ROOT/heldouts/heldout_manifest.json" \
  --stage-root "$SCHEDULE_STAGE_ROOT" \
  --bridge-common "$BRIDGE_COMMON" \
  --group-index "$SLURM_ARRAY_TASK_ID"

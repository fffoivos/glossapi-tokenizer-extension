#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_bins
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=04:00:00
set -euo pipefail

: "${SOURCE_REPO_ROOT:?set frozen corpus builder checkout}"
: "${SCHEDULE_STAGE_ROOT:?set new immutable Mini corpus stage}"
: "${TASK_OFFSET:?set array task offset}"
: "${SLURM_ARRAY_TASK_ID:?array job required}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
BRIDGE_DIR="$SOURCE_REPO_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge"
TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

test -s "$SCHEDULE_STAGE_ROOT/input_receipt.json"
test -s "$SCHEDULE_STAGE_ROOT/heldouts/heldout_manifest.json"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$BRIDGE_DIR/scripts/build_binary_shard.py" \
  --input-receipt "$SCHEDULE_STAGE_ROOT/input_receipt.json" \
  --heldout-manifest "$SCHEDULE_STAGE_ROOT/heldouts/heldout_manifest.json" \
  --stage-root "$SCHEDULE_STAGE_ROOT" \
  --task-index "$TASK_INDEX" \
  --workers "${BINARY_WORKERS:-48}" \
  --chunksize "${BINARY_CHUNKSIZE:-8}"

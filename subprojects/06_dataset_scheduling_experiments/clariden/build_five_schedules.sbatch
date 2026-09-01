#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_schedules
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
PACKED_CORPUS_RECEIPT_PATH=${PACKED_CORPUS_RECEIPT_PATH:-$SCHEDULE_STAGE_ROOT/packed_corpus_receipt.json}
SCHEDULE_OUTPUT_DIR=${SCHEDULE_OUTPUT_DIR:-$SCHEDULE_STAGE_ROOT/schedules}
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCHEDULE_PACKING_BUNDLE/dataset/build_five_schedules.py" \
  --packed-receipt "$PACKED_CORPUS_RECEIPT_PATH" \
  --output-dir "$SCHEDULE_OUTPUT_DIR"

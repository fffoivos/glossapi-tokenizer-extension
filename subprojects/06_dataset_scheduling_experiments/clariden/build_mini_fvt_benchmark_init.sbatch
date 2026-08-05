#!/usr/bin/env bash
#SBATCH --job-name=mini_fvt_init
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set frozen base model}"
: "${MINI_OVERLAY_TOKENIZER_DIR:?set frozen tokenizer overlay}"
: "${MINI_FVT_INIT_DIR:?set output directory}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}

test -s "$MINI_BASE_MODEL_DIR/base_model_receipt.json"
test -s "$MINI_OVERLAY_TOKENIZER_DIR/overlay_manifest.json"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/build_tied_retok_init.py" \
  --base-model "$MINI_BASE_MODEL_DIR" \
  --base-tokenizer "$MINI_BASE_MODEL_DIR" \
  --target-tokenizer "$MINI_OVERLAY_TOKENIZER_DIR" \
  --output-dir "$MINI_FVT_INIT_DIR" \
  --dtype bfloat16


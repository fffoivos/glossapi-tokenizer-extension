#!/usr/bin/env bash
#SBATCH --job-name=mini_ckpt_export_receipt
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${EXPORT_ROOT:?set completed checkpoint export root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
TOKENIZER_SHA=cc3f544817da0e8d1623e3f7484df7f67464aeb00867aece956880e9b407ef8f

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$EVALUATION_BUNDLE/evaluation/checkpoint_eval_export_receipt.py" finalize \
  --output-root "$EXPORT_ROOT" \
  --tokenizer-json "$TOKENIZER_DIR/tokenizer.json" \
  --tokenizer-json-sha256 "$TOKENIZER_SHA"

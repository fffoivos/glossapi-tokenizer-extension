#!/usr/bin/env bash
#SBATCH --job-name=mini_token_bytes
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${TOKEN_BYTE_ROOT:?set new token-byte artifact root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}

[[ ! -e "$TOKEN_BYTE_ROOT" ]] || {
  echo "refusing to replace TOKEN_BYTE_ROOT=$TOKEN_BYTE_ROOT" >&2
  exit 2
}
mkdir -p "$TOKEN_BYTE_ROOT"
uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$SCIENTIFIC_BUNDLE/initialization/build_token_byte_lengths.py" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-npy "$TOKEN_BYTE_ROOT/token_utf8_byte_lengths.npy" \
  --output-receipt "$TOKEN_BYTE_ROOT/token_utf8_byte_lengths.receipt.json"

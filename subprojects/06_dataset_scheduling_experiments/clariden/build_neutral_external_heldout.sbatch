#!/usr/bin/env bash
#SBATCH --job-name=mini_neutral_el
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=03:00:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_DEDUP_RECEIPT:?set passed cross-dedup receipt}"
: "${POOL_CORPUS_RECEIPT:?set frozen training pool receipt}"
: "${TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${NEUTRAL_OUTPUT_ROOT:?set new neutral heldout output root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
[[ ! -e "$NEUTRAL_OUTPUT_ROOT" ]] || { echo "refusing to replace $NEUTRAL_OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$NEUTRAL_OUTPUT_ROOT"
corpus="$NEUTRAL_OUTPUT_ROOT/neutral_external_corpus.json"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCIENTIFIC_BUNDLE/evaluation/finalize_neutral_external_corpus.py" \
  --dedup-receipt "$NEUTRAL_DEDUP_RECEIPT" \
  --pool-corpus-receipt "$POOL_CORPUS_RECEIPT" --output "$corpus"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCIENTIFIC_BUNDLE/evaluation/build_neutral_external_binary.py" \
  --corpus-receipt "$corpus" --tokenizer-dir "$TOKENIZER_DIR" \
  --output-prefix "$NEUTRAL_OUTPUT_ROOT/neutral_external_modern_greek"

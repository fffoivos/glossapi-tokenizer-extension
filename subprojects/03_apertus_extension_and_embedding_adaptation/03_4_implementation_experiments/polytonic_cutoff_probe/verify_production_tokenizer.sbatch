#!/usr/bin/env bash
#SBATCH --job-name=poly_tokenizer_audit
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/%x-%j.err

set -euo pipefail

: "${TOKENIZER_AUDIT_SCRIPT:?set TOKENIZER_AUDIT_SCRIPT}"
: "${TOKENIZER_BASE_DIR:?set TOKENIZER_BASE_DIR}"
: "${TOKENIZER_CANDIDATE_DIR:?set TOKENIZER_CANDIDATE_DIR}"
: "${TOKENIZER_AUDIT_OUTPUT:?set TOKENIZER_AUDIT_OUTPUT}"

UENV_BIN="${UENV_BIN:-/usr/bin/uenv}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"

"$UENV_BIN" run "$UENV_IMAGE" --view=default -- \
  python3 "$TOKENIZER_AUDIT_SCRIPT" \
    --base-dir "$TOKENIZER_BASE_DIR" \
    --candidate-dir "$TOKENIZER_CANDIDATE_DIR" \
    --expected-vocab-size 148992 \
    --expected-added-merges 512 \
    --alignment 256 \
    --require-runtime \
    --output "$TOKENIZER_AUDIT_OUTPUT"

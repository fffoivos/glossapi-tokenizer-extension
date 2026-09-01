#!/bin/bash
#SBATCH --job-name=gmmlu-responses
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/logs/gmmlu-responses-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/logs/gmmlu-responses-%j.err

set -euo pipefail

: "${EXTRACTOR:?set EXTRACTOR to extract_greekmmlu_history_matrix.py}"
: "${DRIFT_RECEIPT:?set DRIFT_RECEIPT to greekmmlu_answer_drift.json}"
: "${HISTORY_OUTPUT:?set HISTORY_OUTPUT to the compact exact-response NPZ}"
: "${ANALYZER:?set ANALYZER to analyze_greekmmlu_response_displacement_by_category.py}"
: "${JSON_OUTPUT:?set JSON_OUTPUT to the category result JSON}"
: "${CSV_OUTPUT:?set CSV_OUTPUT to the category result CSV}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
PERMUTATIONS="${PERMUTATIONS:-1999}"

test "${SLURM_JOB_PARTITION:-}" = "debug"
test "${SLURM_JOB_NUM_NODES:-}" = "1"
mkdir -p \
  "$(dirname "$HISTORY_OUTPUT")" \
  "$(dirname "$JSON_OUTPUT")" \
  "$(dirname "$CSV_OUTPUT")" \
  /capstor/scratch/cscs/fffoivos/logs

uenv run "$UENV_IMAGE" --view=default -- \
  python3 "$EXTRACTOR" \
    --drift-receipt "$DRIFT_RECEIPT" \
    --output "$HISTORY_OUTPUT"

uenv run "$UENV_IMAGE" --view=default -- \
  python3 "$ANALYZER" \
    --history "$HISTORY_OUTPUT" \
    --output-json "$JSON_OUTPUT" \
    --output-csv "$CSV_OUTPUT" \
    --permutations "$PERMUTATIONS"

test -s "$HISTORY_OUTPUT"
test -s "$JSON_OUTPUT"
test -s "$CSV_OUTPUT"

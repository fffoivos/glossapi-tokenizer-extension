#!/bin/bash
#SBATCH --job-name=bib-token-count
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:25:00
#SBATCH --partition=debug
#SBATCH --account=a0140
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${BIB_CODE_ROOT:?set BIB_CODE_ROOT}"
: "${BIB_TOKEN_PLAN:?set BIB_TOKEN_PLAN}"
: "${BIB_TOKEN_RECEIPT_DIR:?set BIB_TOKEN_RECEIPT_DIR}"
: "${BIB_PYTHON:?set BIB_PYTHON}"
: "${SLURM_ARRAY_TASK_ID:?requires a Slurm array}"

cd "$BIB_CODE_ROOT"
exec "$BIB_PYTHON" -m \
  subprojects.05_token_distillation_cpt.02_corpus_preparation.15_clean_academic.production.count_tokens \
  task \
  --plan "$BIB_TOKEN_PLAN" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --receipt-dir "$BIB_TOKEN_RECEIPT_DIR" \
  --batch-size 256

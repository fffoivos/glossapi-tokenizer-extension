#!/usr/bin/env bash
#SBATCH --job-name=publish-contam
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${AUDIT_CODE_ROOT:?}"
: "${AUDIT_CODE_RECEIPT:?}"
: "${AUDIT_RUN_ROOT:?}"
: "${AUDIT_PYTHON:?}"
: "${AUDIT_HF_TOKEN_FILE:?}"
"$AUDIT_PYTHON" \
  "$AUDIT_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$AUDIT_CODE_ROOT" --receipt "$AUDIT_CODE_RECEIPT"
"$AUDIT_PYTHON" \
  "$AUDIT_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/publish_benchmark_contamination_audit.py" \
  --payload-dir "$AUDIT_RUN_ROOT/publish_payload" \
  --token-file "$AUDIT_HF_TOKEN_FILE" \
  --receipt "$AUDIT_RUN_ROOT/hf_publication_receipt.json"

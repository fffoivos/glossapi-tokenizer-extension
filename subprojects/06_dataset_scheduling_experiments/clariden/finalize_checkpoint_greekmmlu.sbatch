#!/usr/bin/env bash
#SBATCH --job-name=mini_greekmmlu_receipt
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${EXPORT_RECEIPT:?set exact checkpoint export receipt}"
: "${GREEKMMLU_ROOT:?set completed native GreekMMLU output root}"
: "${MODEL_LABEL:?set evaluator model label}"
: "${OUTPUT_RECEIPT:?set unique output receipt}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen decontaminated GreekMMLU subset}"
: "${EVALUATION_NAMESPACE:?set isolated evaluation namespace}"

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$EVALUATION_BUNDLE/evaluation/finalize_checkpoint_greekmmlu_receipt.py" \
  --export-receipt "$EXPORT_RECEIPT" \
  --evaluation-root "$GREEKMMLU_ROOT" \
  --model-label "$MODEL_LABEL" \
  --evaluation-namespace "$EVALUATION_NAMESPACE" \
  --clean-subset-manifest "$GREEKMMLU_CLEAN_SUBSET" \
  --output "$OUTPUT_RECEIPT"

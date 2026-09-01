#!/usr/bin/env bash
#SBATCH --job-name=mini_native_greekmmlu
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=00:30:00

set -euo pipefail
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${EXPORT_ROOT:?set completed exact-checkpoint export root}"
: "${GREEKMMLU_ROOT:?set unique GreekMMLU output root}"
: "${MODEL_LABEL:?set checkpoint-specific model label}"
: "${EVALUATION_NAMESPACE:?set isolated evaluation namespace}"
[[ "${EVAL_DTYPE:-float32}" == "float32" ]] || {
  echo "authoritative GreekMMLU requires float32, got ${EVAL_DTYPE:-unset}" >&2
  exit 2
}

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
evaluator="$NATIVE_GREEK_EVAL_ROOT/run_native_greek_mcq_eval.py"
registry="$NATIVE_GREEK_EVAL_ROOT/native_greek_benchmark_registry.json"
[[ -f "$evaluator" && -f "$registry" && -f "$EXPORT_ROOT/checkpoint_eval_export_receipt.json" ]] || {
  echo "missing evaluator, registry, or exact-checkpoint export receipt" >&2
  exit 2
}
[[ ! -e "$GREEKMMLU_ROOT" ]] || {
  echo "refusing to overwrite GreekMMLU result: $GREEKMMLU_ROOT" >&2
  exit 2
}

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$evaluator" \
  --registry "$registry" \
  --benchmarks greekmmlu \
  --model "$MODEL_LABEL=$EXPORT_ROOT/hf" \
  --output-dir "$GREEKMMLU_ROOT" \
  --sample-size 0 \
  --random-state 42 \
  --dtype float32 \
  --max-input-tokens 3072 \
  --candidate-batch-size 16 \
  --example-batch-size 16 \
  --trust-remote-code

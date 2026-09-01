#!/usr/bin/env bash
#SBATCH --job-name=mini_greekmmlu_dtype_diag
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=00:30:00

set -euo pipefail
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${HF_MODEL_ROOT:?set diagnostic HF model root}"
: "${OUTPUT_ROOT:?set unique diagnostic output root}"
: "${MODEL_LABEL:?set diagnostic model label}"
: "${EVAL_DTYPE:?set explicit diagnostic dtype}"

case "$EVAL_DTYPE" in
  float32|bfloat16) ;;
  *) echo "unsupported diagnostic dtype: $EVAL_DTYPE" >&2; exit 2 ;;
esac

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
evaluator="$NATIVE_GREEK_EVAL_ROOT/run_native_greek_mcq_eval.py"
registry="$NATIVE_GREEK_EVAL_ROOT/native_greek_benchmark_registry.json"
[[ -f "$evaluator" && -f "$registry" && -f "$HF_MODEL_ROOT/model.safetensors" ]] || {
  echo "missing evaluator, registry, or diagnostic model" >&2
  exit 2
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "refusing to overwrite dtype diagnostic: $OUTPUT_ROOT" >&2
  exit 2
}

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$evaluator" \
  --registry "$registry" \
  --benchmarks greekmmlu \
  --model "$MODEL_LABEL=$HF_MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --sample-size 0 \
  --random-state 42 \
  --dtype "$EVAL_DTYPE" \
  --max-input-tokens 3072 \
  --candidate-batch-size 16 \
  --example-batch-size 16 \
  --trust-remote-code

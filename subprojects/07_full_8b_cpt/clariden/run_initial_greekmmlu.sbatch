#!/usr/bin/env bash
#SBATCH --job-name=full8b-init-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_INITIAL_HF:?set verified roundtrip HF initialization}"
: "${FULL8_INITIAL_EVAL_ROOT:?set new output root}"
[[ ! -e "$FULL8_INITIAL_EVAL_ROOT" ]] || { echo "refusing to overwrite initial evaluation" >&2; exit 2; }
mkdir -p "$FULL8_INITIAL_EVAL_ROOT/greekmmlu"
NATIVE_ROOT="$FULL8_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
CLEAN_SUBSET=${FULL8_GREEKMMLU_CLEAN_SUBSET:-/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_subset_manifest.json}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
label=full8b_mixed_iter0
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$NATIVE_ROOT/run_native_greek_mcq_eval.py" \
  --registry "$NATIVE_ROOT/native_greek_benchmark_registry.json" \
  --benchmarks greekmmlu --model "$label=$FULL8_INITIAL_HF" \
  --output-dir "$FULL8_INITIAL_EVAL_ROOT/greekmmlu" --sample-size 0 \
  --random-state 42 --dtype float32 --max-input-tokens 3072 \
  --candidate-batch-size 16 --example-batch-size 16 --trust-remote-code
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/finalize_hf_greekmmlu.py" \
  --model "$FULL8_INITIAL_HF" --evaluation-root "$FULL8_INITIAL_EVAL_ROOT/greekmmlu" \
  --model-label "$label" --clean-subset-manifest "$CLEAN_SUBSET" \
  --output "$FULL8_INITIAL_EVAL_ROOT/initial_greekmmlu_receipt.json"

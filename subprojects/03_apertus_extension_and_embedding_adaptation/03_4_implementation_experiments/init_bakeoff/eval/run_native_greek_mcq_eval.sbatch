#!/usr/bin/env bash
#SBATCH --job-name=native_greek_mcq
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
MODEL_SPEC="${MODEL_SPEC:?MODEL_SPEC is required, e.g. TokenDistil-5B=/path/to/hf}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
BENCHMARKS="${BENCHMARKS:-all}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
DTYPE="${DTYPE:-bfloat16}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-16}"
EXAMPLE_BATCH_SIZE="${EXAMPLE_BATCH_SIZE:-16}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
PY_ENV="${PY_ENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_cpt20260610}"
CACHE_ROOT="${CACHE_ROOT:-/iopsstor/scratch/cscs/fffoivos/native_greek_eval/cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"/{hf_home,hf_datasets,xdg,tmp}

echo "=== run_native_greek_mcq_eval.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "MODEL_SPEC: $MODEL_SPEC"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BENCHMARKS: $BENCHMARKS"
echo "SAMPLE_SIZE: $SAMPLE_SIZE"
echo "CANDIDATE_BATCH_SIZE: $CANDIDATE_BATCH_SIZE"
echo "EXAMPLE_BATCH_SIZE: $EXAMPLE_BATCH_SIZE"
echo

export MODEL_SPEC OUTPUT_DIR BENCHMARKS SAMPLE_SIZE RANDOM_STATE DTYPE MAX_INPUT_TOKENS
export CANDIDATE_BATCH_SIZE EXAMPLE_BATCH_SIZE PY_ENV CACHE_ROOT SCRIPT_DIR

uenv start "$UENV_IMAGE" --view=default --ignore-tty <<'INNER'
set -euo pipefail
export HF_HOME=$CACHE_ROOT/hf_home
export HF_DATASETS_CACHE=$CACHE_ROOT/hf_datasets
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export TMPDIR=$CACHE_ROOT/tmp
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

mkdir -p "$(dirname "$PY_ENV")"
(
  flock 9
  if [ ! -x "$PY_ENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$PY_ENV"
  fi
  . "$PY_ENV/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install datasets pandas pyarrow accelerate huggingface_hub transformers \
    'dill==0.3.8' 'multiprocess<0.70.17'
) 9>"$PY_ENV.lock"
. "$PY_ENV/bin/activate"

python "$SCRIPT_DIR/run_native_greek_mcq_eval.py" \
  --registry "$SCRIPT_DIR/native_greek_benchmark_registry.json" \
  --benchmarks "$BENCHMARKS" \
  --model "$MODEL_SPEC" \
  --output-dir "$OUTPUT_DIR" \
  --sample-size "$SAMPLE_SIZE" \
  --random-state "$RANDOM_STATE" \
  --dtype "$DTYPE" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --example-batch-size "$EXAMPLE_BATCH_SIZE"
INNER

echo
echo "=== native Greek MCQ eval complete ==="
date -u
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort

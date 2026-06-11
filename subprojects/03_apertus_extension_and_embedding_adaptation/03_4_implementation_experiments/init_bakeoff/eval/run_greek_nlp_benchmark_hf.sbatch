#!/usr/bin/env bash
#SBATCH --job-name=greek_nlp_bench
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/iopsstor/scratch/cscs/fffoivos/benchmarks/greek-nlp-benchmark/main_e2d4bfc}"
MODEL_SPEC="${MODEL_SPEC:?MODEL_SPEC is required, e.g. td_5b=/path/to/hf}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TASK="${TASK:-all}"
SAMPLE_SIZE="${SAMPLE_SIZE:-25}"
TASK_CAP_PROFILE="${TASK_CAP_PROFILE:-none}"
NUM_PREDICT="${NUM_PREDICT:-128}"
TASK_NUM_PREDICT_OVERRIDES="${TASK_NUM_PREDICT_OVERRIDES:-gec=128,intent_classification=16,legal_classification=16,machine_translation=128,ner=96,pos=96,summarization=192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RANDOM_STATE="${RANDOM_STATE:-42}"
REPEATS="${REPEATS:-1}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
PY_ENV="${PY_ENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_nlp_benchmark_py312_cpt20260610}"
CACHE_ROOT="${CACHE_ROOT:-/iopsstor/scratch/cscs/fffoivos/tmp/greek_nlp_benchmark_${SLURM_JOB_ID:-manual}}"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"/{hf_home,hf_datasets,xdg,tmp}

echo "=== run_greek_nlp_benchmark_hf.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BENCHMARK_ROOT: $BENCHMARK_ROOT"
echo "MODEL_SPEC: $MODEL_SPEC"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TASK: $TASK"
echo "SAMPLE_SIZE: $SAMPLE_SIZE"
echo "TASK_CAP_PROFILE: $TASK_CAP_PROFILE"
echo "NUM_PREDICT: $NUM_PREDICT"
echo "TASK_NUM_PREDICT_OVERRIDES: $TASK_NUM_PREDICT_OVERRIDES"
echo "MAX_INPUT_TOKENS: $MAX_INPUT_TOKENS"
echo "DTYPE: $DTYPE"
echo "PY_ENV: $PY_ENV"
echo "CACHE_ROOT: $CACHE_ROOT"
echo

if [ ! -f "$BENCHMARK_ROOT/requirements.txt" ] || [ ! -d "$BENCHMARK_ROOT/benchmark_suite" ]; then
  echo "ERROR: BENCHMARK_ROOT does not look like greek-nlp/benchmark: $BENCHMARK_ROOT" >&2
  exit 2
fi

uenv start "$UENV_IMAGE" --view=default --ignore-tty <<INNER
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
  python -m pip install -r "$BENCHMARK_ROOT/requirements.txt" sacrebleu seqeval rouge-score bert-score nltk openpyxl accelerate \
    'dill==0.3.8' 'multiprocess<0.70.17'
) 9>"$PY_ENV.lock"
. "$PY_ENV/bin/activate"

python "$SCRIPT_DIR/run_greek_nlp_benchmark_hf.py" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --task "$TASK" \
  --models "$MODEL_SPEC" \
  --sample-size "$SAMPLE_SIZE" \
  --repeats "$REPEATS" \
  --random-state "$RANDOM_STATE" \
  --output-dir "$OUTPUT_DIR" \
  --task-cap-profile "$TASK_CAP_PROFILE" \
  --temperature "$TEMPERATURE" \
  --num-predict "$NUM_PREDICT" \
  --task-num-predict-overrides "$TASK_NUM_PREDICT_OVERRIDES" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --dtype "$DTYPE"
INNER

echo
echo "=== greek-nlp benchmark complete ==="
date -u
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort

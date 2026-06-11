#!/usr/bin/env bash
#SBATCH --job-name=greek_nlp4
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=800G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/iopsstor/scratch/cscs/fffoivos/benchmarks/greek-nlp-benchmark/main_e2d4bfc}"
MODEL_SPECS="${MODEL_SPECS:?MODEL_SPECS is required as semicolon-separated LABEL=PATH entries}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TASK="${TASK:-all}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
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
CACHE_ROOT="${CACHE_ROOT:-/iopsstor/scratch/cscs/fffoivos/native_greek_eval/cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"/{hf_home,hf_datasets,xdg,tmp}

echo "=== run_greek_nlp_benchmark_hf_packed.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "BENCHMARK_ROOT: $BENCHMARK_ROOT"
echo "MODEL_SPECS: $MODEL_SPECS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TASK: $TASK"
echo "SAMPLE_SIZE: $SAMPLE_SIZE"
echo "TASK_NUM_PREDICT_OVERRIDES: $TASK_NUM_PREDICT_OVERRIDES"
echo

if [ ! -f "$BENCHMARK_ROOT/requirements.txt" ] || [ ! -d "$BENCHMARK_ROOT/benchmark_suite" ]; then
  echo "ERROR: BENCHMARK_ROOT does not look like greek-nlp/benchmark: $BENCHMARK_ROOT" >&2
  exit 2
fi

export BENCHMARK_ROOT MODEL_SPECS OUTPUT_DIR TASK SAMPLE_SIZE TASK_CAP_PROFILE NUM_PREDICT
export TASK_NUM_PREDICT_OVERRIDES MAX_INPUT_TOKENS DTYPE TEMPERATURE RANDOM_STATE REPEATS
export PY_ENV CACHE_ROOT SCRIPT_DIR

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
  python -m pip install -r "$BENCHMARK_ROOT/requirements.txt" sacrebleu seqeval rouge-score bert-score nltk openpyxl accelerate \
    'dill==0.3.8' 'multiprocess<0.70.17'
) 9>"$PY_ENV.lock"
. "$PY_ENV/bin/activate"

IFS=';' read -r -a SPECS <<< "$MODEL_SPECS"
if [ "${#SPECS[@]}" -gt 4 ]; then
  echo "ERROR: packed greek-nlp eval supports at most 4 MODEL_SPECS per node" >&2
  exit 2
fi

status=0
for idx in "${!SPECS[@]}"; do
  spec="${SPECS[$idx]}"
  label="${spec%%=*}"
  safe_label="$(echo "$label" | tr -c 'A-Za-z0-9._-' '_')"
  out="$OUTPUT_DIR/$safe_label"
  mkdir -p "$out"
  echo "[launch] gpu=$idx model=$label out=$out"
  (
    export CUDA_VISIBLE_DEVICES="$idx"
    python "$SCRIPT_DIR/run_greek_nlp_benchmark_hf.py" \
      --benchmark-root "$BENCHMARK_ROOT" \
      --task "$TASK" \
      --models "$spec" \
      --sample-size "$SAMPLE_SIZE" \
      --repeats "$REPEATS" \
      --random-state "$RANDOM_STATE" \
      --output-dir "$out" \
      --task-cap-profile "$TASK_CAP_PROFILE" \
      --temperature "$TEMPERATURE" \
      --num-predict "$NUM_PREDICT" \
      --task-num-predict-overrides "$TASK_NUM_PREDICT_OVERRIDES" \
      --max-input-tokens "$MAX_INPUT_TOKENS" \
      --dtype "$DTYPE"
  ) > "$out/stdout.log" 2> "$out/stderr.log" &
done

for pid in $(jobs -p); do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
INNER

echo
echo "=== packed greek-nlp benchmark complete ==="
date -u
find "$OUTPUT_DIR" -maxdepth 3 -type f | sort

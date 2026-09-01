#!/usr/bin/env bash
#SBATCH --job-name=mini_td_pilot_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=06:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set frozen base Mini model}"
: "${MINI_FVT_INIT_DIR:?set frozen tied FVT initialization}"
: "${TD_PILOT_ROOT:?set completed TD pilot root}"
: "${TD_HPLT_EVAL_JSONL:?set frozen HPLT pilot heldout JSONL}"
: "${TD_NON_HPLT_EVAL_JSONL:?set frozen non-HPLT pilot heldout JSONL}"
: "${TD_POLYTONIC_EVAL_JSONL:?set frozen polytonic pilot heldout JSONL}"
: "${TD_FAIR_METRICS_SCRIPT:?set frozen tokenizer-fair metrics script}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
FAIR_METRICS_SCRIPT=$TD_FAIR_METRICS_SCRIPT
MAX_DOCS=${TD_PILOT_EVAL_MAX_DOCS:-512}

for path in "$TD_HPLT_EVAL_JSONL" "$TD_NON_HPLT_EVAL_JSONL" "$TD_POLYTONIC_EVAL_JSONL"; do test -s "$path"; done
test -s "$FAIR_METRICS_SCRIPT"
[[ "$(sha256sum "$FAIR_METRICS_SCRIPT" | awk '{print $1}')" == 96c3b0e048753a3414e2ab9abf3b78824375fc0994793fe116d2dbffbb09b3ff ]] || {
  echo "tokenizer-fair metrics script drift" >&2; exit 3;
}
test -s "$TD_PILOT_ROOT/training_complete.txt"

run_one() {
  local pilot_id=$1 gpu=$2
  local model="$TD_PILOT_ROOT/$pilot_id"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
      "$MINI_CODE_BUNDLE/initialization/verify_tied_initialization.py" \
      --base-model "$MINI_BASE_MODEL_DIR" \
      --candidate-model "$model" \
      --probe-jsonl "$TD_POLYTONIC_EVAL_JSONL" \
      --output-json "$model/initialization_verification.json" \
      --max-probe-docs 256 --max-generation-prompts 64 --device cuda
    while IFS=$'\t' read -r label jsonl; do
      mkdir -p "$model/metrics/$label"
      uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
        "$FAIR_METRICS_SCRIPT" \
        --model-path "$model" --eval-jsonl "$jsonl" \
        --output-json "$model/metrics/$label/tokenizer_fair_metrics.json" \
        --max-context 4096 --max-docs "$MAX_DOCS" --device cuda --dtype bfloat16
    done <<EOF
hplt	$TD_HPLT_EVAL_JSONL
non_hplt	$TD_NON_HPLT_EVAL_JSONL
polytonic	$TD_POLYTONIC_EVAL_JSONL
EOF
  ) >"$TD_PILOT_ROOT/logs/${pilot_id}_evaluation.out" 2>"$TD_PILOT_ROOT/logs/${pilot_id}_evaluation.err" &
  pids+=("$!")
}

pids=()
run_one layer7_mse 0
run_one layer7_mse_ce_auto 1
run_one last_mse 2
run_one last_mse_ce_auto 3
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "one or more TD pilot evaluations failed" >&2
  exit "$status"
fi

# Rank TD recipes against the exact initialization they modify.  Without this
# control, a selector could choose the least harmful TD cell even when every
# cell regresses heldout BPB relative to tied FVT.
FVT_BASELINE_ROOT="$TD_PILOT_ROOT/fvt_baseline"
mkdir -p "$FVT_BASELINE_ROOT/metrics"
export CUDA_VISIBLE_DEVICES=0
while IFS=$'\t' read -r label jsonl; do
  mkdir -p "$FVT_BASELINE_ROOT/metrics/$label"
  uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
    "$FAIR_METRICS_SCRIPT" \
    --model-path "$MINI_FVT_INIT_DIR" --eval-jsonl "$jsonl" \
    --output-json "$FVT_BASELINE_ROOT/metrics/$label/tokenizer_fair_metrics.json" \
    --max-context 4096 --max-docs "$MAX_DOCS" --device cuda --dtype bfloat16
done <<EOF
hplt	$TD_HPLT_EVAL_JSONL
non_hplt	$TD_NON_HPLT_EVAL_JSONL
polytonic	$TD_POLYTONIC_EVAL_JSONL
EOF

uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/select_tied_td_pilot.py" \
  --pilot-root "$TD_PILOT_ROOT" \
  --pilot-token-selection "$TD_PILOT_ROOT/selection/pilot_token_selection.json" \
  --fvt-baseline-dir "$FVT_BASELINE_ROOT" \
  --fvt-model-dir "$MINI_FVT_INIT_DIR" \
  --output "$TD_PILOT_ROOT/selected_pilot.json"

#!/usr/bin/env bash
#SBATCH --job-name=mini_td_full_verify
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set frozen base Mini model}"
: "${MINI_FULL_TD_DIR:?set full tied-TD HF directory}"
: "${TD_HPLT_EVAL_JSONL:?set frozen HPLT pilot heldout JSONL}"
: "${TD_NON_HPLT_EVAL_JSONL:?set frozen non-HPLT pilot heldout JSONL}"
: "${TD_POLYTONIC_EVAL_JSONL:?set frozen polytonic pilot heldout JSONL}"
: "${TD_FAIR_METRICS_SCRIPT:?set frozen tokenizer-fair metrics script}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
FAIR_METRICS_SCRIPT=$TD_FAIR_METRICS_SCRIPT
MAX_DOCS=${TD_PILOT_EVAL_MAX_DOCS:-512}

[[ "$(sha256sum "$FAIR_METRICS_SCRIPT" | awk '{print $1}')" == 96c3b0e048753a3414e2ab9abf3b78824375fc0994793fe116d2dbffbb09b3ff ]] || {
  echo "tokenizer-fair metrics script drift" >&2; exit 3;
}

uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/verify_tied_initialization.py" \
  --base-model "$MINI_BASE_MODEL_DIR" --candidate-model "$MINI_FULL_TD_DIR" \
  --probe-jsonl "$TD_POLYTONIC_EVAL_JSONL" \
  --output-json "$MINI_FULL_TD_DIR/initialization_verification.json" \
  --max-probe-docs 256 --max-generation-prompts 64 --device cuda
while IFS=$'\t' read -r label jsonl; do
  mkdir -p "$MINI_FULL_TD_DIR/metrics/$label"
  uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
    "$FAIR_METRICS_SCRIPT" --model-path "$MINI_FULL_TD_DIR" \
    --eval-jsonl "$jsonl" \
    --output-json "$MINI_FULL_TD_DIR/metrics/$label/tokenizer_fair_metrics.json" \
    --max-context 4096 --max-docs "$MAX_DOCS" --device cuda --dtype bfloat16
done <<EOF
hplt	$TD_HPLT_EVAL_JSONL
non_hplt	$TD_NON_HPLT_EVAL_JSONL
polytonic	$TD_POLYTONIC_EVAL_JSONL
EOF

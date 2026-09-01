#!/usr/bin/env bash
#SBATCH --job-name=mini_td_pilots
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
: "${MINI_OVERLAY_TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${MINI_FVT_INIT_DIR:?set frozen tied FVT initialization}"
: "${TD_CANONICAL_ADAPTER:?set frozen canonical Token Distillation adapter}"
: "${TD_COVERAGE_JSONL:?set normalized complete TD coverage JSONL}"
: "${TD_SNIPPETS_JSONL:?set frozen TD snippet index}"
: "${TD_PILOT_ROOT:?set new TD pilot output root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}

[[ ! -e "$TD_PILOT_ROOT" ]] || { echo "refusing to replace $TD_PILOT_ROOT" >&2; exit 2; }
test -s "$MINI_BASE_MODEL_DIR/base_model_receipt.json"
test -s "$MINI_OVERLAY_TOKENIZER_DIR/overlay_manifest.json"
test -s "$MINI_FVT_INIT_DIR/tied_retok_manifest.json"
mkdir -p "$TD_PILOT_ROOT/logs"

uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/select_mini_td_pilot_tokens.py" \
  --coverage-jsonl "$TD_COVERAGE_JSONL" \
  --output-dir "$TD_PILOT_ROOT/selection"
PILOT_IDS="$TD_PILOT_ROOT/selection/pilot_token_ids.txt"

run_one() {
  local pilot_id=$1 target_layer=$2 loss_profile=$3 gpu=$4
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
      "$MINI_CODE_BUNDLE/initialization/train_tied_token_distillation.py" \
      --retok-model "$MINI_FVT_INIT_DIR" \
      --canonical-adapter "$TD_CANONICAL_ADAPTER" \
      --base-tokenizer "$MINI_BASE_MODEL_DIR" \
      --target-tokenizer "$MINI_OVERLAY_TOKENIZER_DIR" \
      --coverage-jsonl "$TD_COVERAGE_JSONL" \
      --snippets-jsonl "$TD_SNIPPETS_JSONL" \
      --token-ids-file "$PILOT_IDS" \
      --output-dir "$TD_PILOT_ROOT/$pilot_id" \
      --scope pilot \
      --target-layer "$target_layer" \
      --loss-profile "$loss_profile" \
      --snippets-per-token 25 \
      --min-accepted-snippets-per-token 25 \
      --min-trained-token-fraction 1.0 \
      --epochs 1 --batch-size 16 --learning-rate 1e-4 \
      --seed 20260801 --device cuda
  ) >"$TD_PILOT_ROOT/logs/$pilot_id.out" 2>"$TD_PILOT_ROOT/logs/$pilot_id.err" &
  pids+=("$!")
}

pids=()
run_one layer7_mse 7 mse 0
run_one layer7_mse_ce_auto 7 mse_ce_auto 1
run_one last_mse -1 mse 2
run_one last_mse_ce_auto -1 mse_ce_auto 3
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "one or more tied-TD pilot cells failed" >&2
  exit "$status"
fi
date -u +%FT%TZ > "$TD_PILOT_ROOT/training_complete.txt"

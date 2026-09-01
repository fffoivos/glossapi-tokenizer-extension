#!/usr/bin/env bash
#SBATCH --job-name=mini_td_full
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=12:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set frozen base Mini model}"
: "${MINI_OVERLAY_TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
: "${MINI_FVT_INIT_DIR:?set frozen tied FVT initialization}"
: "${TD_CANONICAL_ADAPTER:?set frozen canonical Token Distillation adapter}"
: "${TD_COVERAGE_JSONL:?set normalized complete TD coverage JSONL}"
: "${TD_SNIPPETS_JSONL:?set frozen TD snippet index}"
: "${TD_PILOT_SELECTION:?set selected_pilot.json}"
: "${MINI_FULL_TD_DIR:?set new full TD output directory}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}

[[ ! -e "$MINI_FULL_TD_DIR" ]] || { echo "refusing to replace $MINI_FULL_TD_DIR" >&2; exit 2; }
work="${MINI_FULL_TD_DIR}.inputs"
[[ ! -e "$work" ]] || { echo "refusing to replace $work" >&2; exit 2; }
mkdir -p "$work"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/write_full_added_token_ids.py" \
  --output "$work/full_added_token_ids.txt" \
  --receipt "$work/full_added_token_ids_receipt.json"
readarray -t selected < <(uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" - "$TD_PILOT_SELECTION" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["status"] == "selected"
assert d["fvt_macro_non_regression_gate_passed"] is True
assert float(d["selected_delta_macro_bpb_vs_fvt"]) <= 0.0
print(d["selected_target_layer"])
print(d["selected_loss_profile"])
PY
)
target_layer=${selected[0]}
loss_profile=${selected[1]}
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/train_tied_token_distillation.py" \
  --retok-model "$MINI_FVT_INIT_DIR" \
  --canonical-adapter "$TD_CANONICAL_ADAPTER" \
  --base-tokenizer "$MINI_BASE_MODEL_DIR" \
  --target-tokenizer "$MINI_OVERLAY_TOKENIZER_DIR" \
  --coverage-jsonl "$TD_COVERAGE_JSONL" \
  --snippets-jsonl "$TD_SNIPPETS_JSONL" \
  --token-ids-file "$work/full_added_token_ids.txt" \
  --output-dir "$MINI_FULL_TD_DIR" \
  --scope full --target-layer "$target_layer" --loss-profile "$loss_profile" \
  --snippets-per-token 25 --min-accepted-snippets-per-token 25 \
  --min-trained-token-fraction 0.90 --epochs 1 --batch-size 16 \
  --learning-rate 1e-4 --seed 20260801 --device cuda
cp "$TD_PILOT_SELECTION" "$MINI_FULL_TD_DIR/selected_pilot.json"
cp "$work/full_added_token_ids_receipt.json" "$MINI_FULL_TD_DIR/"

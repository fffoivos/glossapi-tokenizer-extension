#!/usr/bin/env bash
#SBATCH --job-name=retok_td_layer_pilot
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=216
#SBATCH --mem=450G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err
#
# Run a small TD layer pilot inside one full-node allocation. Clariden normal
# jobs allocate a whole 4-GPU node even for a one-GPU request, so independent
# layer candidates should be packed into one job instead of submitted as
# separate single-process jobs.

set -euo pipefail

STUDENT_MODEL="${STUDENT_MODEL:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok}"
BASE_TOKENIZER="${BASE_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
COVERAGE_JSONL="${COVERAGE_JSONL:?COVERAGE_JSONL is required}"
SNIPPETS_JSONL="${SNIPPETS_JSONL:?SNIPPETS_JSONL is required}"
TOKEN_IDS_FILE="${TOKEN_IDS_FILE:?TOKEN_IDS_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

TARGET_LAYERS="${TARGET_LAYERS:--1 11}"
SNIPPETS_PER_TOKEN="${SNIPPETS_PER_TOKEN:-50}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIN_ACCEPTED_SNIPPETS_PER_TOKEN="${MIN_ACCEPTED_SNIPPETS_PER_TOKEN:-$SNIPPETS_PER_TOKEN}"
MIN_TRAINED_TOKEN_FRACTION="${MIN_TRAINED_TOKEN_FRACTION:-0.90}"
MAX_SELECTED_TOKENS="${MAX_SELECTED_TOKENS:-1024}"
DRY_RUN="${DRY_RUN:-0}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${SCRIPT_DIR_OVERRIDE:-}" ]; then
    SCRIPT_DIR="$(cd "$SCRIPT_DIR_OVERRIDE" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train_retok_td.py" ]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

mapfile -t LAYERS < <(printf '%s\n' $TARGET_LAYERS)
if [ "${#LAYERS[@]}" -eq 0 ]; then
    echo "ERROR: TARGET_LAYERS is empty" >&2
    exit 2
fi
if [ "${#LAYERS[@]}" -gt 4 ]; then
    echo "ERROR: packed layer pilot supports at most 4 layers, got ${#LAYERS[@]}" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" /capstor/scratch/cscs/fffoivos/runs/token_distillation

echo "=== train_retok_td_layer_pilot_packed.sbatch ==="
date -u
echo "host:              $(hostname)"
echo "STUDENT_MODEL:     $STUDENT_MODEL"
echo "BASE_TOKENIZER:    $BASE_TOKENIZER"
echo "STUDENT_TOKENIZER: $STUDENT_TOKENIZER"
echo "COVERAGE_JSONL:    $COVERAGE_JSONL"
echo "SNIPPETS_JSONL:    $SNIPPETS_JSONL"
echo "TOKEN_IDS_FILE:    $TOKEN_IDS_FILE"
echo "OUTPUT_ROOT:       $OUTPUT_ROOT"
echo "TARGET_LAYERS:     ${LAYERS[*]}"
echo "SNIPPETS_PER_TOKEN:$SNIPPETS_PER_TOKEN"
echo "BATCH_SIZE:        $BATCH_SIZE"
echo "MAX_SELECTED_TOKENS:$MAX_SELECTED_TOKENS (use 0 or all for full token list)"
echo "MIN_ACCEPTED_SNIPPETS_PER_TOKEN:$MIN_ACCEPTED_SNIPPETS_PER_TOKEN"
echo "MIN_TRAINED_TOKEN_FRACTION:$MIN_TRAINED_TOKEN_FRACTION"
echo "PYTORCH_ALLOC_CONF:$PYTORCH_ALLOC_CONF"
echo "DRY_RUN:           $DRY_RUN"
echo "CUDA_VISIBLE_DEVICES at job start: ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

layer_label() {
    local layer="$1"
    if [ "$layer" = "-1" ]; then
        echo "last"
    elif [[ "$layer" == -* ]]; then
        echo "neg${layer#-}"
    else
        echo "layer${layer}"
    fi
}

run_one() {
    local layer="$1"
    local gpu_id="$2"
    local label out_dir inner_script
    label="$(layer_label "$layer")"
    out_dir="$OUTPUT_ROOT/$label"
    inner_script="$OUTPUT_ROOT/logs/run_${label}_inner.sh"

    if [ -d "$out_dir" ] && find "$out_dir" -mindepth 1 -maxdepth 1 | grep -q .; then
        echo "ERROR: output dir is not empty for layer=$layer: $out_dir" >&2
        return 2
    fi
    mkdir -p "$out_dir"

    cat > "$inner_script" <<INNER
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
export PYTHONPATH="$SCRIPT_DIR/external/token-distillation:\${PYTHONPATH:-}"
max_selected_args=()
if [ "$MAX_SELECTED_TOKENS" != "0" ] && [ "$MAX_SELECTED_TOKENS" != "all" ]; then
    max_selected_args=(--max-selected-tokens "$MAX_SELECTED_TOKENS")
fi
dry_run_args=()
if [ "$DRY_RUN" = "1" ]; then
    dry_run_args=(--dry-run)
fi
python3 "$SCRIPT_DIR/train_retok_td.py" \\
  --student-model "$STUDENT_MODEL" \\
  --base-tokenizer "$BASE_TOKENIZER" \\
  --student-tokenizer "$STUDENT_TOKENIZER" \\
  --coverage-jsonl "$COVERAGE_JSONL" \\
  --snippets-jsonl "$SNIPPETS_JSONL" \\
  --token-ids-file "$TOKEN_IDS_FILE" \\
  --output-dir "$out_dir" \\
  --target-layer "$layer" \\
  --snippets-per-token "$SNIPPETS_PER_TOKEN" \\
  --min-accepted-snippets-per-token "$MIN_ACCEPTED_SNIPPETS_PER_TOKEN" \\
  --min-trained-token-fraction "$MIN_TRAINED_TOKEN_FRACTION" \\
  --epochs "$EPOCHS" \\
  --batch-size "$BATCH_SIZE" \\
  --learning-rate "$LEARNING_RATE" \\
  --dtype bfloat16 \\
  --device cuda \\
  "\${max_selected_args[@]}" "\${dry_run_args[@]}"
INNER

    (
        set -euo pipefail
        echo "=== layer=$layer label=$label gpu=$gpu_id ==="
        date -u
        echo "OUTPUT_DIR: $out_dir"
        CUDA_VISIBLE_DEVICES="$gpu_id" uenv start "$UENV_IMAGE" --view=default --ignore-tty < "$inner_script"
        echo "=== layer complete: $layer ==="
        date -u
    ) >"$OUTPUT_ROOT/logs/${label}.out" 2>"$OUTPUT_ROOT/logs/${label}.err"
}

pids=()
gpu_id=0
for layer in "${LAYERS[@]}"; do
    echo "Launching target_layer=$layer on GPU $gpu_id"
    run_one "$layer" "$gpu_id" &
    pids+=("$!")
    gpu_id=$((gpu_id + 1))
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo
if [ "$status" -ne 0 ]; then
    echo "ERROR: one or more TD layer candidates failed" >&2
    exit "$status"
fi

python3 - "$OUTPUT_ROOT" "${LAYERS[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
layers = sys.argv[2:]

def label(layer: str) -> str:
    if layer == "-1":
        return "last"
    if layer.startswith("-"):
        return "neg" + layer[1:]
    return "layer" + layer

manifest = {
    "output_root": str(root),
    "target_layers": layers,
    "completed": True,
    "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "runs": {},
}
for layer in layers:
    run_dir = root / label(layer)
    run_manifest = run_dir / "retok_td_manifest.json"
    entry = {"output_dir": str(run_dir), "manifest": str(run_manifest)}
    if run_manifest.exists():
        data = json.loads(run_manifest.read_text())
        entry.update({
            "requested_token_count": data.get("requested_token_count"),
            "trained_token_count": data.get("trained_token_count"),
            "skipped_token_count": data.get("skipped_token_count"),
            "trained_token_fraction": data.get("trained_token_fraction"),
            "completed": data.get("completed"),
            "elapsed_seconds": data.get("elapsed_seconds"),
        })
    manifest["runs"][layer] = entry

(root / "layer_pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo
echo "=== packed TD layer pilot complete ==="
date -u
echo "output root: $OUTPUT_ROOT"

#!/usr/bin/env bash
#SBATCH --job-name=td_pilot_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=216
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err
#
# Run intrinsic TD-pilot evals for ReTok, TD-last, and TD-layer11 inside one
# full-node allocation. Clariden normal jobs allocate a whole 4-GPU node even
# for a one-GPU request, so pack these independent single-GPU evals together.

set -euo pipefail

EVAL_JSONL="${EVAL_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_SPEC_TSV="${MODEL_SPEC_TSV:-}"

MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_DOCS="${MAX_DOCS:-}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SKIP_GREEDY="${SKIP_GREEDY:-0}"
GREEDY_MAX_NEW_TOKENS="${GREEDY_MAX_NEW_TOKENS:-100}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ -n "${SCRIPT_DIR_OVERRIDE:-}" ]; then
    SCRIPT_DIR="$(cd "$SCRIPT_DIR_OVERRIDE" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/compute_tokenizer_fair_metrics.py" ]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs" /capstor/scratch/cscs/fffoivos/runs/token_distillation

DEFAULT_SPEC="$OUTPUT_ROOT/model_spec.tsv"
if [ -z "$MODEL_SPEC_TSV" ]; then
    cat > "$DEFAULT_SPEC" <<'SPEC'
retok	/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok
td_last	/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_layer_pilot_20260523T082252Z/last
td_layer11	/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_layer_pilot_20260523T082252Z/layer11
SPEC
    MODEL_SPEC_TSV="$DEFAULT_SPEC"
fi

echo "=== run_td_pilot_intrinsics_packed.sbatch ==="
date -u
echo "host:             $(hostname)"
echo "SCRIPT_DIR:       $SCRIPT_DIR"
echo "EVAL_JSONL:       $EVAL_JSONL"
echo "OUTPUT_ROOT:      $OUTPUT_ROOT"
echo "MODEL_SPEC_TSV:   $MODEL_SPEC_TSV"
echo "MAX_CONTEXT:      $MAX_CONTEXT"
echo "MAX_DOCS:         ${MAX_DOCS:-none}"
echo "DTYPE:            $DTYPE"
echo "SKIP_GREEDY:      $SKIP_GREEDY"
echo "GREEDY_MAX_NEW_TOKENS: $GREEDY_MAX_NEW_TOKENS"
echo "PYTORCH_ALLOC_CONF: $PYTORCH_ALLOC_CONF"
echo "CUDA_VISIBLE_DEVICES at job start: ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

if [ ! -f "$EVAL_JSONL" ]; then
    echo "ERROR: missing EVAL_JSONL: $EVAL_JSONL" >&2
    exit 2
fi
if [ ! -f "$MODEL_SPEC_TSV" ]; then
    echo "ERROR: missing MODEL_SPEC_TSV: $MODEL_SPEC_TSV" >&2
    exit 2
fi

mapfile -t spec_rows < <(grep -v '^[[:space:]]*$' "$MODEL_SPEC_TSV")
if [ "${#spec_rows[@]}" -eq 0 ]; then
    echo "ERROR: MODEL_SPEC_TSV has no rows" >&2
    exit 2
fi
if [ "${#spec_rows[@]}" -gt 4 ]; then
    echo "ERROR: packed TD intrinsic eval supports at most 4 rows, got ${#spec_rows[@]}" >&2
    exit 2
fi

run_one() {
    local arm="$1"
    local model_path="$2"
    local gpu_id="$3"
    local out_dir inner_script
    out_dir="$OUTPUT_ROOT/$arm"
    inner_script="$OUTPUT_ROOT/logs/run_${arm}_inner.sh"

    mkdir -p "$out_dir" "$out_dir/cache"/{hf_home,hf_datasets,xdg,tmp}

    cat > "$out_dir/run_metadata.json" <<META
{
  "arm": "$arm",
  "model_path": "$model_path",
  "eval_jsonl": "$EVAL_JSONL",
  "max_context": $MAX_CONTEXT,
  "max_docs": ${MAX_DOCS:-null},
  "dtype": "$DTYPE",
  "skip_greedy": "$SKIP_GREEDY",
  "greedy_max_new_tokens": $GREEDY_MAX_NEW_TOKENS,
  "uenv_image": "$UENV_IMAGE",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "host": "$(hostname)",
  "packed_gpu_id": "$gpu_id",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

    cat > "$inner_script" <<INNER
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$gpu_id"
export HF_HOME="$out_dir/cache/hf_home"
export HF_DATASETS_CACHE="$out_dir/cache/hf_datasets"
export XDG_CACHE_HOME="$out_dir/cache/xdg"
export TMPDIR="$out_dir/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-24}"
export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
max_docs_args=()
if [ -n "$MAX_DOCS" ]; then
    max_docs_args=(--max-docs "$MAX_DOCS")
fi
skip_greedy_args=()
if [ "$SKIP_GREEDY" = "1" ]; then
    skip_greedy_args=(--skip-greedy)
fi
python3 "$SCRIPT_DIR/compute_tokenizer_fair_metrics.py" \\
  --model-path "$model_path" \\
  --eval-jsonl "$EVAL_JSONL" \\
  --output-json "$out_dir/tokenizer_fair_metrics.json" \\
  --max-context "$MAX_CONTEXT" \\
  --device "$DEVICE" \\
  --dtype "$DTYPE" \\
  "\${max_docs_args[@]}"
python3 "$SCRIPT_DIR/compute_new_token_diagnostics.py" \\
  --model-path "$model_path" \\
  --eval-jsonl "$EVAL_JSONL" \\
  --output-json "$out_dir/new_token_diagnostics.json" \\
  --new-id-range 131072 148480 \\
  --base-vocab-size 131072 \\
  --max-context "$MAX_CONTEXT" \\
  --device "$DEVICE" \\
  --dtype "$DTYPE" \\
  --greedy-max-new-tokens "$GREEDY_MAX_NEW_TOKENS" \\
  "\${max_docs_args[@]}" "\${skip_greedy_args[@]}"
INNER

    (
        set -euo pipefail
        echo "=== td pilot intrinsic eval arm=$arm gpu=$gpu_id ==="
        date -u
        echo "MODEL_PATH: $model_path"
        echo "OUT_DIR:    $out_dir"
        CUDA_VISIBLE_DEVICES="$gpu_id" uenv start "$UENV_IMAGE" --view=default --ignore-tty < "$inner_script"
        python3 - "$out_dir/run_metadata.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
data["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p.write_text(json.dumps(data, indent=2) + "\n")
PY
        echo "=== complete arm=$arm ==="
        date -u
    ) >"$OUTPUT_ROOT/logs/${arm}.out" 2>"$OUTPUT_ROOT/logs/${arm}.err"
}

pids=()
gpu_id=0
for row in "${spec_rows[@]}"; do
    IFS=$'\t' read -r arm model_path extra <<<"$row"
    if [ -n "${extra:-}" ] || [ -z "${arm:-}" ] || [ -z "${model_path:-}" ]; then
        echo "ERROR: bad spec row (expected 2 tab-separated fields): $row" >&2
        exit 2
    fi
    if [ ! -d "$model_path" ]; then
        echo "ERROR: model path for arm=$arm does not exist: $model_path" >&2
        exit 3
    fi
    echo "Launching arm=$arm on GPU $gpu_id"
    run_one "$arm" "$model_path" "$gpu_id" &
    pids+=("$!")
    gpu_id=$((gpu_id + 1))
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "ERROR: one or more TD intrinsic eval arms failed" >&2
    exit "$status"
fi

python3 "$SCRIPT_DIR/summarize_td_pilot_intrinsics.py" \
  --output-root "$OUTPUT_ROOT" \
  --eval-jsonl "$EVAL_JSONL"

echo
echo "=== packed TD pilot intrinsic eval complete ==="
date -u
echo "output root: $OUTPUT_ROOT"

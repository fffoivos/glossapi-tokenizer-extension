#!/usr/bin/env bash
#SBATCH --job-name=apertus_eval_packed
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=216
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Run several single-GPU lm-eval arms concurrently inside one full-node
# allocation. Clariden's normal partition bills/alllocates whole 4-GPU nodes
# even for --gpus-per-node=1 eval jobs, so three independent per-arm eval jobs
# waste 9 idle GPUs. This packed job uses GPU0/GPU1/GPU2 for the three bakeoff
# arms and leaves one spare GPU on the node.
#
# Required:
#   EVAL_SPEC_TSV  — tab-separated rows: arm<TAB>model_path<TAB>output_dir
#   TASK_GROUP     — full | retention_only | greek_only | safety_only
#
# Optional:
#   BATCH_SIZE     — default "auto"
#   LIMIT          — optional per-task sample cap
#   DTYPE          — default "bfloat16"
#   UENV_IMAGE     — default pytorch/v2.9.1:v2

set -euo pipefail

EVAL_SPEC_TSV="${EVAL_SPEC_TSV:?EVAL_SPEC_TSV is required}"
TASK_GROUP="${TASK_GROUP:-full}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT_VALUE="${LIMIT:-}"

LM_EVAL_PYTHONPATH="${LM_EVAL_PYTHONPATH:-/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-/iopsstor/scratch/cscs/fffoivos/tmp/eval_cache_${SLURM_JOB_ID:-manual}_packed}"
SHARE_EVAL_CACHE="${SHARE_EVAL_CACHE:-1}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"

declare -A TASK_LISTS
TASK_LISTS[retention]="arc_challenge,arc_easy,hellaswag,winogrande,piqa,mmlu,global_mmlu,xnli,xcopa"
TASK_LISTS[greek]="arc_challenge_mt_el,xnli_el,xquad_el,belebele_ell_Grek,global_mmlu_full_el,include_base_44_greek_few_shot_en,global_piqa_completions_ell_grek"
TASK_LISTS[safety]="harmbench,polyglotoxicityprompts"

case "$TASK_GROUP" in
    full)            TASK_LIST="${TASK_LISTS[retention]},${TASK_LISTS[greek]}" ;;
    retention_only)  TASK_LIST="${TASK_LISTS[retention]}" ;;
    greek_only)      TASK_LIST="${TASK_LISTS[greek]}" ;;
    safety_only)     TASK_LIST="${TASK_LISTS[safety]}" ;;
    *)               echo "ERROR: TASK_GROUP must be full|retention_only|greek_only|safety_only" >&2; exit 2 ;;
esac

if [ ! -f "$EVAL_SPEC_TSV" ]; then
    echo "ERROR: missing EVAL_SPEC_TSV: $EVAL_SPEC_TSV" >&2
    exit 2
fi

echo "=== run_eval_packed_arms.sbatch ==="; date -u
echo "host: $(hostname)"
echo "EVAL_SPEC_TSV: $EVAL_SPEC_TSV"
echo "TASK_GROUP: $TASK_GROUP -> $TASK_LIST"
echo "DTYPE: $DTYPE  BATCH: $BATCH_SIZE"
echo "EVAL_CACHE_ROOT: $EVAL_CACHE_ROOT"
echo "SHARE_EVAL_CACHE: $SHARE_EVAL_CACHE"
echo "CUDA_VISIBLE_DEVICES at job start: ${CUDA_VISIBLE_DEVICES:-unset}"
echo
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

mapfile -t spec_rows < <(grep -v '^[[:space:]]*$' "$EVAL_SPEC_TSV")
if [ "${#spec_rows[@]}" -eq 0 ]; then
    echo "ERROR: EVAL_SPEC_TSV has no rows" >&2
    exit 2
fi
if [ "${#spec_rows[@]}" -gt 4 ]; then
    echo "ERROR: packed eval supports at most 4 rows, got ${#spec_rows[@]}" >&2
    exit 2
fi

run_one() {
    local arm="$1"
    local model_path="$2"
    local output_dir="$3"
    local gpu_id="$4"
    local cache_dir
    if [ "$SHARE_EVAL_CACHE" = "1" ]; then
        cache_dir="$EVAL_CACHE_ROOT"
    else
        cache_dir="$EVAL_CACHE_ROOT/$arm"
    fi

    mkdir -p "$output_dir" "$cache_dir"/{hf_home,hf_datasets,xdg,tmp}

    cat > "$output_dir/run_metadata.json" <<META
{
  "model_path": "$model_path",
  "task_group": "$TASK_GROUP",
  "task_list": "$TASK_LIST",
  "batch_size": "$BATCH_SIZE",
  "dtype": "$DTYPE",
  "lm_eval_pythonpath": "$LM_EVAL_PYTHONPATH",
  "eval_cache_root": "$cache_dir",
  "share_eval_cache": "$SHARE_EVAL_CACHE",
  "lm_eval_source": "swiss-ai/lm-evaluation-harness (target-install)",
  "uenv_image": "$UENV_IMAGE",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "host": "$(hostname)",
  "packed_eval": true,
  "packed_gpu_id": "$gpu_id",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

    (
        set -euo pipefail
        cd "$output_dir"
        inner_script="$output_dir/packed_eval_${arm}_inner.sh"
        echo "=== packed eval arm=$arm gpu=$gpu_id ==="
        date -u
        echo "MODEL_PATH: $model_path"
        echo "OUTPUT_DIR: $output_dir"
        echo "CACHE: $cache_dir"

        cat > "$inner_script" <<INNER
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH="$LM_EVAL_PYTHONPATH"
export HF_HOME="$cache_dir/hf_home"
export HF_DATASETS_CACHE="$cache_dir/hf_datasets"
export XDG_CACHE_HOME="$cache_dir/xdg"
export TMPDIR="$cache_dir/tmp"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
limit_args=()
if [ -n "$LIMIT_VALUE" ]; then
    limit_args=(--limit "$LIMIT_VALUE")
fi
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True" \
    --tasks "$TASK_LIST" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$output_dir/results.json" \
    --log_samples \
    "\${limit_args[@]}"
INNER
        CUDA_VISIBLE_DEVICES="$gpu_id" uenv start "$UENV_IMAGE" --view=default --ignore-tty < "$inner_script"

        python3 - "$output_dir/run_metadata.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

p = Path(sys.argv[1])
md = json.loads(p.read_text())
md["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p.write_text(json.dumps(md, indent=2))
print("updated run_metadata.json with end_time")
PY
        echo "=== packed eval complete arm=$arm ==="
        date -u
        ls -la "$output_dir" || true
    ) >"$output_dir/packed_eval_${arm}.out" 2>"$output_dir/packed_eval_${arm}.err"
}

pids=()
gpu_id=0
for row in "${spec_rows[@]}"; do
    IFS=$'\t' read -r arm model_path output_dir extra <<<"$row"
    if [ -n "${extra:-}" ] || [ -z "${arm:-}" ] || [ -z "${model_path:-}" ] || [ -z "${output_dir:-}" ]; then
        echo "ERROR: bad spec row (expected 3 tab-separated fields): $row" >&2
        exit 2
    fi
    if [ ! -d "$model_path" ]; then
        echo "ERROR: model path for arm=$arm does not exist: $model_path" >&2
        exit 3
    fi
    echo "Launching arm=$arm on GPU $gpu_id"
    run_one "$arm" "$model_path" "$output_dir" "$gpu_id" &
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
    echo "ERROR: one or more packed eval arms failed" >&2
    exit "$status"
fi
echo "=== packed eval complete for all arms ==="
date -u

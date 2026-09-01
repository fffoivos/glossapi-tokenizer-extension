#!/usr/bin/env bash
#SBATCH --job-name=poly_cut_probe
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=216
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/%x-%j.err

set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024}"
CODE_ROOT="${CODE_ROOT:-$RUN_ROOT/code}"
PROBE_CODE="$CODE_ROOT/polytonic_cutoff_probe"
TD_CODE="$CODE_ROOT/token_distillation"
EVAL_CODE="$CODE_ROOT/eval"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
UENV_BIN="${UENV_BIN:-/usr/bin/uenv}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/users/fffoivos/.venvs/hfupload/bin/python}"

BASE_MODEL="$RUN_ROOT/assets/apertus-tokenizer-extension/experiment-checkpoints/TokenDistil-3.5B"
PROBE_DATA="$RUN_ROOT/probe_data"
RESULTS="$RUN_ROOT/results"
mkdir -p "$RESULTS/logs"

test -f "$RUN_ROOT/asset_manifest.json"
test -f "$PROBE_CODE/build_incremental_checkpoint.py"
test -f "$PROBE_CODE/aggregate_probe.py"
test -f "$TD_CODE/train_retok_td.py"
test -f "$EVAL_CODE/compute_tokenizer_fair_metrics.py"

echo "=== polytonic cutoff model probe ==="
date -u
echo "host:       $(hostname)"
echo "RUN_ROOT:   $RUN_ROOT"
echo "BASE_MODEL: $BASE_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

write_eval_commands() {
    local model_path="$1"
    local output_dir="$2"
    cat <<INNER
python3 '$EVAL_CODE/compute_tokenizer_fair_metrics.py' \
  --model-path '$model_path' \
  --eval-jsonl '$PROBE_DATA/ancient_test.jsonl' \
  --output-json '$output_dir/ancient_metrics.json' \
  --max-context 4096 --max-docs 1000 --device cuda --dtype bfloat16
python3 '$EVAL_CODE/compute_tokenizer_fair_metrics.py' \
  --model-path '$model_path' \
  --eval-jsonl '$PROBE_DATA/modern_test.jsonl' \
  --output-json '$output_dir/modern_metrics.json' \
  --max-context 4096 --max-docs 1000 --device cuda --dtype bfloat16
INNER
}

run_candidate() {
    local cutoff="$1"
    local end_id="$2"
    local gpu_id="$3"
    local label tokenizer init_dir td_dir out_dir inner
    label="$(printf '%04d' "$cutoff")"
    tokenizer="$RUN_ROOT/candidates/c3p_poly_added_${label}"
    init_dir="$RESULTS/init_${label}"
    td_dir="$RESULTS/td_${label}"
    out_dir="$RESULTS/cutoff_${label}"
    inner="$RESULTS/logs/cutoff_${label}_inner.sh"
    mkdir -p "$out_dir"

    cat > "$inner" <<INNER
set -euo pipefail
export CUDA_VISIBLE_DEVICES='$gpu_id'
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='$TD_CODE/external/token-distillation':\${PYTHONPATH:-}
python3 '$PROBE_CODE/build_incremental_checkpoint.py' \
  --base-model '$BASE_MODEL' \
  --candidate-tokenizer '$tokenizer' \
  --output-dir '$init_dir' \
  --expected-base-vocab 148480 \
  --modern-id-start 131072 \
  --device cuda
python3 '$TD_CODE/train_retok_td.py' \
  --student-model '$init_dir' \
  --base-tokenizer '$BASE_MODEL' \
  --student-tokenizer '$tokenizer' \
  --coverage-jsonl '$RUN_ROOT/coverage_${label}/td_coverage_prepass.jsonl' \
  --snippets-jsonl '$RUN_ROOT/coverage_${label}/td_snippet_index/snippets.jsonl' \
  --token-ids-file '$RUN_ROOT/train_token_ids_${label}.txt' \
  --output-dir '$td_dir' \
  --base-vocab-size 148480 \
  --new-id-start 148480 \
  --new-id-end '$end_id' \
  --snippets-per-token 25 \
  --min-accepted-snippets-per-token 25 \
  --min-trained-token-fraction 0.90 \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --target-layer 11 \
  --dtype bfloat16 \
  --device cuda
$(write_eval_commands "$td_dir" "$out_dir")
INNER
    (
        set -euo pipefail
        echo "starting cutoff=$cutoff on GPU $gpu_id"
        date -u
        CUDA_VISIBLE_DEVICES="$gpu_id" "$UENV_BIN" start "$UENV_IMAGE" --view=default --ignore-tty < "$inner"
        date -u
        echo "complete cutoff=$cutoff"
    ) >"$RESULTS/logs/cutoff_${label}.out" 2>"$RESULTS/logs/cutoff_${label}.err"
}

run_baseline() {
    local gpu_id="$1"
    local out_dir="$RESULTS/baseline"
    local inner="$RESULTS/logs/baseline_inner.sh"
    mkdir -p "$out_dir"
    {
        echo "set -euo pipefail"
        echo "export CUDA_VISIBLE_DEVICES='$gpu_id'"
        echo "export TOKENIZERS_PARALLELISM=false"
        echo "export PYTORCH_ALLOC_CONF=expandable_segments:True"
        write_eval_commands "$BASE_MODEL" "$out_dir"
    } > "$inner"
    (
        set -euo pipefail
        echo "starting baseline on GPU $gpu_id"
        date -u
        CUDA_VISIBLE_DEVICES="$gpu_id" "$UENV_BIN" start "$UENV_IMAGE" --view=default --ignore-tty < "$inner"
        date -u
        echo "complete baseline"
    ) >"$RESULTS/logs/baseline.out" 2>"$RESULTS/logs/baseline.err"
}

run_candidate 512 148992 0 &
pid_512=$!
run_candidate 1024 149504 1 &
pid_1024=$!
run_baseline 2 &
pid_baseline=$!

status=0
for pid in "$pid_512" "$pid_1024" "$pid_baseline"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "one or more packed probe arms failed; inspect $RESULTS/logs" >&2
    exit "$status"
fi

"$CONTROL_PYTHON" "$PROBE_CODE/aggregate_probe.py" \
  --baseline-ancient "$RESULTS/baseline/ancient_metrics.json" \
  --baseline-modern "$RESULTS/baseline/modern_metrics.json" \
  --cutoff-512-ancient "$RESULTS/cutoff_0512/ancient_metrics.json" \
  --cutoff-512-modern "$RESULTS/cutoff_0512/modern_metrics.json" \
  --cutoff-1024-ancient "$RESULTS/cutoff_1024/ancient_metrics.json" \
  --cutoff-1024-modern "$RESULTS/cutoff_1024/modern_metrics.json" \
  --tokenizer-512 "$RUN_ROOT/candidates/c3p_poly_added_0512" \
  --tokenizer-1024 "$RUN_ROOT/candidates/c3p_poly_added_1024" \
  --output-json "$RESULTS/selection.json" \
  --modern-max-regression 0.005 \
  --min-ancient-relative-gain-for-1024 0.01

date -u
echo "probe complete: $RESULTS/selection.json"

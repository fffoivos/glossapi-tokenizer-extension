#!/usr/bin/env bash
#SBATCH --job-name=poly_cut_cal_probe
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=216
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/%x-%j.err

set -euo pipefail
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024}"
CODE_ROOT="$RUN_ROOT/code"
PROBE_CODE="$CODE_ROOT/polytonic_cutoff_probe"
EVAL_CODE="$CODE_ROOT/eval"
RESULTS="$RUN_ROOT/results"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
UENV_BIN="${UENV_BIN:-/usr/bin/uenv}"
CONTROL_PYTHON="${CONTROL_PYTHON:-/users/fffoivos/.venvs/hfupload/bin/python}"
CALIBRATION_JSONL="$RUN_ROOT/probe_data/output_calibration.jsonl"

test -s "$CALIBRATION_JSONL"
mkdir -p "$RESULTS/logs"

run_arm() {
    local cutoff="$1"
    local vocab_size="$2"
    local gpu_id="$3"
    local label input_model calibrated output_dir inner
    label="$(printf '%04d' "$cutoff")"
    input_model="$RESULTS/td_${label}"
    calibrated="$RESULTS/calibrated_${label}"
    output_dir="$RESULTS/cutoff_${label}_calibrated"
    inner="$RESULTS/logs/calibrated_${label}_inner.sh"
    mkdir -p "$output_dir"
    cat > "$inner" <<INNER
set -euo pipefail
export CUDA_VISIBLE_DEVICES='$gpu_id'
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
python3 '$PROBE_CODE/calibrate_new_output_rows.py' \
  --model '$input_model' \
  --calibration-jsonl '$CALIBRATION_JSONL' \
  --output-dir '$calibrated' \
  --base-vocab-size 148480 \
  --new-vocab-size '$vocab_size' \
  --steps 400 \
  --seq-length 512 \
  --learning-rate 2e-4 \
  --max-grad-norm 1.0 \
  --device cuda
python3 '$EVAL_CODE/compute_tokenizer_fair_metrics.py' \
  --model-path '$calibrated' \
  --eval-jsonl '$RUN_ROOT/probe_data/ancient_test.jsonl' \
  --output-json '$output_dir/ancient_metrics.json' \
  --max-context 4096 --max-docs 1000 --device cuda --dtype bfloat16
python3 '$EVAL_CODE/compute_tokenizer_fair_metrics.py' \
  --model-path '$calibrated' \
  --eval-jsonl '$RUN_ROOT/probe_data/modern_test.jsonl' \
  --output-json '$output_dir/modern_metrics.json' \
  --max-context 4096 --max-docs 1000 --device cuda --dtype bfloat16
INNER
    CUDA_VISIBLE_DEVICES="$gpu_id" "$UENV_BIN" start "$UENV_IMAGE" \
      --view=default --ignore-tty < "$inner" \
      >"$RESULTS/logs/calibrated_${label}.out" \
      2>"$RESULTS/logs/calibrated_${label}.err"
}

run_arm 512 148992 0 &
pid_512=$!
run_arm 1024 149504 1 &
pid_1024=$!
status=0
for pid in "$pid_512" "$pid_1024"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [ "$status" -ne 0 ]; then
    echo "calibrated arm failed" >&2
    exit "$status"
fi

"$CONTROL_PYTHON" "$PROBE_CODE/aggregate_probe.py" \
  --baseline-ancient "$RESULTS/baseline/ancient_metrics.json" \
  --baseline-modern "$RESULTS/baseline/modern_metrics.json" \
  --cutoff-512-ancient "$RESULTS/cutoff_0512_calibrated/ancient_metrics.json" \
  --cutoff-512-modern "$RESULTS/cutoff_0512_calibrated/modern_metrics.json" \
  --cutoff-1024-ancient "$RESULTS/cutoff_1024_calibrated/ancient_metrics.json" \
  --cutoff-1024-modern "$RESULTS/cutoff_1024_calibrated/modern_metrics.json" \
  --tokenizer-512 "$RUN_ROOT/candidates/c3p_poly_added_0512" \
  --tokenizer-1024 "$RUN_ROOT/candidates/c3p_poly_added_1024" \
  --output-json "$RESULTS/selection_calibrated.json" \
  --modern-max-regression 0.005 \
  --min-ancient-relative-gain-for-1024 0.01

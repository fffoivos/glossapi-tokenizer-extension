#!/usr/bin/env bash
#SBATCH --job-name=full8-dtype-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${EVAL_CODE_ROOT:?set immutable evaluation code root}"
: "${EVAL_CODE_BUNDLE_RECEIPT:?set immutable evaluation code receipt}"
: "${EVAL_CONTRACT:?set frozen contract}"
: "${EVAL_MANIFEST:?set frozen example manifest}"
: "${EVAL_OUTPUT_ROOT:?set new output root}"
: "${EVAL_VENV:?set evaluation venv}"
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $EVAL_OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_OUTPUT_ROOT/logs"

eval_root="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$eval_root/verify_eval_bundle.py" --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_BUNDLE_RECEIPT"
model=$(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$EVAL_CONTRACT" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))['checkpoint_scope'][0]
print(f'{row["label"]}={row["model_path"]}')
PY
)

pids=()
for dtype in float32 bfloat16; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=220G \
    --gpus-per-node=1 --gpus-per-task=1 \
    --cpu-bind=cores --output="$EVAL_OUTPUT_ROOT/logs/$dtype.out" --error="$EVAL_OUTPUT_ROOT/logs/$dtype.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$eval_root/run_checkpoint_suite.py" \
      --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
      --model "$model" --output-dir "$EVAL_OUTPUT_ROOT/$dtype" --dtype "$dtype" \
      --candidate-batch-size 16 --example-batch-size 16 --max-examples-per-benchmark 128 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$eval_root/compare_dtype_parity.py" \
  --float32 "$EVAL_OUTPUT_ROOT/float32/predictions.jsonl" \
  --bfloat16 "$EVAL_OUTPUT_ROOT/bfloat16/predictions.jsonl" \
  --output "$EVAL_OUTPUT_ROOT/parity_receipt.json"

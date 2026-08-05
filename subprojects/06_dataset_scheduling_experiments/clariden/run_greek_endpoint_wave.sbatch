#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-greek-endpoints
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RUN_ROOT:?set campaign run root}"
uenv run pytorch/v2.9.1:v2 --view=default -- test -x "$EVAL_VENV/bin/python" || {
  echo "missing endpoint evaluation Python inside uenv: $EVAL_VENV/bin/python" >&2
  exit 2
}
if [[ -n "${RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT:-}" ]]; then
  "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/verify_code_bundle.py" \
    --root "$SCIENTIFIC_BUNDLE" --receipt "$RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT" \
    --kind scientific
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/greek_endpoint_benchmarks}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to replace $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"
wave="$OUTPUT_ROOT/wave_manifest.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/build_endpoint_wave_manifest.py" \
  --run-root "$RUN_ROOT" --output-root "$OUTPUT_ROOT/tasks" --output "$wave"

mapfile -t assets < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); a=c["assets"]
print(a["native_greek_eval_root"])
print(a["python_compat_dir"])
PY
)
native_runner="${assets[0]}/run_native_greek_mcq_eval.py"
compat="${assets[1]}"
contract="$SCIENTIFIC_BUNDLE/evaluation/endpoint_benchmark_contract.json"
runner="$SCIENTIFIC_BUNDLE/evaluation/run_greek_endpoint_mcq_eval.py"

mapfile -t rows < <("$HOST_PYTHON" - "$wave" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["tasks"]:
    fields=(row["arm_id"],row["model_path"],row["output_root"])
    if any(any(c in value for c in "\t\r\n") for value in fields): raise SystemExit("unsafe wave field")
    print("\t".join(fields))
PY
)
for ((base=0; base<${#rows[@]}; base+=4)); do
  pids=()
  for ((offset=0; offset<4 && base+offset<${#rows[@]}; offset++)); do
    IFS=$'\t' read -r arm model output <<<"${rows[base+offset]}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --gres=gpu:1 \
      --cpus-per-task=64 --mem=105G --cpu-bind=cores \
      --output="$OUTPUT_ROOT/logs/${arm}.out" --error="$OUTPUT_ROOT/logs/${arm}.err" \
      uenv run pytorch/v2.9.1:v2 --view=default -- \
      env PYTHONPATH="$compat" "$EVAL_VENV/bin/python" "$runner" --contract "$contract" \
        --native-runner "$native_runner" --model "$model" --model-label "$arm" \
        --output-dir "$output" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  ((status == 0)) || { echo "Greek endpoint benchmark lane failed" >&2; exit 1; }
done
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_endpoint_wave.py" \
  --wave-manifest "$wave" --output "$OUTPUT_ROOT/endpoint_wave_receipt.json"

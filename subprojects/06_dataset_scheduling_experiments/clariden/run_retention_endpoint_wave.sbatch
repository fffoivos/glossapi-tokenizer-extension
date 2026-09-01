#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-retention
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=08:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RUN_ROOT:?set campaign run root}"
OUTPUT_ROOT=${OUTPUT_ROOT:-$RUN_ROOT/retention_endpoint_benchmarks}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to replace $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT/logs"
wave="$OUTPUT_ROOT/wave_manifest.json"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/build_endpoint_wave_manifest.py" \
  --run-root "$RUN_ROOT" --output-root "$OUTPUT_ROOT/tasks" --output "$wave"
mapfile -t assets < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["assets"]
print(a["lm_eval_root"])
print(a["python_compat_dir"])
PY
)
lm_eval_root=${assets[0]}; compat=${assets[1]}
worker="$SCIENTIFIC_BUNDLE/clariden/run_retention_endpoint_one.sh"
if [[ -n "${RETENTION_SHARED_CACHE_ROOT:-}" ]]; then
  : "${RETENTION_DATASET_CACHE_RECEIPT:?set frozen retention dataset-cache receipt}"
  uenv run pytorch/v2.9.1:v2 --view=default -- \
    env PYTHONPATH="$compat:$lm_eval_root" \
      LD_LIBRARY_PATH="$lm_eval_root/scipy.libs:$lm_eval_root/numpy.libs:$lm_eval_root/scikit_learn.libs:${LD_LIBRARY_PATH:-}" \
      HF_HOME="$RETENTION_SHARED_CACHE_ROOT/hf_home" \
      HF_DATASETS_CACHE="$RETENTION_SHARED_CACHE_ROOT/hf_datasets" \
      XDG_CACHE_HOME="$RETENTION_SHARED_CACHE_ROOT/xdg" \
      TMPDIR="$RETENTION_SHARED_CACHE_ROOT/tmp" \
      HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      python3 "$SCIENTIFIC_BUNDLE/evaluation/freeze_retention_dataset_cache.py" \
        --cache-root "$RETENTION_SHARED_CACHE_ROOT" \
        --expected-receipt "$RETENTION_DATASET_CACHE_RECEIPT"
fi
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
    IFS=$'\t' read -r label model output <<<"${rows[base+offset]}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --gres=gpu:1 \
      --cpus-per-task=64 --mem=105G --cpu-bind=cores \
      --output="$OUTPUT_ROOT/logs/${label}.out" --error="$OUTPUT_ROOT/logs/${label}.err" \
      uenv run pytorch/v2.9.1:v2 --view=default -- \
      env PYTHONPATH="$compat:$lm_eval_root" MODEL_PATH="$model" MODEL_LABEL="$label" \
        OUTPUT_DIR="$output" LM_EVAL_ROOT="$lm_eval_root" \
        RETENTION_SHARED_CACHE_ROOT="${RETENTION_SHARED_CACHE_ROOT:-}" bash "$worker" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  ((status == 0)) || { echo "retention endpoint lane failed" >&2; exit 1; }
done
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_retention_endpoint_wave.py" \
  --wave-manifest "$wave" --output "$OUTPUT_ROOT/retention_wave_receipt.json"

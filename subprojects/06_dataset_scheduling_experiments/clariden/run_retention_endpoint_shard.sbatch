#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-retention-shard
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=03:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT:?set recovery bundle receipt}"
: "${WAVE_MANIFEST:?set prebuilt endpoint wave manifest}"
: "${SHARD_ARMS:?set comma-separated disjoint model labels}"
: "${SHARD_RECEIPT:?set unique shard receipt path}"
: "${RETENTION_SHARED_CACHE_ROOT:?set frozen shared cache root}"
: "${RETENTION_DATASET_CACHE_RECEIPT:?set frozen cache receipt}"

"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/verify_code_bundle.py" \
  --root "$SCIENTIFIC_BUNDLE" --receipt "$RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT" \
  --kind scientific

mapfile -t assets < <("$HOST_PYTHON" - "$CAMPAIGN_MANIFEST" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))["assets"]
print(a["lm_eval_root"])
print(a["python_compat_dir"])
PY
)
lm_eval_root=${assets[0]}; compat=${assets[1]}

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

mapfile -t rows < <("$HOST_PYTHON" - "$WAVE_MANIFEST" "$SHARD_ARMS" <<'PY'
import json,sys
wave=json.load(open(sys.argv[1])); requested=sys.argv[2].split(",")
if len(requested) != len(set(requested)) or not 1 <= len(requested) <= 4:
    raise SystemExit("invalid shard arm inventory")
tasks={row["arm_id"]:row for row in wave["tasks"]}
if set(requested) - set(tasks): raise SystemExit("unknown shard arm")
for arm in requested:
    row=tasks[arm]; fields=(arm,row["model_path"],row["output_root"])
    if any(any(c in value for c in "\t\r\n") for value in fields):
        raise SystemExit("unsafe wave field")
    print("\t".join(fields))
PY
)
worker="$SCIENTIFIC_BUNDLE/clariden/run_retention_endpoint_one.sh"
pids=()
for row in "${rows[@]}"; do
  IFS=$'\t' read -r label model output <<<"$row"
  [[ ! -e "$output" ]] || { echo "refusing to replace $output" >&2; exit 2; }
  log_root=$(dirname "$(dirname "$output")")/logs
  mkdir -p "$log_root"
  srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --gres=gpu:1 \
    --cpus-per-task=64 --mem=105G --cpu-bind=cores \
    --output="$log_root/${label}.out" --error="$log_root/${label}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
    env PYTHONPATH="$compat:$lm_eval_root" MODEL_PATH="$model" MODEL_LABEL="$label" \
      OUTPUT_DIR="$output" LM_EVAL_ROOT="$lm_eval_root" \
      RETENTION_SHARED_CACHE_ROOT="$RETENTION_SHARED_CACHE_ROOT" bash "$worker" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || { echo "retention endpoint shard lane failed" >&2; exit 1; }

shard_args=()
IFS=',' read -ra shard_arms <<<"$SHARD_ARMS"
for arm in "${shard_arms[@]}"; do shard_args+=(--arm "$arm"); done
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/finalize_retention_endpoint_shard.py" \
  --wave-manifest "$WAVE_MANIFEST" "${shard_args[@]}" --output "$SHARD_RECEIPT"

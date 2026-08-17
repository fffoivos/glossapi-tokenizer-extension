#!/usr/bin/env bash
# Score one frozen GreekMMLU panel on one four-GPU debug node.
# Clariden debug-qos caps jobs at 90 total node-minutes. The fixed sixteen
# shards therefore run in four sequential four-GPU groups without changing the
# questions, shard partition, scorer, model, or aggregation.
#SBATCH --job-name=h2g-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_SCALE H2G_ITERATION \
  H2G_CHECKPOINT_EXPORT H2G_GREEKMMLU_MODE H2G_GREEKMMLU_CLEAN_EXAMPLES \
  H2G_GREEKMMLU_SENTINEL_MANIFEST H2G_GREEKMMLU_OUTPUT EVAL_VENV; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "GreekMMLU evaluation must run on debug" >&2; exit 2;
}
observed_nodes="${SLURM_STEP_NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}}"
[[ "$observed_nodes" == 1 ]] || {
  echo "GreekMMLU evaluation requires one debug node" >&2; exit 2;
}
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "invalid scale" >&2; exit 2; }
[[ "$H2G_ITERATION" =~ ^[0-9]+$ ]] || { echo "invalid iteration" >&2; exit 2; }
[[ "$H2G_GREEKMMLU_MODE" == full_clean || "$H2G_GREEKMMLU_MODE" == sentinel_pair ]] || {
  echo "invalid GreekMMLU mode" >&2; exit 2;
}
[[ ! -e "$H2G_GREEKMMLU_OUTPUT" ]] || { echo "immutable GreekMMLU output exists" >&2; exit 2; }

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

readarray -t resolved < <(
  uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
    "$H2G_CHECKPOINT_EXPORT" "$H2G_SCALE" "$H2G_ITERATION" \
    "$H2G_GREEKMMLU_CLEAN_EXAMPLES" "$H2G_GREEKMMLU_SENTINEL_MANIFEST" \
    "$H2G_GREEKMMLU_MODE" <<'PY'
import hashlib, json, pathlib, sys
receipt_path, scale, iteration, clean_path, manifest_path, mode = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding="utf-8"))
if not (receipt.get("schema_version") == "apertus_hard_h_to_g_checkpoint_export_v1"
        and receipt.get("status") == "completed" and receipt.get("scale") == scale
        and int(receipt.get("iteration", -1)) == int(iteration)
        and receipt.get("ready_for_frozen_evaluators") is True):
    raise SystemExit("checkpoint export drift")
model = pathlib.Path(receipt["hf_export"]["path"]).resolve()
if not model.is_dir(): raise SystemExit("HF export missing")
manifest = json.load(open(manifest_path, encoding="utf-8"))
if not (manifest.get("schema_version") == "apertus_greekmmlu_sentinel_manifest_v1"
        and manifest.get("status") == "frozen" and manifest.get("sizes") == [4096, 8192]):
    raise SystemExit("sentinel manifest drift")
if mode == "full_clean":
    panel = pathlib.Path(clean_path).resolve()
else:
    binding = manifest["panels"]["8192"]
    panel = pathlib.Path(binding["path"]).resolve()
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    if panel.stat().st_size != int(binding["bytes"]) or digest != binding["sha256"]:
        raise SystemExit("sentinel panel binding drift")
print(model)
print(panel)
PY
)
[[ "${#resolved[@]}" == 2 ]] || { echo "failed to resolve GreekMMLU inputs" >&2; exit 2; }
export H2G_GREEKMMLU_MODEL="${resolved[0]}"
export H2G_GREEKMMLU_PANEL="${resolved[1]}"
export H2G_GREEKMMLU_LABEL="${H2G_SCALE}_iter_$(printf '%07d' "$H2G_ITERATION")"
export H2G_GREEKMMLU_NATIVE_RUNNER="$H2G_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
parent=$(dirname "$H2G_GREEKMMLU_OUTPUT")
base=$(basename "$H2G_GREEKMMLU_OUTPUT")
export H2G_GREEKMMLU_STAGING="$parent/.${base}.${SLURM_JOB_ID}.partial"
[[ ! -e "$H2G_GREEKMMLU_STAGING" ]] || { echo "partial GreekMMLU output exists" >&2; exit 2; }
mkdir -p "$H2G_GREEKMMLU_STAGING/shards" "$H2G_GREEKMMLU_STAGING/logs"
cleanup() {
  status=$?
  if [[ "$status" != 0 && -d "$H2G_GREEKMMLU_STAGING" ]]; then
    mv "$H2G_GREEKMMLU_STAGING" "$parent/_failed_${base}.${SLURM_JOB_ID}"
  fi
  exit "$status"
}
trap cleanup EXIT

for offset in 0 4 8 12; do
  export H2G_GREEKMMLU_SHARD_OFFSET="$offset"
  srun --exclusive --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 \
    --cpus-per-task=54 --mem-per-cpu=2900M --gpus-per-task=1 --cpu-bind=cores \
    --output="$H2G_GREEKMMLU_STAGING/logs/shard-${offset}-%t.out" \
    --error="$H2G_GREEKMMLU_STAGING/logs/shard-${offset}-%t.err" \
    bash "$subproject/clariden/run_greekmmlu_shard.sh"
done
unset H2G_GREEKMMLU_SHARD_OFFSET

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$subproject/evaluation/aggregate_frozen_greekmmlu.py" \
  --mode "$H2G_GREEKMMLU_MODE" \
  --clean-examples "$H2G_GREEKMMLU_CLEAN_EXAMPLES" \
  --sentinel-manifest "$H2G_GREEKMMLU_SENTINEL_MANIFEST" \
  --panel "$H2G_GREEKMMLU_PANEL" \
  --shards-root "$H2G_GREEKMMLU_STAGING/shards" --expected-shards 16 \
  --model "$H2G_GREEKMMLU_MODEL" --model-label "$H2G_GREEKMMLU_LABEL" \
  --scale "$H2G_SCALE" --iteration "$H2G_ITERATION" \
  --checkpoint-export "$H2G_CHECKPOINT_EXPORT" \
  --output-dir "$H2G_GREEKMMLU_STAGING/aggregate"
mv "$H2G_GREEKMMLU_STAGING" "$H2G_GREEKMMLU_OUTPUT"
trap - EXIT
printf '%s\n' "$H2G_GREEKMMLU_OUTPUT/aggregate/receipt.json"

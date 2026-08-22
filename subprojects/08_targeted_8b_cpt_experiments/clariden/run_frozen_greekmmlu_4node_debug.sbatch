#!/usr/bin/env bash
# Score one full-clean or nested-sentinel GreekMMLU checkpoint.
#SBATCH --job-name=h2g-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks=16
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
case "${SLURM_JOB_PARTITION:-}" in
  debug|normal) ;;
  *) echo "GreekMMLU evaluation requires a debug or held normal allocation" >&2; exit 2 ;;
esac
[[ "${SLURM_NNODES:-0}" == 4 ]] || {
  echo "GreekMMLU evaluation requires exactly four allocated nodes" >&2; exit 2;
}
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "invalid scale" >&2; exit 2; }
[[ "$H2G_ITERATION" =~ ^[0-9]+$ ]] || { echo "invalid iteration" >&2; exit 2; }
[[ "$H2G_GREEKMMLU_MODE" == full_clean || "$H2G_GREEKMMLU_MODE" == full_public || "$H2G_GREEKMMLU_MODE" == sentinel_pair ]] || {
  echo "invalid GreekMMLU mode" >&2; exit 2;
}
if [[ "$H2G_GREEKMMLU_MODE" == full_public ]]; then
  : "${H2G_GREEKMMLU_PUBLIC_EXAMPLES:?full-public mode requires its frozen examples}"
fi
[[ ! -e "$H2G_GREEKMMLU_OUTPUT" ]] || { echo "immutable GreekMMLU output exists" >&2; exit 2; }

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

readarray -t resolved < <(
  uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
    "$H2G_CHECKPOINT_EXPORT" "$H2G_SCALE" "$H2G_ITERATION" \
    "$H2G_GREEKMMLU_CLEAN_EXAMPLES" "$H2G_GREEKMMLU_SENTINEL_MANIFEST" \
    "$H2G_GREEKMMLU_MODE" "${H2G_GREEKMMLU_PUBLIC_EXAMPLES:-}" <<'PY'
import hashlib, json, pathlib, sys
receipt_path, scale, iteration, clean_path, manifest_path, mode, public_path = sys.argv[1:]
receipt = json.load(open(receipt_path, encoding="utf-8"))
canonical = (
    receipt.get("schema_version") == "apertus_hard_h_to_g_checkpoint_export_v1"
    and receipt.get("ready_for_frozen_evaluators") is True
)
trajectory = (
    receipt.get("schema_version") == "apertus_h2g_trajectory_checkpoint_export_v1"
    and receipt.get("ready_for_trajectory_evaluator") is True
    and receipt.get("exact_weight_mapping_passed") is True
    and receipt.get("scope") == "matched_cross_scale_greekmmlu_trajectory_only"
)
if not (receipt.get("status") == "completed" and receipt.get("scale") == scale
        and int(receipt.get("iteration", -1)) == int(iteration)
        and (canonical or trajectory)):
    raise SystemExit("checkpoint export drift")
model = pathlib.Path(receipt["hf_export"]["path"]).resolve()
if not model.is_dir(): raise SystemExit("HF export missing")
manifest = json.load(open(manifest_path, encoding="utf-8"))
if not (manifest.get("schema_version") == "apertus_greekmmlu_sentinel_manifest_v1"
        and manifest.get("status") == "frozen" and manifest.get("sizes") == [4096, 8192]):
    raise SystemExit("sentinel manifest drift")
if mode == "full_clean":
    panel = pathlib.Path(clean_path).resolve()
elif mode == "full_public":
    panel = pathlib.Path(public_path).resolve()
    payload = json.load(panel.open(encoding="utf-8"))
    if not (payload.get("schema_version") == "apertus_greekmmlu_public_examples_v1"
            and payload.get("status") == "frozen"
            and int(payload.get("public_count", -1)) == 16632):
        raise SystemExit("public GreekMMLU examples drift")
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
if [[ "$H2G_GREEKMMLU_MODE" == full_public ]]; then
  H2G_GREEKMMLU_CLEAN_EXAMPLES="$H2G_GREEKMMLU_PANEL"
fi
export H2G_GREEKMMLU_LABEL="${H2G_SCALE}_iter_$(printf '%07d' "$H2G_ITERATION")"
export H2G_GREEKMMLU_NATIVE_RUNNER="$H2G_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
parent=$(dirname "$H2G_GREEKMMLU_OUTPUT")
base=$(basename "$H2G_GREEKMMLU_OUTPUT")
# Write directly at the immutable final path.  Aggregate and shard receipts
# contain absolute paths, so renaming a staging directory after receipt
# creation makes an otherwise successful evaluation unverifiable.
export H2G_GREEKMMLU_STAGING="$H2G_GREEKMMLU_OUTPUT"
[[ ! -e "$H2G_GREEKMMLU_STAGING" ]] || { echo "GreekMMLU output exists" >&2; exit 2; }
mkdir -p "$H2G_GREEKMMLU_STAGING/shards" "$H2G_GREEKMMLU_STAGING/logs"
cleanup() {
  status=$?
  if [[ "$status" != 0 && -d "$H2G_GREEKMMLU_STAGING" ]]; then
    failed="$parent/_failed_${base}.${SLURM_JOB_ID}"
    [[ ! -e "$failed" ]] || failed="${failed}.$(date +%s)"
    mv "$H2G_GREEKMMLU_STAGING" "$failed"
  fi
  exit "$status"
}
trap cleanup EXIT

overlap=()
[[ -n "${SLURM_STEP_ID:-}" ]] && overlap=(--overlap)
srun "${overlap[@]}" --exclusive --exact --nodes=4 --ntasks=16 --ntasks-per-node=4 \
  --cpus-per-task=54 --mem-per-cpu=2900M --gpus-per-task=1 --cpu-bind=cores \
  --output="$H2G_GREEKMMLU_STAGING/logs/shard-%t.out" \
  --error="$H2G_GREEKMMLU_STAGING/logs/shard-%t.err" \
  bash "$subproject/clariden/run_greekmmlu_shard.sh"

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
trap - EXIT
printf '%s\n' "$H2G_GREEKMMLU_OUTPUT/aggregate/receipt.json"

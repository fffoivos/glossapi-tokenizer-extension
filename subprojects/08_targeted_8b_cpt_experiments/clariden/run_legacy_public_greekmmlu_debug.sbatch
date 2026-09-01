#!/usr/bin/env bash
# Run only the frozen historical BF16 public-GreekMMLU compatibility endpoint.
#SBATCH --job-name=h2g-legacy-gmmlu
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:10:00
#SBATCH --no-requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_SCALE H2G_ITERATION \
  H2G_CHECKPOINT_EXPORT H2G_LEGACY_CONTRACT H2G_LEGACY_OUTPUT EVAL_VENV; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "legacy GreekMMLU requires one debug node" >&2; exit 2;
}
[[ "$H2G_SCALE" == 8b && "$H2G_ITERATION" == 3218 ]] || {
  echo "legacy compatibility is defined only for 8b update 3218" >&2; exit 2;
}
[[ ! -e "$H2G_LEGACY_OUTPUT" ]] || { echo "immutable legacy output exists" >&2; exit 2; }

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

readarray -t resolved < <(
  uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
    "$H2G_CHECKPOINT_EXPORT" "$H2G_LEGACY_CONTRACT" <<'PY'
import hashlib, json, pathlib, sys
export_path, contract_path = map(pathlib.Path, sys.argv[1:])
export = json.load(export_path.open(encoding="utf-8"))
if not (export.get("schema_version") == "apertus_hard_h_to_g_checkpoint_export_v1"
        and export.get("status") == "completed" and export.get("scale") == "8b"
        and int(export.get("iteration", -1)) == 3218
        and export.get("ready_for_frozen_evaluators") is True):
    raise SystemExit("checkpoint export drift")
model = pathlib.Path(export["hf_export"]["path"]).resolve()
config = json.load((model / "config.json").open(encoding="utf-8"))
if int(config.get("vocab_size", -1)) != 148480: raise SystemExit("legacy vocab drift")
contract = json.load(contract_path.open(encoding="utf-8"))
if not (contract.get("schema_version") == "apertus_legacy_public_greekmmlu_receipt_v1"
        and contract.get("status") == "frozen"
        and contract.get("loader_change_scope") == "dataset_loading_only"):
    raise SystemExit("legacy contract drift")
def verify(binding):
    path = pathlib.Path(binding["path"]).resolve()
    if path.stat().st_size != int(binding["bytes"]) or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
        raise SystemExit(f"legacy binding drift: {path}")
    return path
source = pathlib.Path(contract["source_root"]).resolve()
for binding in contract["sources"].values(): verify(binding)
snapshot = verify(contract["snapshot"])
adapter = verify(contract["snapshot_adapter"])
print(model)
print(source / "run_native_greek_mcq_eval.py")
print(source / "native_greek_benchmark_registry.json")
print(snapshot)
print(adapter)
PY
)
[[ "${#resolved[@]}" == 5 ]] || { echo "failed to resolve legacy inputs" >&2; exit 2; }
model=${resolved[0]}
evaluator=${resolved[1]}
registry=${resolved[2]}
snapshot=${resolved[3]}
adapter=${resolved[4]}
label=8b_iter_0003218_legacy_public
parent=$(dirname "$H2G_LEGACY_OUTPUT")
base=$(basename "$H2G_LEGACY_OUTPUT")
staging="$parent/.${base}.${SLURM_JOB_ID}.partial"
[[ ! -e "$staging" ]] || { echo "legacy partial output exists" >&2; exit 2; }
mkdir -p "$staging/evaluation"
cleanup() {
  status=$?
  if [[ "$status" != 0 && -d "$staging" ]]; then
    mv "$staging" "$parent/_failed_${base}.${SLURM_JOB_ID}"
  fi
  exit "$status"
}
trap cleanup EXIT

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$adapter" \
  --legacy-evaluator "$evaluator" --snapshot "$snapshot" \
  --registry "$registry" --benchmarks greekmmlu --model "$label=$model" \
  --output-dir "$staging/evaluation" --sample-size 0 --random-state 42 \
  --dtype bfloat16 --max-input-tokens 3072 \
  --candidate-batch-size 16 --example-batch-size 16 --trust-remote-code
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$subproject/evaluation/finalize_legacy_public_greekmmlu.py" \
  --evaluation-root "$staging/evaluation" --model-label "$label" --model "$model" \
  --checkpoint-export "$H2G_CHECKPOINT_EXPORT" \
  --legacy-contract-receipt "$H2G_LEGACY_CONTRACT" \
  --output "$staging/result_receipt.json"
mv "$staging" "$H2G_LEGACY_OUTPUT"
trap - EXIT
printf '%s\n' "$H2G_LEGACY_OUTPUT/result_receipt.json"

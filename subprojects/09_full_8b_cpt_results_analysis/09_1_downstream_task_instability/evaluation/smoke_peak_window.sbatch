#!/usr/bin/env bash
#SBATCH --job-name=full8-peak-smoke
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=00:08:00
#SBATCH --no-requeue
#SBATCH --comment=AUTHORIZED_NATIVE_GREEK_PEAK_WINDOW_DO_NOT_CANCEL

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${PEAK_WRAPPER_ROOT:?set frozen peak-window wrapper root}"
: "${PEAK_ASSETS_ROOT:?set frozen clean-subset asset root}"
: "${PEAK_SMOKE_ROOT:?set new smoke output root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == 1 ]] || {
  echo "peak-window smoke requires one debug node" >&2
  exit 2
}
[[ ! -e "$PEAK_SMOKE_ROOT" ]] || {
  echo "refusing to overwrite smoke root" >&2
  exit 2
}

SOURCE_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$SOURCE_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
eval_dir="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation"
runner="$eval_dir/run_checkpoint_suite.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
verifier="$eval_dir/verify_eval_bundle.py"
runtime_wheel="$PEAK_WRAPPER_ROOT/vendor/accelerate-1.14.0-py3-none-any.whl"
contract="$PEAK_ASSETS_ROOT/peak_window_missing4_contract.json"
manifest="$PEAK_ASSETS_ROOT/peak_window_examples_manifest.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$verifier" --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_ROOT/bundle_receipt.json"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$verifier" --root "$PEAK_WRAPPER_ROOT" \
  --receipt "$PEAK_WRAPPER_ROOT/bundle_receipt.json"
model=$(uenv run pytorch/v2.9.1:v2 --view=default -- python3 - "$contract" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['checkpoint_scope']:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
mkdir -p "$PEAK_SMOKE_ROOT/logs"
pids=()
while IFS= read -r binding; do
  label=${binding%%=*}
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 \
    --mem=160G --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$PEAK_SMOKE_ROOT/logs/${label}.out" \
    --error="$PEAK_SMOKE_ROOT/logs/${label}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
      env PYTHONPATH="$runtime_wheel" python3 "$runner" \
      --contract "$contract" --manifest "$manifest" --native-runner "$native_runner" \
      --model "$binding" --output-dir "$PEAK_SMOKE_ROOT/$label" --dtype float32 \
      --scorer-mode legacy --benchmarks demosqa --candidate-batch-size 1 \
      --example-batch-size 1 --max-examples-per-benchmark 1 \
      --row-shard-index 0 --row-shard-count 1 &
  pids+=("$!")
done <<<"$model"
[[ ${#pids[@]} -eq 4 ]]
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "one or more smoke evaluators failed" >&2; exit 1; }
uenv run pytorch/v2.9.1:v2 --view=default -- python3 - \
  "$PEAK_SMOKE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); receipts=sorted(root.glob('iter_*/receipt.json'))
if len(receipts) != 4:
    raise SystemExit(f'expected four checkpoint smoke receipts, got {len(receipts)}')
labels=[]; bindings=[]
for path in receipts:
    d=json.load(path.open())
    if d.get('status') != 'completed' or d.get('counts') != {'demosqa':1}:
        raise SystemExit(f'peak-window smoke did not pass: {path}: {d}')
    labels.append(d['model']['label'])
    bindings.append({'model':d['model']['label'],'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'smoke_receipt.json').write_text(json.dumps({'schema_version':'apertus_full8_native_greek_peak_window_smoke_v1','status':'passed','checkpoints':bindings},indent=2,sort_keys=True)+'\n')
print({'ok':True,'models':labels,'examples_per_model':1})
PY

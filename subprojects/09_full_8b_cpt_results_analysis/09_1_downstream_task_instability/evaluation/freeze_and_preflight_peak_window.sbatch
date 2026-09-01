#!/usr/bin/env bash
#SBATCH --job-name=full8-peak-preflight
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --comment=AUTHORIZED_NATIVE_GREEK_PEAK_WINDOW_DO_NOT_CANCEL

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${PEAK_WRAPPER_ROOT:?set the materialized wrapper root}"
: "${PEAK_ASSETS_ROOT:?set the new clean-subset asset root}"
: "${PEAK_GIT_COMMIT:?set the source commit label}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == 1 ]] || {
  echo "peak-window preflight requires one debug node" >&2
  exit 2
}

SOURCE_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$SOURCE_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
eval_dir="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation"
freeze="$eval_dir/freeze_eval_bundle.py"
verify="$eval_dir/verify_eval_bundle.py"
binder="$PEAK_WRAPPER_ROOT/evaluation/bind_peak_window_native_suite.py"
bindings="$PEAK_WRAPPER_ROOT/evaluation/peak_window_checkpoint_bindings.json"
source_contract="$eval_dir/native_greek_3cp_contract.json"
source_manifest="$SOURCE_ROOT/frozen_examples_v2/manifest.json"
source_gate="$SOURCE_ROOT/execution_gate_final_v1/execution_gate.json"
code_receipt="$EVAL_CODE_ROOT/bundle_receipt.json"
audit=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/audit_receipt.json
exclusions=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/recommended_excluded_example_ids.jsonl
runtime_wheel="$PEAK_WRAPPER_ROOT/vendor/accelerate-1.14.0-py3-none-any.whl"

[[ ! -e "$PEAK_WRAPPER_ROOT/bundle_receipt.json" ]] || {
  echo "wrapper is already frozen" >&2
  exit 2
}
[[ ! -e "$PEAK_ASSETS_ROOT" ]] || {
  echo "refusing to overwrite clean-subset assets" >&2
  exit 2
}
for path in "$freeze" "$verify" "$binder" "$bindings" "$source_contract" \
  "$source_manifest" "$source_gate" "$code_receipt" "$audit" "$exclusions" \
  "$runtime_wheel"; do
  [[ -f "$path" ]] || { echo "missing preflight input $path" >&2; exit 2; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- \
  env PYTHONPATH="$runtime_wheel" python3 - <<'PY'
import accelerate, torch, transformers
if torch.__version__.split('+')[0] != '2.9.1':
    raise SystemExit(f'unexpected torch version: {torch.__version__}')
if transformers.__version__ != '4.57.0':
    raise SystemExit(f'unexpected transformers version: {transformers.__version__}')
if accelerate.__version__ != '1.14.0':
    raise SystemExit(f'unexpected accelerate version: {accelerate.__version__}')
print({'torch':torch.__version__,'transformers':transformers.__version__,'accelerate':accelerate.__version__})
PY

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$freeze" --root "$PEAK_WRAPPER_ROOT" --git-commit "$PEAK_GIT_COMMIT"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verify" --root "$PEAK_WRAPPER_ROOT" \
  --receipt "$PEAK_WRAPPER_ROOT/bundle_receipt.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$binder" --source-contract "$source_contract" \
  --source-manifest "$source_manifest" --source-execution-gate "$source_gate" \
  --eval-code-receipt "$code_receipt" --bindings "$bindings" \
  --contamination-audit-receipt "$audit" --exclusions "$exclusions" \
  --output-dir "$PEAK_ASSETS_ROOT"

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$PEAK_ASSETS_ROOT/rebind_receipt.json" <<'PY'
import json, sys
receipt=json.load(open(sys.argv[1]))
if receipt.get('status') != 'passed' or not all(receipt.get('checks', {}).values()):
    raise SystemExit('clean-subset preflight receipt did not pass')
subset=receipt['clean_subset']
if subset['exclusions']['rows'] != 10076:
    raise SystemExit('expected 10,076 audited exclusions')
if sum(subset['retained_by_benchmark'].values()) != 73894:
    raise SystemExit('expected 73,894 retained clean examples')
print(json.dumps({'ok':True,'excluded':10076,'retained':73894},sort_keys=True))
PY

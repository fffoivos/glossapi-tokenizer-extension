#!/usr/bin/env bash
#SBATCH --job-name=full8-eval-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=720G
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
for spec in legacy_fp32:float32:legacy:1 suffix_fp32:float32:suffix:16 suffix_bf16:bfloat16:suffix:16; do
  IFS=: read -r label dtype mode batch <<<"$spec"
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=220G \
    --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$EVAL_OUTPUT_ROOT/logs/$label.out" --error="$EVAL_OUTPUT_ROOT/logs/$label.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$eval_root/run_checkpoint_suite.py" \
      --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
      --model "$model" --output-dir "$EVAL_OUTPUT_ROOT/$label" --dtype "$dtype" --scorer-mode "$mode" \
      --candidate-batch-size "$batch" --example-batch-size 16 --max-examples-per-benchmark 128 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || exit 1

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$eval_root/compare_scorer_parity.py" \
  --legacy "$EVAL_OUTPUT_ROOT/legacy_fp32/predictions.jsonl" \
  --suffix "$EVAL_OUTPUT_ROOT/suffix_fp32/predictions.jsonl" \
  --output "$EVAL_OUTPUT_ROOT/scorer_parity_receipt.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$eval_root/compare_dtype_parity.py" \
  --float32 "$EVAL_OUTPUT_ROOT/suffix_fp32/predictions.jsonl" \
  --bfloat16 "$EVAL_OUTPUT_ROOT/suffix_bf16/predictions.jsonl" \
  --output "$EVAL_OUTPUT_ROOT/dtype_parity_receipt.json"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_OUTPUT_ROOT" "$EVAL_CODE_BUNDLE_RECEIPT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,bundle_path,contract_path,manifest_path=map(Path,sys.argv[1:])
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
scorer=json.load(open(root/'scorer_parity_receipt.json'))
dtype=json.load(open(root/'dtype_parity_receipt.json'))
bundle=json.load(open(bundle_path))
out={'schema_version':'apertus_full8_native_greek_acceleration_gate_v1','status':'passed' if scorer['status']=='passed' and dtype['status']=='passed' else 'rejected','scorer_parity':{'status':scorer['status'],'receipt_sha256':sha(root/'scorer_parity_receipt.json')},'dtype_parity':{'status':dtype['status'],'receipt_sha256':sha(root/'dtype_parity_receipt.json')},'code_tree_sha256':bundle['tree_sha256'],'contract_sha256':sha(contract_path),'manifest_sha256':sha(manifest_path)}
(root/'acceleration_gate.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if out['status']!='passed': raise SystemExit(1)
PY

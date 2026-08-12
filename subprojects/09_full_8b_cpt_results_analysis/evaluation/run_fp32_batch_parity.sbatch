#!/usr/bin/env bash
#SBATCH --job-name=full8-fp32-batch-gate
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
: "${EVAL_BF16_REJECTION_RECEIPT:?set rejected BF16 parity receipt}"
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
for batch in 1 4; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=220G \
    --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$EVAL_OUTPUT_ROOT/logs/batch${batch}.out" --error="$EVAL_OUTPUT_ROOT/logs/batch${batch}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$eval_root/run_checkpoint_suite.py" \
      --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
      --model "$model" --output-dir "$EVAL_OUTPUT_ROOT/batch${batch}" --dtype float32 --scorer-mode legacy \
      --candidate-batch-size "$batch" --example-batch-size 16 --max-examples-per-benchmark 128 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || exit 1

set +e
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$eval_root/compare_batch_parity.py" \
  --batch1 "$EVAL_OUTPUT_ROOT/batch1/predictions.jsonl" \
  --batch4 "$EVAL_OUTPUT_ROOT/batch4/predictions.jsonl" \
  --output "$EVAL_OUTPUT_ROOT/parity_receipt.json"
parity_rc=$?
set -e
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_OUTPUT_ROOT" "$EVAL_CODE_BUNDLE_RECEIPT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" "$EVAL_BF16_REJECTION_RECEIPT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,bundle_path,contract_path,manifest_path,bf16_path=map(Path,sys.argv[1:])
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
parity=json.load(open(root/'parity_receipt.json')); bundle=json.load(open(bundle_path)); bf16=json.load(open(bf16_path))
if bf16.get('schema_version')!='apertus_full8_native_greek_dtype_parity_v1' or bf16.get('status')!='rejected':
 raise SystemExit('BF16 rejection receipt is not valid')
out={
 'schema_version':'apertus_full8_native_greek_execution_gate_v1',
 'status':parity['status'],
 'selected':{'dtype':'float32','scorer_mode':'legacy','candidate_batch_size':4,'example_batch_size':16},
 'code_tree_sha256':bundle['tree_sha256'],
 'contract_sha256':sha(contract_path),
 'manifest_sha256':sha(manifest_path),
 'batch_parity_receipt_sha256':sha(root/'parity_receipt.json'),
 'bf16_rejection_receipt_sha256':sha(bf16_path),
}
(root/'execution_gate.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY
[[ $parity_rc -eq 0 ]] || exit "$parity_rc"

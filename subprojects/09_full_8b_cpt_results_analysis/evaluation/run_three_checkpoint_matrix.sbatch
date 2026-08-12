#!/usr/bin/env bash
#SBATCH --job-name=full8-native3cp
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=720G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${EVAL_CODE_ROOT:?set immutable evaluation code root}"
: "${EVAL_CONTRACT:?set frozen contract}"
: "${EVAL_MANIFEST:?set frozen example manifest}"
: "${EVAL_OUTPUT_ROOT:?set new output root}"
: "${EVAL_VENV:?set evaluation venv}"

runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
[[ -f "$runner" && -f "$native_runner" && -f "$EVAL_CONTRACT" && -f "$EVAL_MANIFEST" ]]
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $EVAL_OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_OUTPUT_ROOT/logs"

mapfile -t bindings < <(python3 - "$EVAL_CONTRACT" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["checkpoint_scope"]:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#bindings[@]} -eq 3 ]]

dtype=${EVAL_DTYPE:-float32}
candidate_batch=${EVAL_CANDIDATE_BATCH_SIZE:-64}
example_batch=${EVAL_EXAMPLE_BATCH_SIZE:-64}
sample_limit=${EVAL_MAX_EXAMPLES_PER_BENCHMARK:-0}

pids=()
for binding in "${bindings[@]}"; do
  label=${binding%%=*}
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --gpus-per-task=1 \
    --cpu-bind=cores --output="$EVAL_OUTPUT_ROOT/logs/${label}.out" --error="$EVAL_OUTPUT_ROOT/logs/${label}.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
      --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
      --model "$binding" --output-dir "$EVAL_OUTPUT_ROOT/$label" --dtype "$dtype" \
      --candidate-batch-size "$candidate_batch" --example-batch-size "$example_batch" \
      --max-examples-per-benchmark "$sample_limit" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "one or more checkpoint evaluators failed" >&2; exit 1; }

python3 - "$EVAL_OUTPUT_ROOT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,contract,manifest=map(Path,sys.argv[1:])
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
receipts=[]
for p in sorted(root.glob('iter_*/receipt.json')):
 d=json.load(open(p));
 if d.get('status')!='completed': raise SystemExit(f'incomplete {p}')
 receipts.append({'path':str(p.resolve()),'sha256':sha(p),'model':d['model']['label'],'wall_seconds':d['wall_seconds']})
if len(receipts)!=3: raise SystemExit(f'expected 3 receipts, got {len(receipts)}')
out={'schema_version':'apertus_full8_native_greek_3cp_matrix_v1','status':'completed','slurm_job_id':__import__('os').environ.get('SLURM_JOB_ID'),'contract_sha256':sha(contract),'manifest_sha256':sha(manifest),'checkpoint_receipts':receipts}
(root/'matrix_receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

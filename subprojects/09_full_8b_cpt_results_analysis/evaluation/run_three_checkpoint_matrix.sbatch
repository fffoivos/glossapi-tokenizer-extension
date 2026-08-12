#!/usr/bin/env bash
#SBATCH --job-name=full8-native3cp
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=660G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${EVAL_CODE_ROOT:?set immutable evaluation code root}"
: "${EVAL_CODE_BUNDLE_RECEIPT:?set immutable evaluation code receipt}"
: "${EVAL_CONTRACT:?set frozen contract}"
: "${EVAL_MANIFEST:?set frozen example manifest}"
: "${EVAL_OUTPUT_ROOT:?set new output root}"
: "${EVAL_VENV:?set evaluation venv}"
: "${EVAL_DTYPE_GATE:?set passing BF16 parity receipt}"

runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
[[ -f "$runner" && -f "$native_runner" && -f "$EVAL_CONTRACT" && -f "$EVAL_MANIFEST" ]]
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_BUNDLE_RECEIPT"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_DTYPE_GATE" "$EVAL_CODE_BUNDLE_RECEIPT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
gate,bundle_path,contract_path,manifest_path=map(Path,sys.argv[1:])
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
d=json.load(open(gate)); bundle=json.load(open(bundle_path))
if d.get('schema_version')!='apertus_full8_native_greek_dtype_gate_v1' or d.get('status')!='passed':
    raise SystemExit('BF16 dtype gate is not passed')
expected={'code_tree_sha256':bundle['tree_sha256'],'contract_sha256':sha(contract_path),'manifest_sha256':sha(manifest_path)}
if any(d.get(key)!=value for key,value in expected.items()):
    raise SystemExit('BF16 dtype gate is not bound to this code/contract/manifest')
PY
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $EVAL_OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_OUTPUT_ROOT/logs"

mapfile -t bindings < <(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$EVAL_CONTRACT" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["checkpoint_scope"]:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#bindings[@]} -eq 3 ]]

dtype=${EVAL_DTYPE:-bfloat16}
candidate_batch=${EVAL_CANDIDATE_BATCH_SIZE:-16}
example_batch=${EVAL_EXAMPLE_BATCH_SIZE:-16}
sample_limit=${EVAL_MAX_EXAMPLES_PER_BENCHMARK:-0}

pids=()
for binding in "${bindings[@]}"; do
  label=${binding%%=*}
  for shard in core wic; do
    if [[ "$shard" == core ]]; then
      benchmarks=demosqa,medical_mcqa,asep_mcqa,gpcr,oyxoy_nli,oyxoy_wsd_definition,oyxoy_metaphor
    else
      benchmarks=oyxoy_wic
    fi
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=220G \
      --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
      --output="$EVAL_OUTPUT_ROOT/logs/${label}-${shard}.out" --error="$EVAL_OUTPUT_ROOT/logs/${label}-${shard}.err" \
      uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
        --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
        --model "$binding" --output-dir "$EVAL_OUTPUT_ROOT/$label/$shard" --dtype "$dtype" \
        --scorer-mode legacy --benchmarks "$benchmarks" \
        --candidate-batch-size "$candidate_batch" --example-batch-size "$example_batch" \
        --max-examples-per-benchmark "$sample_limit" &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "one or more checkpoint evaluators failed" >&2; exit 1; }

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_OUTPUT_ROOT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,contract,manifest=map(Path,sys.argv[1:])
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
receipts=[]
expected={row['label'] for row in json.load(open(contract))['checkpoint_scope']}
for p in sorted(root.glob('iter_*/*/receipt.json')):
 d=json.load(open(p));
 if d.get('status')!='completed': raise SystemExit(f'incomplete {p}')
 receipts.append({'path':str(p.resolve()),'sha256':sha(p),'model':d['model']['label'],'shard':p.parent.name,'counts':d['counts'],'wall_seconds':d['wall_seconds']})
if len(receipts)!=6: raise SystemExit(f'expected 6 receipts, got {len(receipts)}')
by_model={label:{} for label in expected}
for receipt in receipts:
 by_model[receipt['model']][receipt['shard']]=receipt
for label,shards in by_model.items():
 if set(shards)!={'core','wic'}: raise SystemExit(f'incomplete shards for {label}: {sorted(shards)}')
 counts={}
 for shard in shards.values():
  overlap=set(counts)&set(shard['counts'])
  if overlap: raise SystemExit(f'overlapping benchmark shards for {label}: {sorted(overlap)}')
  counts.update(shard['counts'])
 if counts != json.load(open(manifest))['counts']: raise SystemExit(f'count mismatch for {label}')
out={'schema_version':'apertus_full8_native_greek_3cp_matrix_v1','status':'completed','slurm_job_id':__import__('os').environ.get('SLURM_JOB_ID'),'contract_sha256':sha(contract),'manifest_sha256':sha(manifest),'checkpoint_receipts':receipts}
(root/'matrix_receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

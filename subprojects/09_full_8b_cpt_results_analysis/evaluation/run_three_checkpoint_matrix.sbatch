#!/usr/bin/env bash
#SBATCH --job-name=full8-native3cp
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=00:22:30
#SBATCH --no-requeue

set -euo pipefail
: "${EVAL_CODE_ROOT:?set immutable evaluation code root}"
: "${EVAL_CODE_BUNDLE_RECEIPT:?set immutable evaluation code receipt}"
: "${EVAL_CONTRACT:?set frozen contract}"
: "${EVAL_MANIFEST:?set frozen example manifest}"
: "${EVAL_OUTPUT_ROOT:?set new output root}"
: "${EVAL_VENV:?set evaluation venv}"
: "${EVAL_EXECUTION_GATE:?set passing execution-profile receipt}"
: "${EVAL_SEGMENT_INDEX:?set matrix segment index 0, 1, or 2}"
[[ "$EVAL_SEGMENT_INDEX" =~ ^[012]$ ]] || { echo "invalid EVAL_SEGMENT_INDEX=$EVAL_SEGMENT_INDEX" >&2; exit 2; }

runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
[[ -f "$runner" && -f "$aggregator" && -f "$native_runner" && -f "$EVAL_CONTRACT" && -f "$EVAL_MANIFEST" ]]
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py" \
  --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_BUNDLE_RECEIPT"
mapfile -t execution_profile < <(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_EXECUTION_GATE" "$EVAL_CODE_BUNDLE_RECEIPT" "$EVAL_CONTRACT" "$EVAL_MANIFEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
gate,bundle_path,contract_path,manifest_path=map(Path,sys.argv[1:])
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
d=json.load(open(gate)); bundle=json.load(open(bundle_path))
if d.get('schema_version')!='apertus_full8_native_greek_execution_gate_v1' or d.get('status')!='passed':
    raise SystemExit('execution-profile gate is not passed')
expected={'code_tree_sha256':bundle['tree_sha256'],'contract_sha256':sha(contract_path),'manifest_sha256':sha(manifest_path)}
if any(d.get(key)!=value for key,value in expected.items()):
    raise SystemExit('execution-profile gate is not bound to this code/contract/manifest')
selected=d.get('selected') or {}
if selected.get('dtype')!='float32' or selected.get('scorer_mode')!='legacy':
    raise SystemExit(f'unsupported selected profile: {selected}')
if int(selected.get('candidate_batch_size', 0)) not in {1, 4} or int(selected.get('example_batch_size', 0)) != 16:
    raise SystemExit(f'unsupported selected batch profile: {selected}')
print(selected['dtype'])
print(int(selected['candidate_batch_size']))
print(int(selected['example_batch_size']))
PY
)
[[ ${#execution_profile[@]} -eq 3 ]]
segment_receipt="$EVAL_OUTPUT_ROOT/segment_${EVAL_SEGMENT_INDEX}_receipt.json"
if [[ "$EVAL_SEGMENT_INDEX" -eq 0 ]]; then
  [[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $EVAL_OUTPUT_ROOT" >&2; exit 2; }
  mkdir -p "$EVAL_OUTPUT_ROOT/logs"
else
  [[ -d "$EVAL_OUTPUT_ROOT" && ! -e "$segment_receipt" && ! -e "$EVAL_OUTPUT_ROOT/matrix_receipt.json" ]]
  previous=$((EVAL_SEGMENT_INDEX - 1))
  uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
    "$EVAL_OUTPUT_ROOT/segment_${previous}_receipt.json" "$previous" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('status')!='completed' or d.get('segment_index')!=int(sys.argv[2]):
 raise SystemExit('previous matrix segment is not complete')
PY
fi

mapfile -t bindings < <(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$EVAL_CONTRACT" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["checkpoint_scope"]:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#bindings[@]} -eq 3 ]]

dtype=${execution_profile[0]}
candidate_batch=${execution_profile[1]}
example_batch=${execution_profile[2]}
sample_limit=${EVAL_MAX_EXAMPLES_PER_BENCHMARK:-0}

pids=()
global_task_index=0
for binding in "${bindings[@]}"; do
  label=${binding%%=*}
  for shard in demos other_mcq oyxoy_other oyxoy_wsd0 oyxoy_wsd1 oyxoy_wsd2 oyxoy_wsd3 oyxoy_wsd4 oyxoy_wic0 oyxoy_wic1 oyxoy_wic2 oyxoy_wic3 oyxoy_wic4 oyxoy_wic5 oyxoy_wic6; do
    assigned_segment=$((global_task_index % 3))
    global_task_index=$((global_task_index + 1))
    [[ "$assigned_segment" -eq "$EVAL_SEGMENT_INDEX" ]] || continue
    row_shard_index=0
    row_shard_count=1
    case "$shard" in
      demos) benchmarks=demosqa ;;
      other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
      oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
      oyxoy_wsd0) benchmarks=oyxoy_wsd_definition; row_shard_count=5 ;;
      oyxoy_wsd1) benchmarks=oyxoy_wsd_definition; row_shard_index=1; row_shard_count=5 ;;
      oyxoy_wsd2) benchmarks=oyxoy_wsd_definition; row_shard_index=2; row_shard_count=5 ;;
      oyxoy_wsd3) benchmarks=oyxoy_wsd_definition; row_shard_index=3; row_shard_count=5 ;;
      oyxoy_wsd4) benchmarks=oyxoy_wsd_definition; row_shard_index=4; row_shard_count=5 ;;
      oyxoy_wic0) benchmarks=oyxoy_wic; row_shard_count=7 ;;
      oyxoy_wic1) benchmarks=oyxoy_wic; row_shard_index=1; row_shard_count=7 ;;
      oyxoy_wic2) benchmarks=oyxoy_wic; row_shard_index=2; row_shard_count=7 ;;
      oyxoy_wic3) benchmarks=oyxoy_wic; row_shard_index=3; row_shard_count=7 ;;
      oyxoy_wic4) benchmarks=oyxoy_wic; row_shard_index=4; row_shard_count=7 ;;
      oyxoy_wic5) benchmarks=oyxoy_wic; row_shard_index=5; row_shard_count=7 ;;
      oyxoy_wic6) benchmarks=oyxoy_wic; row_shard_index=6; row_shard_count=7 ;;
    esac
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 --mem=160G \
      --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
      --output="$EVAL_OUTPUT_ROOT/logs/${label}-${shard}.out" --error="$EVAL_OUTPUT_ROOT/logs/${label}-${shard}.err" \
      uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$runner" \
        --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST" --native-runner "$native_runner" \
        --model "$binding" --output-dir "$EVAL_OUTPUT_ROOT/$label/$shard" --dtype "$dtype" \
        --scorer-mode legacy --benchmarks "$benchmarks" \
        --candidate-batch-size "$candidate_batch" --example-batch-size "$example_batch" \
        --max-examples-per-benchmark "$sample_limit" \
        --row-shard-index "$row_shard_index" --row-shard-count "$row_shard_count" &
    pids+=("$!")
  done
done
[[ ${#pids[@]} -eq 15 ]]

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "one or more checkpoint evaluators failed" >&2; exit 1; }

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$EVAL_OUTPUT_ROOT" "$EVAL_SEGMENT_INDEX" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); segment=int(sys.argv[2])
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): d.update(chunk)
 return d.hexdigest()
receipts=[]
for path in sorted(root.glob('iter_*/*/receipt.json')):
 d=json.load(open(path))
 if d.get('status')!='completed': raise SystemExit(f'incomplete shard {path}')
 receipts.append({'path':str(path.resolve()),'sha256':sha(path)})
expected=15*(segment+1)
if len(receipts)!=expected: raise SystemExit(f'expected {expected} cumulative shard receipts, got {len(receipts)}')
out={'schema_version':'apertus_full8_native_greek_matrix_segment_v1','status':'completed','segment_index':segment,'slurm_job_id':__import__('os').environ.get('SLURM_JOB_ID'),'cumulative_shard_receipts':receipts}
(root/f'segment_{segment}_receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

if [[ "$EVAL_SEGMENT_INDEX" -eq 2 ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$aggregator" \
    --root "$EVAL_OUTPUT_ROOT" --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST"
fi

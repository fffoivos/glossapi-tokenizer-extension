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
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${EVAL_CODE_ROOT:?set immutable evaluation code root}"
: "${EVAL_CODE_BUNDLE_RECEIPT:?set immutable evaluation code receipt}"
: "${EVAL_CONTRACT:?set frozen contract}"
: "${EVAL_MANIFEST:?set frozen example manifest}"
: "${EVAL_OUTPUT_ROOT:?set new output root}"
: "${EVAL_VENV:?set evaluation venv}"
: "${EVAL_EXECUTION_GATE:?set passing execution-profile receipt}"

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
[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || { echo "refusing to overwrite $EVAL_OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_OUTPUT_ROOT/logs"

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
for binding in "${bindings[@]}"; do
  label=${binding%%=*}
  for shard in mcq oyxoy_other oyxoy_wsd oyxoy_wic0 oyxoy_wic1; do
    row_shard_index=0
    row_shard_count=1
    case "$shard" in
      mcq) benchmarks=demosqa,medical_mcqa,asep_mcqa,gpcr ;;
      oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
      oyxoy_wsd) benchmarks=oyxoy_wsd_definition ;;
      oyxoy_wic0) benchmarks=oyxoy_wic; row_shard_count=2 ;;
      oyxoy_wic1) benchmarks=oyxoy_wic; row_shard_index=1; row_shard_count=2 ;;
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

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ $status -eq 0 ]] || { echo "one or more checkpoint evaluators failed" >&2; exit 1; }

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" "$aggregator" \
  --root "$EVAL_OUTPUT_ROOT" --contract "$EVAL_CONTRACT" --manifest "$EVAL_MANIFEST"

#!/usr/bin/env bash
#SBATCH --job-name=full8-peak-native
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=00:22:00
#SBATCH --no-requeue
#SBATCH --comment=AUTHORIZED_NATIVE_GREEK_PEAK_WINDOW_DO_NOT_CANCEL

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${PEAK_WRAPPER_ROOT:?set frozen peak-window wrapper root}"
: "${PEAK_ASSETS_ROOT:?set frozen clean-subset asset root}"
: "${PEAK_OUTPUT_ROOT:?set new peak-window output root}"
: "${PEAK_SEGMENT_INDEX:?set segment index 0..5}"
: "${PEAK_EXPECTED_NNODES:=4}"
[[ "$PEAK_SEGMENT_INDEX" =~ ^[0-5]$ ]] || {
  echo "invalid PEAK_SEGMENT_INDEX=$PEAK_SEGMENT_INDEX" >&2
  exit 2
}
[[ "$PEAK_EXPECTED_NNODES" == 2 || "$PEAK_EXPECTED_NNODES" == 4 ]] || {
  echo "PEAK_EXPECTED_NNODES must be 2 or 4" >&2
  exit 2
}
resume=${PEAK_RESUME:-0}
[[ "$resume" == 0 || "$resume" == 1 ]] || {
  echo "PEAK_RESUME must be 0 or 1" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$SLURM_NNODES" == "$PEAK_EXPECTED_NNODES" ]] || {
  echo "peak-window evaluation requires $PEAK_EXPECTED_NNODES debug nodes" >&2
  exit 2
}

SOURCE_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$SOURCE_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_CODE_RECEIPT="$EVAL_CODE_ROOT/bundle_receipt.json"
SOURCE_CONTRACT="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json"
SOURCE_MANIFEST="$SOURCE_ROOT/frozen_examples_v2/manifest.json"
SOURCE_GATE="$SOURCE_ROOT/execution_gate_final_v1/execution_gate.json"
SOURCE_MATRIX="$SOURCE_ROOT/matrix_v6"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
verifier="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py"
wrapper_receipt="$PEAK_WRAPPER_ROOT/bundle_receipt.json"
finalizer="$PEAK_WRAPPER_ROOT/evaluation/finalize_peak_window.py"
runtime_wheel="$PEAK_WRAPPER_ROOT/vendor/accelerate-1.14.0-py3-none-any.whl"
exclusions=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/recommended_excluded_example_ids.jsonl
contamination_audit=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/audit_receipt.json
reference_filtered=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/filtered_scores_v1
expected_exclusions_sha=7a8559461b15a308f599faf0ff25cd16c07be0a597078864f779af7f2f1fdd32
expected_audit_sha=78273fcc9e45e2e8d54c72d4b765fe9d43af2b29a5602c59b1aab8cee4849feb
expected_reference_filtered_sha=8b6dc8ebf8281425e1df416c1b3767a5329c9b70b8e77f0ab4a771328d70721e

for path in "$EVAL_CODE_RECEIPT" "$SOURCE_CONTRACT" "$SOURCE_MANIFEST" \
  "$SOURCE_GATE" "$SOURCE_MATRIX/matrix_receipt.json" "$runner" "$aggregator" \
  "$native_runner" "$verifier" "$wrapper_receipt" \
  "$finalizer" "$runtime_wheel" "$exclusions" "$contamination_audit" \
  "$reference_filtered/receipt.json"; do
  [[ -f "$path" ]] || { echo "missing input $path" >&2; exit 2; }
done
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verifier" --root "$EVAL_CODE_ROOT" --receipt "$EVAL_CODE_RECEIPT"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verifier" --root "$PEAK_WRAPPER_ROOT" --receipt "$wrapper_receipt"
actual_exclusions_sha=$(sha256sum "$exclusions" | awk '{print $1}')
[[ "$actual_exclusions_sha" == "$expected_exclusions_sha" ]] || {
  echo "contamination-exclusion hash drift" >&2
  exit 2
}
actual_audit_sha=$(sha256sum "$contamination_audit" | awk '{print $1}')
[[ "$actual_audit_sha" == "$expected_audit_sha" ]] || {
  echo "contamination-audit receipt hash drift" >&2
  exit 2
}
actual_reference_filtered_sha=$(sha256sum "$reference_filtered/receipt.json" | awk '{print $1}')
[[ "$actual_reference_filtered_sha" == "$expected_reference_filtered_sha" ]] || {
  echo "reference filtered-score receipt hash drift" >&2
  exit 2
}

assets="$PEAK_ASSETS_ROOT"
if [[ "$PEAK_SEGMENT_INDEX" -eq 0 && "$resume" -eq 0 ]]; then
  [[ ! -e "$PEAK_OUTPUT_ROOT" ]] || {
    echo "refusing to overwrite $PEAK_OUTPUT_ROOT" >&2
    exit 2
  }
  mkdir -p "$PEAK_OUTPUT_ROOT/logs"
else
  [[ -d "$PEAK_OUTPUT_ROOT" ]] || {
    echo "peak-window output root is absent" >&2
    exit 2
  }
fi
[[ -f "$assets/rebind_receipt.json" ]] || {
  echo "peak-window clean-subset asset receipt is absent" >&2
  exit 2
}

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$assets/rebind_receipt.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
receipt=json.load(open(sys.argv[1]))
if receipt.get('status')!='passed' or not all(receipt.get('checks',{}).values()):
    raise SystemExit('peak-window rebind receipt is not passed')
for key in ('contract','manifest'):
    row=receipt[key]; path=Path(row['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit(f'peak-window {key} hash drift')
subset=receipt.get('clean_subset',{})
if subset.get('exclusions',{}).get('rows') != 10076:
    raise SystemExit('clean-subset exclusion count drift')
if sum(subset.get('retained_by_benchmark',{}).values()) != 73894:
    raise SystemExit('clean-subset retained count drift')
PY

contract="$assets/peak_window_missing4_contract.json"
manifest="$assets/peak_window_examples_manifest.json"
segment_receipt="$PEAK_OUTPUT_ROOT/segment_${PEAK_SEGMENT_INDEX}_receipt.json"
[[ ! -e "$segment_receipt" ]] || {
  echo "segment receipt already exists: $segment_receipt" >&2
  exit 2
}
if [[ "$PEAK_SEGMENT_INDEX" -gt 0 ]]; then
  previous=$((PEAK_SEGMENT_INDEX - 1))
  [[ -f "$PEAK_OUTPUT_ROOT/segment_${previous}_receipt.json" ]] || {
    echo "previous segment $previous is not complete" >&2
    exit 2
  }
fi

mapfile -t models < <(uenv run pytorch/v2.9.1:v2 --view=default \
  -- "$EVAL_VENV/bin/python" - "$contract" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))['checkpoint_scope']:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#models[@]} -eq 4 ]]
shards=(demos other_mcq oyxoy_other oyxoy_wsd{0..9} oyxoy_wic{0..7})

run_task() {
  local model=$1 shard=$2 label benchmarks row_index=0 row_count=1
  label=${model%%=*}
  case "$shard" in
    demos) benchmarks=demosqa ;;
    other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
    oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
    oyxoy_wsd*)
      benchmarks=oyxoy_wsd_definition
      row_index=${shard#oyxoy_wsd}
      row_count=10
      ;;
    oyxoy_wic*)
      benchmarks=oyxoy_wic
      row_index=${shard#oyxoy_wic}
      row_count=8
      ;;
    *) echo "unknown shard $shard" >&2; return 2 ;;
  esac
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 \
    --mem=160G --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$PEAK_OUTPUT_ROOT/logs/${label}-${shard}-seg${PEAK_SEGMENT_INDEX}-%j.out" \
    --error="$PEAK_OUTPUT_ROOT/logs/${label}-${shard}-seg${PEAK_SEGMENT_INDEX}-%j.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
      env PYTHONPATH="$runtime_wheel" python3 \
      "$runner" --contract "$contract" --manifest "$manifest" \
      --native-runner "$native_runner" --model "$model" \
      --output-dir "$PEAK_OUTPUT_ROOT/$label/$shard" --dtype float32 \
      --scorer-mode legacy --benchmarks "$benchmarks" \
      --candidate-batch-size 1 --example-batch-size 16 \
      --max-examples-per-benchmark 0 --row-shard-index "$row_index" \
      --row-shard-count "$row_count"
}

pids=()
global_index=0
quarantine="$PEAK_OUTPUT_ROOT/incomplete_from_${SLURM_JOB_ID:-manual}"
for model in "${models[@]}"; do
  for shard in "${shards[@]}"; do
    assigned=$((global_index % 6))
    global_index=$((global_index + 1))
    [[ "$assigned" -eq "$PEAK_SEGMENT_INDEX" ]] || continue
    label=${model%%=*}
    task_root="$PEAK_OUTPUT_ROOT/$label/$shard"
    if [[ -f "$task_root/receipt.json" ]]; then
      continue
    fi
    if [[ -e "$task_root" ]]; then
      mkdir -p "$quarantine/$label"
      mv "$task_root" "$quarantine/$label/$shard"
    fi
    run_task "$model" "$shard" &
    pids+=("$!")
  done
done
[[ ${#pids[@]} -le 14 ]]
echo "peak-window segment=$PEAK_SEGMENT_INDEX launched=${#pids[@]}"
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || {
  echo "one or more peak-window shards failed" >&2
  exit 1
}

expected_receipts=$(((PEAK_SEGMENT_INDEX + 1) * 14))
receipt_count=$(find "$PEAK_OUTPUT_ROOT" -mindepth 3 -maxdepth 3 \
  -name receipt.json -path '*/iter_*/*' ! -path '*/combined/*' | wc -l | tr -d ' ')
[[ "$receipt_count" -eq "$expected_receipts" ]] || {
  echo "expected $expected_receipts cumulative shard receipts, found $receipt_count" >&2
  exit 1
}

if [[ "$PEAK_SEGMENT_INDEX" -eq 5 ]]; then
  [[ ! -e "$PEAK_OUTPUT_ROOT/matrix_receipt.json" ]] && \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
      "$aggregator" --root "$PEAK_OUTPUT_ROOT" --contract "$contract" \
      --manifest "$manifest"
  if [[ ! -e "$PEAK_OUTPUT_ROOT/peak_window_results.json" ]]; then
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
      "$finalizer" --new-matrix-root "$PEAK_OUTPUT_ROOT" \
      --reference-matrix-root "$SOURCE_MATRIX" \
      --reference-filtered-root "$reference_filtered" \
      --subset-receipt "$assets/rebind_receipt.json" \
      --output "$PEAK_OUTPUT_ROOT/peak_window_results.json"
  fi
fi

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$PEAK_OUTPUT_ROOT" "$PEAK_SEGMENT_INDEX" "$wrapper_receipt" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); segment=int(sys.argv[2]); wrapper=Path(sys.argv[3])
def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): d.update(block)
    return d.hexdigest()
receipts=[]
for path in sorted(root.glob('iter_*/*/receipt.json')):
    if path.parent.name == 'combined':
        continue
    data=json.load(open(path))
    if data.get('status')!='completed': raise SystemExit(f'incomplete shard {path}')
    receipts.append({'path':str(path.resolve()),'sha256':sha(path)})
expected=(segment+1)*14
if len(receipts)!=expected: raise SystemExit(f'expected {expected} receipts, got {len(receipts)}')
payload={
    'schema_version':'apertus_full8_native_greek_peak_window_segment_v1',
    'status':'completed','segment_index':segment,
    'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
    'wrapper_receipt':{'path':str(wrapper.resolve()),'sha256':sha(wrapper)},
    'cumulative_shard_receipts':receipts,
}
if segment==5:
    result=root/'peak_window_results.json'
    if not result.is_file(): raise SystemExit('final peak-window result missing')
    payload['final_result']={'path':str(result.resolve()),'sha256':sha(result)}
(root/f'segment_{segment}_receipt.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

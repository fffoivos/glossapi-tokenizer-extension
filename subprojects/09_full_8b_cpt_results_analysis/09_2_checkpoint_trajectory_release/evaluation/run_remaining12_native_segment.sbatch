#!/usr/bin/env bash
#SBATCH --job-name=full8-r12-native
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=54
#SBATCH --mem=640G
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/%x-%j.err
#SBATCH --comment=AUTHORIZED_FULL8_REMAINING12_EVALUATION

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${REMAINING12_WRAPPER_ROOT:?set frozen remaining12 wrapper root}"
: "${REMAINING12_ASSETS_ROOT:?set frozen remaining12 asset root}"
: "${REMAINING12_OUTPUT_ROOT:?set new remaining12 output root}"
: "${REMAINING12_SEGMENT_INDEX:?set segment index 0..17}"
: "${REMAINING12_EXPECTED_NNODES:=1}"
[[ "$REMAINING12_SEGMENT_INDEX" =~ ^([0-9]|1[0-7])$ ]] || { echo "invalid segment index" >&2; exit 2; }
[[ "$REMAINING12_EXPECTED_NNODES" == 1 || "$REMAINING12_EXPECTED_NNODES" == 2 ]] || {
  echo "remaining12 expected node count must be one or two" >&2; exit 2;
}
resume=${REMAINING12_RESUME:-0}
[[ "$resume" == 0 || "$resume" == 1 ]] || { echo "invalid resume flag" >&2; exit 2; }
allocation_nodes=${SLURM_NNODES:-${SLURM_STEP_NUM_NODES:-}}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "$allocation_nodes" == "$REMAINING12_EXPECTED_NNODES" ]] || {
  echo "remaining12 evaluation requires the requested debug-node geometry" >&2; exit 2;
}

SOURCE_ROOT=/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812
EVAL_CODE_ROOT="$SOURCE_ROOT/code/036e1e2e53e13d3cd58cc8cb22b3c38a52881580"
EVAL_VENV=/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026
runner="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py"
aggregator="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py"
native_runner="$EVAL_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py"
verifier="$EVAL_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py"
source_bundle_receipt="$EVAL_CODE_ROOT/bundle_receipt.json"
wrapper_receipt="$REMAINING12_WRAPPER_ROOT/bundle_receipt.json"
runtime_wheel="$REMAINING12_WRAPPER_ROOT/vendor/accelerate-1.14.0-py3-none-any.whl"
contract="$REMAINING12_ASSETS_ROOT/remaining12_contract.json"
manifest="$REMAINING12_ASSETS_ROOT/remaining12_manifest.json"
asset_receipt="$REMAINING12_ASSETS_ROOT/rebind_receipt.json"

for path in "$runner" "$aggregator" "$native_runner" "$verifier" "$source_bundle_receipt" \
  "$wrapper_receipt" "$runtime_wheel" "$contract" "$manifest" "$asset_receipt"; do
  [[ -f "$path" ]] || { echo "missing runner input $path" >&2; exit 2; }
done
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verifier" --root "$EVAL_CODE_ROOT" --receipt "$source_bundle_receipt"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
  "$verifier" --root "$REMAINING12_WRAPPER_ROOT" --receipt "$wrapper_receipt"
uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$asset_receipt" "$contract" "$manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt=json.load(open(sys.argv[1]))
if receipt.get("status") != "passed" or not all(receipt.get("checks", {}).values()):
    raise SystemExit("asset receipt is not passing")
if len(receipt.get("checkpoints", [])) != 12:
    raise SystemExit("asset checkpoint scope drift")
for index, path in enumerate(map(Path, sys.argv[2:]), start=0):
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=receipt[("contract", "manifest")[index]]["sha256"]
    if actual != expected:
        raise SystemExit(f"asset hash drift: {path}")
print(json.dumps({"ok":True,"checkpoints":12},sort_keys=True))
PY

if [[ "$REMAINING12_SEGMENT_INDEX" -eq 0 && "$resume" -eq 0 ]]; then
  [[ ! -e "$REMAINING12_OUTPUT_ROOT" ]] || { echo "refusing to overwrite output root" >&2; exit 2; }
  mkdir -p "$REMAINING12_OUTPUT_ROOT/logs"
else
  [[ -d "$REMAINING12_OUTPUT_ROOT" ]] || { echo "output root is absent" >&2; exit 2; }
fi
segment_receipt="$REMAINING12_OUTPUT_ROOT/segment_${REMAINING12_SEGMENT_INDEX}_receipt.json"
[[ ! -e "$segment_receipt" ]] || { echo "segment receipt exists" >&2; exit 2; }
if [[ "$REMAINING12_SEGMENT_INDEX" -gt 0 ]]; then
  previous=$((REMAINING12_SEGMENT_INDEX - 1))
  [[ -f "$REMAINING12_OUTPUT_ROOT/segment_${previous}_receipt.json" ]] || {
    echo "previous segment is incomplete" >&2; exit 2;
  }
fi

mapfile -t models < <(uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - "$contract" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["checkpoint_scope"]:
    print(f'{row["label"]}={row["model_path"]}')
PY
)
[[ ${#models[@]} -eq 12 ]] || { echo "expected twelve models" >&2; exit 2; }
shards=(demos other_mcq oyxoy_other oyxoy_wsd{0..9} oyxoy_wic{0..7})

run_task() {
  local model=$1 shard=$2 label benchmarks row_index=0 row_count=1
  label=${model%%=*}
  case "$shard" in
    demos) benchmarks=demosqa ;;
    other_mcq) benchmarks=medical_mcqa,asep_mcqa,gpcr ;;
    oyxoy_other) benchmarks=oyxoy_nli,oyxoy_metaphor ;;
    oyxoy_wsd*) benchmarks=oyxoy_wsd_definition; row_index=${shard#oyxoy_wsd}; row_count=10 ;;
    oyxoy_wic*) benchmarks=oyxoy_wic; row_index=${shard#oyxoy_wic}; row_count=8 ;;
    *) echo "unknown shard $shard" >&2; return 2 ;;
  esac
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=54 \
    --mem=160G --gpus-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    --output="$REMAINING12_OUTPUT_ROOT/logs/${label}-${shard}-seg${REMAINING12_SEGMENT_INDEX}-%j.out" \
    --error="$REMAINING12_OUTPUT_ROOT/logs/${label}-${shard}-seg${REMAINING12_SEGMENT_INDEX}-%j.err" \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
      env PYTHONPATH="$runtime_wheel" python3 "$runner" \
      --contract "$contract" --manifest "$manifest" --native-runner "$native_runner" \
      --model "$model" --output-dir "$REMAINING12_OUTPUT_ROOT/$label/$shard" \
      --dtype float32 --scorer-mode legacy --benchmarks "$benchmarks" \
      --candidate-batch-size 1 --example-batch-size 16 --max-examples-per-benchmark 0 \
      --row-shard-index "$row_index" --row-shard-count "$row_count"
}

pids=()
global_index=0
quarantine="$REMAINING12_OUTPUT_ROOT/incomplete_from_${SLURM_JOB_ID:-manual}"
for model in "${models[@]}"; do
  for shard in "${shards[@]}"; do
    assigned=$((global_index % 18))
    global_index=$((global_index + 1))
    [[ "$assigned" -eq "$REMAINING12_SEGMENT_INDEX" ]] || continue
    label=${model%%=*}
    task_root="$REMAINING12_OUTPUT_ROOT/$label/$shard"
    if [[ -f "$task_root/receipt.json" ]]; then continue; fi
    if [[ -e "$task_root" ]]; then
      mkdir -p "$quarantine/$label"
      mv "$task_root" "$quarantine/$label/$shard"
    fi
    run_task "$model" "$shard" &
    pids+=("$!")
  done
done
[[ ${#pids[@]} -eq 14 ]] || { echo "expected fourteen assigned shards" >&2; exit 2; }
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
[[ "$failure" -eq 0 ]] || { echo "one or more shards failed" >&2; exit 1; }

expected_receipts=$(((REMAINING12_SEGMENT_INDEX + 1) * 14))
receipt_count=$(find "$REMAINING12_OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -name receipt.json \
  -path '*/iter_*/*' ! -path '*/combined/*' | wc -l | tr -d ' ')
[[ "$receipt_count" -eq "$expected_receipts" ]] || {
  echo "expected $expected_receipts cumulative shard receipts, found $receipt_count" >&2; exit 1;
}
if [[ "$REMAINING12_SEGMENT_INDEX" -eq 17 ]]; then
  [[ ! -e "$REMAINING12_OUTPUT_ROOT/matrix_receipt.json" ]] && \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" \
      "$aggregator" --root "$REMAINING12_OUTPUT_ROOT" --contract "$contract" --manifest "$manifest"
fi

uenv run pytorch/v2.9.1:v2 --view=default -- "$EVAL_VENV/bin/python" - \
  "$REMAINING12_OUTPUT_ROOT" "$REMAINING12_SEGMENT_INDEX" "$wrapper_receipt" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); segment=int(sys.argv[2]); wrapper=Path(sys.argv[3])
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
receipts=[]
for path in sorted(root.glob("iter_*/*/receipt.json")):
    if path.parent.name == "combined": continue
    data=json.load(open(path))
    if data.get("status") != "completed": raise SystemExit(f"incomplete shard {path}")
    receipts.append({"path":str(path.resolve()),"sha256":sha(path)})
expected=(segment+1)*14
if len(receipts) != expected: raise SystemExit(f"expected {expected} receipts, got {len(receipts)}")
payload={"schema_version":"apertus_full8_native_greek_remaining12_segment_v1","status":"completed","segment_index":segment,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"wrapper_receipt":{"path":str(wrapper.resolve()),"sha256":sha(wrapper)},"cumulative_shard_receipts":receipts}
if segment == 17:
    matrix=root/"matrix_receipt.json"
    if not matrix.is_file(): raise SystemExit("final matrix receipt missing")
    payload["matrix_receipt"]={"path":str(matrix.resolve()),"sha256":sha(matrix)}
(root/f"segment_{segment}_receipt.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

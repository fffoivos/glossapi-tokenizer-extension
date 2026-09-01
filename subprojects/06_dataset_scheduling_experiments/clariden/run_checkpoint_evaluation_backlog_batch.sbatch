#!/usr/bin/env bash
#SBATCH --job-name=mini_greekmmlu_backlog4
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=01:25:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${EVALUATION_BUNDLE:?set immutable evaluator bundle}"
: "${BACKLOG_BATCH_MANIFEST:?set frozen backlog batch manifest}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned evaluation Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set frozen compatibility shim}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint plan}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen clean subset}"

worker="$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu_one.sh"
[[ -f "$worker" ]] || { echo "missing frozen checkpoint worker" >&2; exit 2; }
batch_root="${BACKLOG_BATCH_MANIFEST%.json}.execution"
[[ ! -e "$batch_root" ]] || { echo "refusing to overwrite $batch_root" >&2; exit 2; }
mkdir -p "$batch_root/logs"

mapfile -t rows < <("$HOST_PYTHON" - "$BACKLOG_BATCH_MANIFEST" <<'PY'
import hashlib,json,sys
p=sys.argv[1]; d=json.load(open(p))
if d.get("schema_version") != "apertus_mini_greekmmlu_backlog_batch_v1" or d.get("status") != "frozen":
    raise SystemExit("invalid backlog batch manifest")
if d.get("authoritative_evaluation_dtype") != "float32" or d.get("evaluation_namespace") != "fp32_v1":
    raise SystemExit("backlog namespace or dtype drift")
rows=d.get("waves",[])
if not 1 <= len(rows) <= 32 or len(rows) != d.get("wave_count"):
    raise SystemExit("backlog wave-count drift")
for row in rows:
    wave=row["wave_manifest"]
    actual=hashlib.sha256(open(wave,"rb").read()).hexdigest()
    if actual != row["wave_manifest_sha256"]:
        raise SystemExit(f"wave manifest hash drift: {wave}")
    payload=json.load(open(wave))
    tasks=payload.get("tasks",[])
    if not 1 <= len(tasks) <= 5:
        raise SystemExit(f"invalid task count: {wave}")
    for task in tasks:
        fields=(task["arm_id"],str(int(task["source_iteration"])),task["source_checkpoint_root"],task["eval_output_root"])
        if any(any(c in value for c in "\t\r\n") for value in fields):
            raise SystemExit("unsafe backlog manifest field")
        print("\t".join(fields))
PY
)

for ((base=0; base<${#rows[@]}; base+=16)); do
  declare -A pids=()
  for ((offset=0; offset<16 && base+offset<${#rows[@]}; offset++)); do
    IFS=$'\t' read -r arm iteration source output <<<"${rows[base+offset]}"
    label="${arm}_iteration_${iteration}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --gres=gpu:1 \
      --cpus-per-task=64 --mem=105G --cpu-bind=cores \
      --output="$batch_root/logs/${label}.out" --error="$batch_root/logs/${label}.err" \
      env \
        EVALUATION_BUNDLE="$EVALUATION_BUNDLE" \
        NATIVE_GREEK_EVAL_ROOT="$NATIVE_GREEK_EVAL_ROOT" \
        MEGATRON_DIR="$MEGATRON_DIR" TOKENIZER_DIR="$TOKENIZER_DIR" \
        PYTHON_COMPAT_DIR="$PYTHON_COMPAT_DIR" \
        CHECKPOINT_EVALUATION_PLAN="$CHECKPOINT_EVALUATION_PLAN" \
        GREEKMMLU_CLEAN_SUBSET="$GREEKMMLU_CLEAN_SUBSET" \
        CAMPAIGN_MANIFEST="$CAMPAIGN_MANIFEST" \
        EVALUATION_NAMESPACE=fp32_v1 EVAL_DTYPE=float32 \
        SOURCE_CHECKPOINT_ROOT="$source" SOURCE_ITERATION="$iteration" \
        ARM_ID="$arm" EVAL_OUTPUT_ROOT="$output" \
        bash "$worker" &
    pids[$!]=$label
  done
  failed=0
  for pid in "${!pids[@]}"; do
    wait "$pid" || { echo "checkpoint failed: ${pids[$pid]}" >&2; failed=1; }
  done
  ((failed == 0))
done

"$HOST_PYTHON" - "$BACKLOG_BATCH_MANIFEST" "$batch_root/completion_receipt.json" <<'PY'
import datetime,json,os,sys,tempfile
source,output=sys.argv[1:]
d=json.load(open(source))
value={
 "schema_version":"apertus_mini_greekmmlu_backlog_batch_completion_v1",
 "status":"completed","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "batch_manifest":os.path.realpath(source),"wave_count":d["wave_count"],
 "evaluation_namespace":d["evaluation_namespace"],
 "authoritative_evaluation_dtype":d["authoritative_evaluation_dtype"],
 "scientific_execution_unchanged":True,
}
fd,tmp=tempfile.mkstemp(prefix=".completion.",dir=os.path.dirname(output))
with os.fdopen(fd,"w") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n")
os.replace(tmp,output)
PY

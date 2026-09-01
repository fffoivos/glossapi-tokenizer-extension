#!/usr/bin/env bash
#SBATCH --job-name=mini_greekmmlu_wave
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=00:30:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set frozen NumPy compatibility shim directory}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint evaluation plan}"
: "${WAVE_MANIFEST:?set immutable one-to-five checkpoint wave manifest}"
: "${WAVE_OUTPUT_ROOT:?set unique wave log root}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen decontaminated GreekMMLU subset}"

mkdir -p "$WAVE_OUTPUT_ROOT/logs"
mapfile -t task_rows < <("$HOST_PYTHON" - "$WAVE_MANIFEST" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("schema_version") != "apertus_mini_greekmmlu_wave_v1":
    raise SystemExit("invalid GreekMMLU wave schema")
tasks=payload.get("tasks",[])
if not 1 <= len(tasks) <= 5:
    raise SystemExit("a GreekMMLU wave must contain one to five checkpoints")
seen=set()
for task in tasks:
    fields=(task["arm_id"],str(int(task["source_iteration"])),task["source_checkpoint_root"],task["eval_output_root"])
    if any(any(c in value for c in "\t\r\n") for value in fields):
        raise SystemExit("tabs and newlines are forbidden in wave task fields")
    key=fields[:2]
    if key in seen:
        raise SystemExit(f"duplicate wave task: {key}")
    seen.add(key)
    print("\t".join(fields))
PY
)

worker="$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu_one.sh"
[[ -f "$worker" ]] || { echo "missing frozen one-checkpoint worker: $worker" >&2; exit 2; }

for ((base=0; base<${#task_rows[@]}; base+=4)); do
  pids=()
  for ((offset=0; offset<4 && base+offset<${#task_rows[@]}; offset++)); do
    IFS=$'\t' read -r arm iteration source_root output_root <<<"${task_rows[base+offset]}"
    label="${arm}_iter${iteration}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --gres=gpu:1 \
      --cpus-per-task=64 --mem=105G --cpu-bind=cores \
      --output="$WAVE_OUTPUT_ROOT/logs/${label}.out" \
      --error="$WAVE_OUTPUT_ROOT/logs/${label}.err" \
      env \
        EVALUATION_BUNDLE="$EVALUATION_BUNDLE" \
        NATIVE_GREEK_EVAL_ROOT="$NATIVE_GREEK_EVAL_ROOT" \
        MEGATRON_DIR="$MEGATRON_DIR" TOKENIZER_DIR="$TOKENIZER_DIR" \
        PYTHON_COMPAT_DIR="$PYTHON_COMPAT_DIR" \
        CHECKPOINT_EVALUATION_PLAN="$CHECKPOINT_EVALUATION_PLAN" \
        GREEKMMLU_CLEAN_SUBSET="$GREEKMMLU_CLEAN_SUBSET" \
        SOURCE_CHECKPOINT_ROOT="$source_root" SOURCE_ITERATION="$iteration" \
        ARM_ID="$arm" EVAL_OUTPUT_ROOT="$output_root" \
        bash "$worker" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || { echo "one or more native-GreekMMLU lanes failed" >&2; exit 1; }
done

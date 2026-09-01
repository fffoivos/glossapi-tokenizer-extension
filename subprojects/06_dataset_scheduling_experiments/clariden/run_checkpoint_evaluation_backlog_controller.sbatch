#!/usr/bin/env bash
#SBATCH --job-name=mini-greek-backlog-controller
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${PACKING_BUNDLE:?set immutable packing bundle}"
: "${EVALUATION_BUNDLE:?set immutable evaluator bundle}"
: "${BACKLOG_BATCH_ROOT:?set frozen split-manifest root}"
: "${CONTROLLER_ROOT:?set unique controller receipt root}"
: "${RUN_ROOT:?set campaign run root}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned evaluation Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set compatibility shim}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint plan}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen clean subset}"
: "${OPERATIONAL_RECOVERY_RECEIPT:?set evaluation recovery receipt}"

[[ ! -e "$CONTROLLER_ROOT" ]] || { echo "refusing to overwrite $CONTROLLER_ROOT" >&2; exit 2; }
mkdir -p "$CONTROLLER_ROOT" "$RUN_ROOT/logs"
mapfile -t manifests < <(find "$BACKLOG_BATCH_ROOT" -maxdepth 1 -type f \
  -name 'flat16_batch_[0-9][0-9].json' -print | sort)
[[ "${#manifests[@]}" -ge 1 && "${#manifests[@]}" -le 5 ]] || {
  echo "expected between one and five backlog manifests" >&2
  exit 2
}

wait_terminal() {
  local job=$1 state
  while true; do
    state=$(sacct -j "$job" -X -n -P -o State | sed -n '1{s/[ +].*//;p;}')
    case "$state" in
      COMPLETED) return 0 ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|PREEMPTED|REVOKED)
        echo "debug batch $job ended as $state" >&2; return 1 ;;
    esac
    sleep 15
  done
}

submit_debug() {
  local manifest=$1 index=$2 output attempt
  for attempt in $(seq 1 18); do
    if output=$(sbatch --parsable --nodes=4 --time=00:22:00 \
      --job-name="mini_cpt5_greek_flat16_${index}" \
      --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
      --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,NATIVE_GREEK_EVAL_ROOT=$NATIVE_GREEK_EVAL_ROOT,MEGATRON_DIR=$MEGATRON_DIR,TOKENIZER_DIR=$TOKENIZER_DIR,PYTHON_COMPAT_DIR=$PYTHON_COMPAT_DIR,CHECKPOINT_EVALUATION_PLAN=$CHECKPOINT_EVALUATION_PLAN,GREEKMMLU_CLEAN_SUBSET=$GREEKMMLU_CLEAN_SUBSET,CAMPAIGN_MANIFEST=$CAMPAIGN_MANIFEST,EVALUATION_NAMESPACE=fp32_v1,EVAL_DTYPE=float32,BACKLOG_BATCH_MANIFEST=$manifest" \
      "$PACKING_BUNDLE/clariden/run_checkpoint_evaluation_backlog_batch.sbatch" \
      2>"$CONTROLLER_ROOT/submit_${index}_attempt_${attempt}.err"); then
      printf '%s\n' "$output"; return 0
    fi
    sleep 10
  done
  echo "could not submit debug batch $index" >&2
  return 1
}

jobs=()
for index in "${!manifests[@]}"; do
  job=$(submit_debug "${manifests[$index]}" "$index")
  jobs+=("$job")
  printf '%s\t%s\t%s\n' "$index" "$job" "${manifests[$index]}" \
    >> "$CONTROLLER_ROOT/debug_jobs.tsv"
  wait_terminal "$job"
done

watch_exports="ALL,SCIENTIFIC_BUNDLE=$EVALUATION_BUNDLE,CAMPAIGN_MANIFEST=$CAMPAIGN_MANIFEST,RUN_ROOT=$RUN_ROOT,RUN_TAG=mini_cpt5_20260803T074854Z,SEGMENT_ID=0,TRAIN_JOB_ID=2989298,SEGMENT_ATTEMPT=0,EVALUATION_MEGATRON_DIR=$MEGATRON_DIR,OPERATIONAL_RECOVERY_RECEIPT=$OPERATIONAL_RECOVERY_RECEIPT,EVALUATION_NAMESPACE=fp32_v1,EVAL_DTYPE=float32"
watch=$(sbatch --parsable --job-name=mini_cpt5_20260803T074854Z_watch0_fp32v1r4 \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$watch_exports" "$EVALUATION_BUNDLE/clariden/watch_checkpoint_evaluations.sbatch")
supervisor=$(sbatch --parsable --job-name=mini_cpt5_20260803T074854Z_supervise0_fp32v4 \
  --output="$RUN_ROOT/logs/%x-%j.out" --error="$RUN_ROOT/logs/%x-%j.err" \
  --export="$watch_exports,WATCH_JOB_ID=$watch" \
  "$EVALUATION_BUNDLE/clariden/supervise_production_segment.sbatch")

"$HOST_PYTHON" - "$CONTROLLER_ROOT/completion_receipt.json" "$watch" "$supervisor" "${jobs[@]}" <<'PY'
import datetime,json,os,sys,tempfile
output,watch,supervisor,*jobs=sys.argv[1:]
value={
 "schema_version":"apertus_mini_greekmmlu_backlog_controller_completion_v1",
 "status":"completed","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "debug_batch_jobs":jobs,"watcher_job":watch,"supervisor_job":supervisor,
 "evaluation_namespace":"fp32_v1","authoritative_evaluation_dtype":"float32",
 "scientific_execution_unchanged":True,"training_runtime_changed":False,
}
fd,tmp=tempfile.mkstemp(prefix=".controller.",dir=os.path.dirname(output))
with os.fdopen(fd,"w") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
os.replace(tmp,output)
PY

#!/usr/bin/env bash
#SBATCH --job-name=early-wsd-supervise
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${EARLY_CODE_ROOT:?set immutable code root}"
: "${EARLY_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${EARLY_CONTRACT:?set experiment contract}"
: "${EARLY_BRANCH_RECIPE:?set branch recipe}"
: "${EARLY_LAUNCH_GATE:?set launch gate}"
: "${EARLY_OPERATIONAL_GATE:?set operational gate}"
: "${EARLY_RUN_ROOT:?set run root}"
: "${EARLY_TRAIN_JOB_ID:?set completed training job id}"
: "${EARLY_PHASE:?set branch}"
[[ "$EARLY_PHASE" == branch ]] || { echo "invalid supervisor phase" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "supervisor must run on debug" >&2; exit 2; }

/usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$EARLY_CODE_ROOT" --receipt "$EARLY_CODE_BUNDLE_RECEIPT" --kind scientific
receipt="$EARLY_RUN_ROOT/training_receipts/${EARLY_PHASE}_job_${EARLY_TRAIN_JOB_ID}.json"
if [[ ! -f "$receipt" ]]; then
  blocker="$EARLY_RUN_ROOT/orchestration/${EARLY_PHASE}_missing_receipt_${EARLY_TRAIN_JOB_ID}.json"
  /usr/bin/python3.11 - "$blocker" "$EARLY_PHASE" "$EARLY_TRAIN_JOB_ID" <<'PY'
import datetime,json,os,sys,tempfile
out,phase,job=sys.argv[1:]
value={'schema_version':'apertus_full8_early_cooldown_campaign_blocker_v1','status':'blocked','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'job_id':job,'reason':'terminal_job_produced_no_classified_receipt','automatic_retry':False}
os.makedirs(os.path.dirname(out),exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='.blocker.',suffix='.partial',dir=os.path.dirname(out))
with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out)
PY
  echo "${EARLY_PHASE} produced no classified receipt; refusing blind retry" >&2
  exit 2
fi
readarray -t state < <(/usr/bin/python3.11 - "$receipt" "$EARLY_TRAIN_JOB_ID" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('schema_version')!='apertus_full8_early_cooldown_training_v2' or d.get('job_id')!=sys.argv[2]: raise SystemExit('training receipt/job drift')
print(d['status']); print(d.get('last_logged_iteration') or 0)
PY
)
common="ALL,EARLY_CODE_ROOT=$EARLY_CODE_ROOT,EARLY_CODE_BUNDLE_RECEIPT=$EARLY_CODE_BUNDLE_RECEIPT,EARLY_CONTRACT=$EARLY_CONTRACT,EARLY_BRANCH_RECIPE=$EARLY_BRANCH_RECIPE,EARLY_LAUNCH_GATE=$EARLY_LAUNCH_GATE,EARLY_OPERATIONAL_GATE=$EARLY_OPERATIONAL_GATE,EARLY_RUN_ROOT=$EARLY_RUN_ROOT"
mkdir -p "$EARLY_RUN_ROOT/logs" "$EARLY_RUN_ROOT/orchestration"
if [[ "${state[0]}" == branch_completed && "${state[1]}" == 13193 ]]; then
  child=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=4 --time=01:29:00 \
    --job-name=early_wsd_eval_10728 --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,EARLY_ITERATION=10728,EARLY_EVAL_INDEX=0,EARLY_TRAIN_JOB_ID=$EARLY_TRAIN_JOB_ID" \
    "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/clariden/run_checkpoint_evaluation_debug.sbatch")
  role=evaluation_10728
elif [[ "${state[0]}" == branch_gracefully_stopped && "${state[1]}" -gt 9536 && "${state[1]}" -lt 13193 ]]; then
  child=$(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=12:00:00 --switches=1 \
    --job-name=early_wsd_recovery --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,EARLY_PHASE=branch,EARLY_RECOVERY_MODE=1" \
    "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/clariden/train_and_gate.sbatch")
  next_supervisor=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --nodes=1 --time=00:10:00 \
    --dependency="afterany:$child" --job-name=early_wsd_supervise_recovery \
    --output="$EARLY_RUN_ROOT/logs/%x-%j.out" --error="$EARLY_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,EARLY_PHASE=branch,EARLY_TRAIN_JOB_ID=$child" \
    "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/clariden/supervise_after_training_debug.sbatch")
  role=recovery_train
else
  blocker="$EARLY_RUN_ROOT/orchestration/${EARLY_PHASE}_classified_blocker_${EARLY_TRAIN_JOB_ID}.json"
  /usr/bin/python3.11 - "$blocker" "$EARLY_PHASE" "$EARLY_TRAIN_JOB_ID" "${state[0]}" "${state[1]}" <<'PY'
import datetime,json,os,sys,tempfile
out,phase,job,status,iteration=sys.argv[1:]; value={'schema_version':'apertus_full8_early_cooldown_campaign_blocker_v1','status':'blocked','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'job_id':job,'classified_training_status':status,'last_logged_iteration':int(iteration),'automatic_retry':False}
fd,tmp=tempfile.mkstemp(prefix='.blocker.',suffix='.partial',dir=os.path.dirname(out))
with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out)
PY
  echo "unsupported classified terminal state: phase=$EARLY_PHASE status=${state[0]} iteration=${state[1]}" >&2
  exit 2
fi
if ! /usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/audit_submitted_job.py" \
  --job-id "$child" --role "$role" --output "$EARLY_RUN_ROOT/orchestration/${role}_${child}.json"; then
  scancel "$child"; [[ -z "${next_supervisor:-}" ]] || scancel "$next_supervisor"; exit 1
fi
if [[ -n "${next_supervisor:-}" ]]; then
  if ! /usr/bin/python3.11 "$EARLY_CODE_ROOT/subprojects/10_early_cooldown_causal_experiment/scripts/audit_submitted_job.py" \
    --job-id "$next_supervisor" --role recovery_supervisor --output "$EARLY_RUN_ROOT/orchestration/recovery_supervisor_${next_supervisor}.json"; then
    scancel "$next_supervisor" "$child"; exit 1
  fi
fi
printf '%s\n' "$child"

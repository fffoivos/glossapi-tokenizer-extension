#!/usr/bin/env bash
# Lightweight Mac-side coordinator for the one safe v19 -> v29 supervisor swap.
# It performs no data/GPU work. The immutable v29 helper remains responsible for
# all receipt checks, resource audit, and the only permitted cancellation.
set -euo pipefail
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH

interval_seconds=${FULL8_WATCH_INTERVAL_SECONDS:-60}
max_checks=${FULL8_WATCH_MAX_CHECKS:-360}
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "invalid interval" >&2; exit 2; }
[[ "$max_checks" =~ ^[1-9][0-9]*$ ]] || { echo "invalid max checks" >&2; exit 2; }

for ((check=1; check<=max_checks; check++)); do
  if result=$(ssh -o BatchMode=yes clariden bash -s <<'REMOTE'
set -euo pipefail
science=/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt/20260808T023300Z-sanitized-v45
ops=/iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/20260809T030000Z-prequeue-v30-legacy-receipt-bridge
stage=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge
run=/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12
pre=/iopsstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_prelaunch/20260808T121000Z-resource-aware-v12
old_receipt="$run/orchestration/supervisor_submission_receipts/segment_1.json"
transition="$run/orchestration/supervisor_transitions/segment_1_v30.json"
if [[ -f "$transition" ]]; then
  /usr/bin/python3.11 - "$transition" <<'PY'
import json,sys
v=json.load(open(sys.argv[1],encoding='utf-8'))
print('DONE '+json.dumps(v,sort_keys=True))
PY
  exit 0
fi
if [[ ! -f "$old_receipt" ]]; then
  /usr/bin/python3.11 "$ops/scripts/reconstruct_legacy_supervisor_receipt.py" \
    --run-root "$run" \
    --prequeued-manifest "$run/submissions/prequeued_launch_graph_v1.json" \
    --allocation-routing-receipt "$run/orchestration/allocation_receipts/supervisor_3037861.json" \
    --operational-root /iopsstor/scratch/cscs/fffoivos/orchestration/full8b-cpt-ops/20260808T224500Z-prequeue-v19 \
    --supervisor-job 3037861 --next-train-job 3037145 \
    --source-segment 0 --next-segment 1 --next-segment-start 4000 \
    --completed-iteration 3576 --output "$old_receipt"
fi
old_job=$(/usr/bin/python3.11 - "$old_receipt" <<'PY'
import json,sys
v=json.load(open(sys.argv[1],encoding='utf-8'))
expected={
 'schema_version':'apertus_full_8b_supervisor_submission_v1',
 'status':'passed', 'next_segment':1, 'next_segment_start':4000,
 'source_train_job':'3037145',
}
if any(v.get(k)!=x for k,x in expected.items()):
 raise SystemExit('legacy receipt binding drift')
job=str(v.get('supervisor_job',''))
if not job.isdigit(): raise SystemExit('legacy receipt lacks numeric supervisor id')
print(job)
PY
)
state=$(squeue -h -j "$old_job" -o '%T' || true)
if [[ "$state" != PENDING ]]; then
  echo "WAIT legacy_supervisor_${old_job}_state_${state:-ABSENT}"
  exit 0
fi
debug_rows=$(squeue -h -u fffoivos -p debug -o '%i|%T' || true)
if [[ "$debug_rows" != "$old_job|PENDING" ]]; then
  echo "WAIT debug_slots_not_exclusive $(tr '\n' ',' <<<"$debug_rows")"
  exit 0
fi
/usr/bin/python3.11 "$ops/scripts/transition_pending_supervisor.py" \
  --scientific-root "$science" --scientific-receipt "$science.receipt.json" \
  --ops-root "$ops" --ops-receipt "$ops.receipt.json" \
  --stage-root "$stage" --run-root "$run" \
  --initial-megatron /capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/20260731T124000Z-cpt25b-v1/megatron_tp2_r17patched \
  --selected-profile "$stage/contracts/rebind_v3/selected_execution_profile.json" \
  --launch-gate "$pre/launch_gate.json" --prelaunch-root "$pre" \
  --recipe "$stage/contracts/rebind_v3/recipe_8b_full_mixed.sanitized.json" \
  --profiles "$stage/contracts/rebind_v3/execution_profiles.sanitized.json" \
  --train-leaf-switch group29 \
  --prequeued-manifest "$run/submissions/prequeued_launch_graph_v1.json" \
  --prequeue-schedule "$ops/configs/prequeue_schedule_8b.json" \
  --old-supervisor-job "$old_job" --old-supervisor-receipt "$old_receipt" \
  --segment 1 --attempt 0 --attempt-start 4000 --source-train-job 3037145 \
  --output "$transition"
echo "TRANSITIONED old=$old_job receipt=$transition"
REMOTE
); then
    :
  else
    printf '%s check=%s WAIT ssh_or_remote_error\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$check"
    sleep "$interval_seconds"
    continue
  fi
  printf '%s check=%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$check" "$result"
  [[ "$result" == DONE* || "$result" == TRANSITIONED* ]] && exit 0
  sleep "$interval_seconds"
done
echo "watch exhausted after $max_checks checks without a transition" >&2
exit 1

#!/usr/bin/env bash
# Bounded, allocation-free observer for a submitted causal early-cooldown run.
# It intentionally has no scheduler-mutation or data-processing path.
set -euo pipefail

usage() {
  echo "usage: $0 RUN_ROOT TRAINING_JOB REPLAY_UPLOAD_JOB STATUS_LOG [MAX_SECONDS]" >&2
  exit 2
}

[[ $# -ge 4 && $# -le 5 ]] || usage
run_root=$1
training_job=$2
replay_upload_job=$3
status_log=$4
maximum_seconds=${5:-108000}
[[ "$training_job" =~ ^[0-9]+$ && "$replay_upload_job" =~ ^[0-9]+$ ]] || usage
[[ "$maximum_seconds" =~ ^[1-9][0-9]*$ ]] || usage
[[ -d "$run_root" && -d "$(dirname "$status_log")" ]] || {
  echo "run root or status-log directory does not exist" >&2
  exit 2
}

deadline=$(( $(date +%s) + maximum_seconds ))
previous=""
json_status() {
  local path=$1
  if [[ -f "$path" ]]; then
    /usr/bin/python3.11 - "$path" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("status", "unknown"))
PY
  else
    printf 'absent\n'
  fi
}

while [[ $(date +%s) -lt $deadline ]]; do
  train=$(squeue -h -j "$training_job" -o '%T|%M|%R' 2>/dev/null || true)
  upload=$(squeue -h -j "$replay_upload_job" -o '%T|%M|%R' 2>/dev/null || true)
  parity=$(json_status "$run_root/branch_control/sandwich_restart_control_receipt.json")
  terminal=$(json_status "$run_root/training_receipts/branch_job_${training_job}.json")
  eval_count=$( { find "$run_root/checkpoint_evaluations" -name evaluation_receipt.json -type f 2>/dev/null || true; } | wc -l | tr -d ' ')
  endpoint=$(json_status "$run_root/native_greek_endpoint/native_endpoint_receipt.json")
  snapshot="train=${train:-terminal};upload=${upload:-terminal};parity=$parity;terminal=$terminal;checkpoint_evaluations=$eval_count;native_endpoint=$endpoint"
  if [[ "$snapshot" != "$previous" ]]; then
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$snapshot" >>"$status_log"
    previous=$snapshot
  fi
  if [[ "$terminal" != absent && "$eval_count" == 4 && "$endpoint" == completed ]]; then
    exit 0
  fi
  sleep 300
done

printf '%s watcher_timeout maximum_seconds=%s\n' "$(date -u +%FT%TZ)" "$maximum_seconds" >>"$status_log"
exit 0

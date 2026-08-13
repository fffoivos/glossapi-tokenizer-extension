#!/bin/bash
# A launchd-safe, allocation-free observer. It only reads Slurm and receipts.
set -euo pipefail

usage() {
  echo "usage: $0 RUN_ROOT TRAINING_JOB REPLAY_UPLOAD_JOB STATE_FILE LOG_FILE [MAX_SECONDS]" >&2
  exit 2
}

[[ $# == 5 || $# == 6 ]] || usage
run_root=$1
training_job=$2
replay_upload_job=$3
state_file=$4
log_file=$5
maximum_seconds=${6:-0}
[[ "$training_job" =~ ^[0-9]+$ && "$replay_upload_job" =~ ^[0-9]+$ ]] || usage
[[ "$maximum_seconds" =~ ^[0-9]+$ ]] || usage

state_dir=$(dirname "$state_file")
[[ -d "$state_dir" && -d "$(dirname "$log_file")" ]] || {
  echo "state/log directory does not exist" >&2
  exit 2
}
lock_dir="${state_file}.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

observe() {
snapshot=$(
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=20 clariden \
    /usr/bin/env RUN_ROOT="$run_root" TRAINING_JOB="$training_job" REPLAY_UPLOAD_JOB="$replay_upload_job" \
    /bin/bash -s <<'REMOTE'
set -u
status() {
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
queue_state() {
  local job=$1 label=$2
  local row
  row=$(squeue -h -j "$job" -o '%T|%R' 2>/dev/null || true)
  if [[ -n "$row" ]]; then
    printf '%s=%s\n' "$label" "$row"
  else
    row=$(sacct -j "$job" -X -n -P --format=State 2>/dev/null | head -1 | tr -d ' ')
    printf '%s=%s\n' "$label" "${row:-unknown}"
  fi
}
queue_state "$TRAINING_JOB" training
queue_state "$REPLAY_UPLOAD_JOB" replay_upload
printf 'parity=%s\n' "$(status "$RUN_ROOT/branch_control/sandwich_restart_control_receipt.json")"
printf 'terminal=%s\n' "$(status "$RUN_ROOT/training_receipts/branch_job_${TRAINING_JOB}.json")"
printf 'evaluations=%s\n' "$( { find "$RUN_ROOT/checkpoint_evaluations" -name evaluation_receipt.json -type f 2>/dev/null || true; } | wc -l | tr -d ' ')"
printf 'native_endpoint=%s\n' "$(status "$RUN_ROOT/native_greek_endpoint/native_endpoint_receipt.json")"
REMOTE
)
snapshot=$(printf '%s\n' "$snapshot" | tr '\n' ';')
if [[ ! -f "$state_file" ]] || ! cmp -s <(printf '%s' "$snapshot") "$state_file"; then
  temporary="${state_file}.$$.partial"
  printf '%s' "$snapshot" >"$temporary"
  mv "$temporary" "$state_file"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$snapshot" >>"$log_file"
fi
}

observe
[[ "$maximum_seconds" -gt 0 ]] || exit 0
deadline=$(( $(date +%s) + maximum_seconds ))
while [[ $(date +%s) -lt $deadline ]]; do
  sleep 300
  observe
  if grep -q 'terminal=branch_completed;.*evaluations=4;native_endpoint=completed;' "$state_file"; then
    exit 0
  fi
done
printf '%s watcher_timeout maximum_seconds=%s\n' "$(date -u +%FT%TZ)" "$maximum_seconds" >>"$log_file"

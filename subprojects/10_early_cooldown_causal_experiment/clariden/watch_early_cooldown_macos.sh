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

notify_transition() {
  local previous=$1 current=$2 message=

  # Do not alert for benign scheduler-reason changes. Alert only when an event
  # needs an operator/agent follow-up or establishes a scientific milestone.
  if [[ "$current" == *'training=RUNNING|'* && "$previous" != *'training=RUNNING|'* ]]; then
    message="Training job ${training_job} has started."
  elif [[ "$current" == *'parity=passed;'* && "$previous" != *'parity=passed;'* ]]; then
    message="In-allocation parity control passed."
  elif [[ "$current" == *'parity=failed;'* && "$previous" != *'parity=failed;'* ]]; then
    message="Parity control failed; training requires review."
  elif [[ "$current" == *'terminal=branch_completed;'* && "$previous" != *'terminal=branch_completed;'* ]]; then
    message="Training completed; checkpoint evaluations are queued."
  elif [[ "$current" =~ terminal=([^\;]+)\; ]]; then
    terminal_status=${BASH_REMATCH[1]}
    if [[ "$terminal_status" != "absent" && "$previous" != *"terminal=${terminal_status};"* ]]; then
      message="Training stopped with ${terminal_status}; immediate review required."
    fi
  elif [[ "$current" == *'native_endpoint=completed;'* && "$previous" != *'native_endpoint=completed;'* ]]; then
    message="Native Greek endpoint evaluation completed."
  elif [[ "$current" =~ evaluations=([1-9][0-9]*)\; ]]; then
    evaluation_count=${BASH_REMATCH[1]}
    if [[ "$previous" != *"evaluations=${evaluation_count};"* ]]; then
      message="Checkpoint evaluation ${evaluation_count} completed."
    fi
  fi

  [[ -n "$message" ]] || return 0
  /usr/bin/osascript -e "display notification \"${message}\" with title \"Apertus early-cooldown\"" >/dev/null 2>&1 || true
}

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
  previous=$(cat "$state_file" 2>/dev/null || true)
  temporary="${state_file}.$$.partial"
  printf '%s' "$snapshot" >"$temporary"
  mv "$temporary" "$state_file"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$snapshot" >>"$log_file"
  notify_transition "$previous" "$snapshot"
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

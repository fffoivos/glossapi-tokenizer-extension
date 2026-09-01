#!/bin/bash
set -euo pipefail

# Mac-side coordinator only: it waits for a debug submit slot and launches no
# work locally. The active training/evaluation chain always has priority.
: "${D0_RELEASE_JOB_ID:?job whose completion frees one debug submit slot}"
: "${D0_GUARD_JOB_ID:?long-running training job protecting the launch window}"
: "${D0_ACTIVE_SUPERVISOR_JOB_ID:?remaining active-chain debug supervisor job}"
: "${D0_WRAPPER_ROOT:?frozen remote D0 wrapper root}"
: "${D0_EVAL_OUTPUT_ROOT:?new remote evaluation root}"

poll_seconds=${D0_POLL_SECONDS:-30}
minimum_guard_seconds=${D0_MINIMUM_GUARD_SECONDS:-3600}
pending_grace_seconds=${D0_PENDING_GRACE_SECONDS:-300}
launcher="$D0_WRAPPER_ROOT/evaluation/run_d0_0p5b_three_checkpoint_matrix.sbatch"
log_root="$(dirname "$D0_EVAL_OUTPUT_ROOT")/logs"

time_to_seconds() {
  /usr/bin/python3 - "$1" <<'PY'
import sys
value=sys.argv[1]
days=0
if '-' in value:
    raw_days,value=value.split('-',1); days=int(raw_days)
parts=[int(part) for part in value.split(':')]
if len(parts)==3: hours,minutes,seconds=parts
elif len(parts)==2: hours,minutes,seconds=0,*parts
else: raise SystemExit(2)
print(days*86400+hours*3600+minutes*60+seconds)
PY
}

ssh clariden "mkdir -p '$log_root'; test ! -e '$D0_EVAL_OUTPUT_ROOT'; test -f '$launcher'"
while true; do
  release_state=$(ssh clariden "sacct -X -n -P -j '$D0_RELEASE_JOB_ID' --format=State | head -1" | cut -d'|' -f1)
  case "$release_state" in
    COMPLETED*) ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "release job $D0_RELEASE_JOB_ID ended as $release_state; refusing opportunistic launch" >&2
      exit 2
      ;;
    *)
      echo "waiting: release_job=$D0_RELEASE_JOB_ID state=${release_state:-unknown}"
      sleep "$poll_seconds"
      continue
      ;;
  esac

  guard=$(ssh clariden "squeue -h -j '$D0_GUARD_JOB_ID' -o '%T|%L' | head -1")
  guard_state=${guard%%|*}
  if [[ "$guard_state" == RUNNING ]]; then
    remaining=${guard#*|}
    remaining_seconds=$(time_to_seconds "$remaining")
    if (( remaining_seconds < minimum_guard_seconds )); then
      echo "guard job has only $remaining remaining; wait for the active debug chain to finish" >&2
      exit 3
    fi
  elif [[ "$guard_state" != PENDING ]]; then
    echo "guard job is not safely pending/running: ${guard_state:-terminal_or_unknown}" >&2
    exit 3
  fi

  mapfile_output=$(ssh clariden "squeue -h -p debug -u fffoivos -o '%i' | sort -u")
  debug_jobs=$(printf '%s\n' "$mapfile_output" | sed '/^$/d')
  debug_count=$(printf '%s\n' "$debug_jobs" | sed '/^$/d' | wc -l | tr -d ' ')
  if [[ "$debug_count" != 1 ]] || [[ "$debug_jobs" != "$D0_ACTIVE_SUPERVISOR_JOB_ID" ]]; then
    echo "waiting: debug submitted jobs are [${debug_jobs//$'\n'/,}], expected only $D0_ACTIVE_SUPERVISOR_JOB_ID"
    sleep "$poll_seconds"
    continue
  fi

  export_arg="ALL,D0_WRAPPER_ROOT=$D0_WRAPPER_ROOT,D0_EVAL_OUTPUT_ROOT=$D0_EVAL_OUTPUT_ROOT"
  test_result=$(ssh clariden "sbatch --test-only --export='$export_arg' --output='$log_root/d0-0p5b-native3cp-%j.out' --error='$log_root/d0-0p5b-native3cp-%j.err' '$launcher'" 2>&1) || {
    echo "$test_result" >&2
    sleep "$poll_seconds"
    continue
  }
  echo "$test_result"
  job_id=$(ssh clariden "sbatch --parsable --export='$export_arg' --output='$log_root/d0-0p5b-native3cp-%j.out' --error='$log_root/d0-0p5b-native3cp-%j.err' '$launcher'")
  job_id=${job_id%%;*}
  audit=$(ssh clariden "scontrol show job -o '$job_id'")
  [[ "$audit" == *"Partition=debug"* && "$audit" == *"NumNodes=4"* && "$audit" == *"TimeLimit=00:22:30"* ]] || {
    ssh clariden "scancel '$job_id'"
    echo "submitted job failed resource audit: $audit" >&2
    exit 4
  }
  echo "submitted D0 native-Greek matrix job $job_id"

  waited=0
  while (( waited < pending_grace_seconds )); do
    state=$(ssh clariden "squeue -h -j '$job_id' -o '%T' | head -1")
    [[ "$state" == RUNNING ]] && break
    [[ -z "$state" ]] && break
    sleep "$poll_seconds"
    waited=$((waited + poll_seconds))
  done
  if [[ "${state:-}" == PENDING ]]; then
    ssh clariden "scancel '$job_id'"
    echo "cancelled our still-pending D0 evaluation after ${waited}s to protect the active chain" >&2
    exit 5
  fi

  while true; do
    state=$(ssh clariden "sacct -X -n -P -j '$job_id' --format=State | head -1" | cut -d'|' -f1)
    case "$state" in
      COMPLETED*) echo "D0 native-Greek matrix completed: job=$job_id root=$D0_EVAL_OUTPUT_ROOT"; exit 0 ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
        echo "D0 native-Greek matrix ended as $state: job=$job_id" >&2; exit 6 ;;
      *) echo "monitoring: job=$job_id state=${state:-unknown}"; sleep 60 ;;
    esac
  done
done

#!/bin/bash
set -euo pipefail

# Mac-side coordinator only. GPU work is performed by the receipt-bound remote
# wrapper. Keep at most one running/eligible segment plus one dependent
# successor in Clariden's two submitted debug slots.
: "${PEAK_WRAPPER_ROOT_REMOTE:?set frozen remote wrapper root}"
: "${PEAK_ASSETS_ROOT_REMOTE:?set frozen remote clean-subset asset root}"
: "${PEAK_OUTPUT_ROOT_REMOTE:?set new remote output root}"
poll_seconds=${PEAK_POLL_SECONDS:-30}
launcher="$PEAK_WRAPPER_ROOT_REMOTE/evaluation/run_peak_window_segment.sbatch"
log_root="$(dirname "$PEAK_OUTPUT_ROOT_REMOTE")/logs"
authorization_comment=AUTHORIZED_NATIVE_GREEK_PEAK_WINDOW_DO_NOT_CANCEL
job_ids=""
previous_job=""

remote_state() {
  ssh clariden "sacct -X -n -P -j '$1' --format=State | head -1" | cut -d'|' -f1
}

remote_reason() {
  ssh clariden "squeue -h -j '$1' -o '%R' | head -1"
}

ssh clariden "mkdir -p '$log_root'; test ! -e '$PEAK_OUTPUT_ROOT_REMOTE'; test -f '$launcher'; test -f '$PEAK_WRAPPER_ROOT_REMOTE/bundle_receipt.json'; test -f '$PEAK_ASSETS_ROOT_REMOTE/rebind_receipt.json'"

segment=0
while (( segment < 6 )); do
  if [[ -n "$previous_job" ]]; then
    state=$(remote_state "$previous_job")
    reason=$(remote_reason "$previous_job")
    if [[ "$reason" == DependencyNeverSatisfied* ]]; then
      echo "previous peak-window job $previous_job has impossible dependency: $reason" >&2
      exit 2
    fi
    case "$state" in
      COMPLETED*|RUNNING*|PENDING*) ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
        echo "previous peak-window job $previous_job ended as $state" >&2
        exit 2
        ;;
      *)
        echo "waiting for reliable state of $previous_job: ${state:-unknown}"
        sleep "$poll_seconds"
        continue
        ;;
    esac
  fi

  debug_count=$(ssh clariden "squeue -h -p debug -u fffoivos -o '%i' | sort -u | sed '/^$/d' | wc -l" | tr -d ' ')
  if (( debug_count >= 2 )); then
    echo "waiting for debug submit slot: submitted=$debug_count next_segment=$segment"
    sleep "$poll_seconds"
    continue
  fi

  export_arg="ALL,PEAK_WRAPPER_ROOT=$PEAK_WRAPPER_ROOT_REMOTE,PEAK_ASSETS_ROOT=$PEAK_ASSETS_ROOT_REMOTE,PEAK_OUTPUT_ROOT=$PEAK_OUTPUT_ROOT_REMOTE,PEAK_SEGMENT_INDEX=$segment,PEAK_RESUME=0"
  dependency_args=""
  if [[ -n "$previous_job" ]]; then
    dependency_args="--dependency=afterok:$previous_job"
  fi
  test_result=$(ssh clariden "sbatch --test-only --comment='$authorization_comment' $dependency_args --export='$export_arg' --output='$log_root/peak-window-seg${segment}-%j.out' --error='$log_root/peak-window-seg${segment}-%j.err' '$launcher'" 2>&1) || {
    echo "$test_result" >&2
    sleep "$poll_seconds"
    continue
  }
  echo "$test_result"
  job_id=$(ssh clariden "sbatch --parsable --comment='$authorization_comment' $dependency_args --export='$export_arg' --output='$log_root/peak-window-seg${segment}-%j.out' --error='$log_root/peak-window-seg${segment}-%j.err' '$launcher'")
  job_id=${job_id%%;*}
  audit=$(ssh clariden "scontrol show job -o '$job_id'")
  [[ "$audit" == *"Partition=debug"* && "$audit" == *"NumNodes=4"* && "$audit" == *"TimeLimit=00:22:00"* && "$audit" == *"Comment=$authorization_comment"* && "$audit" != *"Reason=QOSMaxNodeMinutesPerJob"* ]] || {
    ssh clariden "scancel '$job_id'"
    echo "submitted segment failed resource audit: $audit" >&2
    exit 3
  }
  echo "submitted peak-window segment=$segment job=$job_id dependency=${previous_job:-none}"
  job_ids="$job_ids $job_id"
  previous_job="$job_id"
  segment=$((segment + 1))
done

final_job=$previous_job
while true; do
  state=$(remote_state "$final_job")
  case "$state" in
    COMPLETED*) break ;;
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "final peak-window job $final_job ended as $state" >&2
      exit 4
      ;;
    *)
      echo "monitoring peak-window chain: final_job=$final_job state=${state:-unknown}"
      sleep 60
      ;;
  esac
done

for job_id in $job_ids; do
  state=$(remote_state "$job_id")
  [[ "$state" == COMPLETED* ]] || {
    echo "peak-window chain contains non-completed job $job_id: $state" >&2
    exit 5
  }
done
ssh clariden "test -f '$PEAK_OUTPUT_ROOT_REMOTE/segment_5_receipt.json'; test -f '$PEAK_OUTPUT_ROOT_REMOTE/peak_window_results.json'"
echo "peak-window evaluation completed: jobs=$job_ids root=$PEAK_OUTPUT_ROOT_REMOTE"

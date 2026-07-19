#!/usr/bin/env bash
# Run a non-overlapping downstream task range with bounded local concurrency.
set -eEuo pipefail

: "${STAGE:?STAGE is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${TASK_START:?TASK_START is required}"
: "${TASK_END:?TASK_END is required (exclusive)}"
: "${TASK_CONCURRENCY:?TASK_CONCURRENCY is required}"
: "${ATTEMPT_ROOT:?ATTEMPT_ROOT is required}"

case "$STAGE" in
  bucket|shingles|verify|filter) ;;
  *) echo "unsupported bounded stage: $STAGE" >&2; exit 2 ;;
esac
[[ "$TASK_START" =~ ^[0-9]+$ ]]
[[ "$TASK_END" =~ ^[0-9]+$ ]]
[[ "$TASK_CONCURRENCY" =~ ^[1-5]$ ]]
(( TASK_END > TASK_START ))

stage_script="$PIPELINE_ROOT/slurm/agent1_v5_eiger/stage.sh"
[[ -x "$stage_script" ]]
time_bin="${TIME_BIN:-/usr/bin/time}"
[[ -x "$time_bin" ]]
logs_root="$ATTEMPT_ROOT/task-logs"
metrics_root="$ATTEMPT_ROOT/metrics"
stop_sentinel="${STOP_SENTINEL:-$RUN_ROOT/dedup_downstream.stop}"
mkdir -p "$logs_root" "$metrics_root"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

drain_requested=0
trap 'drain_requested=1' USR1
drain_epoch="${SLURM_JOB_END_TIME:-0}"
if [[ "$drain_epoch" =~ ^[0-9]+$ ]] && (( drain_epoch > 0 )); then
  drain_epoch=$((drain_epoch - 900))
fi

should_drain() {
  [[ -e "$stop_sentinel" ]] && return 0
  (( drain_requested == 1 )) && return 0
  (( drain_epoch > 0 && $(date +%s) >= drain_epoch )) && return 0
  return 1
}

run_task() {
  local task="$1" start end status metric time_report out_log err_log
  start="$(date +%s)"
  time_report="$(mktemp "$metrics_root/.time-${task}.XXXXXX")"
  out_log="$logs_root/task-$(printf '%06d' "$task").out"
  err_log="$logs_root/task-$(printf '%06d' "$task").err"
  if "$time_bin" -v -o "$time_report" \
      env TASK_INDEX="$task" "$stage_script" >"$out_log" 2>"$err_log"; then
    status=passed
  else
    status=failed
  fi
  end="$(date +%s)"
  metric="$metrics_root/task-$(printf '%06d' "$task").json"
  python3 - "$metric" "$STAGE" "$task" "$start" "$end" "$status" \
    "$TASK_CONCURRENCY" "$time_report" "$out_log" "$err_log" \
    "${ATTEMPT_ID:-bounded-stage}" "${SLURM_JOB_ID:-local}" "$PIPELINE_ROOT" <<'PY'
import json
import os
import re
import sys
import tempfile

(
    path, stage, task, start, end, status, workers, time_report,
    out_log, err_log, attempt_id, job_id, pipeline_root,
) = sys.argv[1:]
values = {}
with open(time_report, encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        match = re.match(r"\s*([^:]+):\s*(.*)$", raw)
        if match:
            values[match.group(1).strip()] = match.group(2).strip()

def integer(label):
    value = values.get(label, "0").replace(",", "")
    return int(value) if value.isdigit() else 0

payload = {
    "schema_version": "agent1_v5_bounded_downstream_task_metric_v1",
    "status": status,
    "stage": stage,
    "task_index": int(task),
    "workers": int(workers),
    "started_epoch": int(start),
    "finished_epoch": int(end),
    "elapsed_seconds": int(end) - int(start),
    "max_rss_kib": integer("Maximum resident set size (kbytes)"),
    "user_cpu_seconds": values.get("User time (seconds)", "0"),
    "system_cpu_seconds": values.get("System time (seconds)", "0"),
    "filesystem_input_kib": integer("File system inputs"),
    "filesystem_output_kib": integer("File system outputs"),
    "attempt_id": attempt_id,
    "slurm_job_id": job_id,
    "pipeline_root": pipeline_root,
    "stdout": out_log,
    "stderr": err_log,
}
fd, temporary = tempfile.mkstemp(prefix=".metric-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
  rm -f "$time_report"
  [[ "$status" == passed ]]
}

if should_drain; then
  echo "stop sentinel or drain window present before $STAGE range" >&2
  exit 75
fi

status=0
for ((wave_start=TASK_START; wave_start<TASK_END; wave_start+=TASK_CONCURRENCY)); do
  if should_drain; then
    echo "drain requested before $STAGE task $wave_start" >&2
    exit 75
  fi
  wave_end=$((wave_start + TASK_CONCURRENCY))
  if (( wave_end > TASK_END )); then wave_end="$TASK_END"; fi
  pids=()
  for ((task=wave_start; task<wave_end; task++)); do
    run_task "$task" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  (( status == 0 )) || exit "$status"
done
exit "$status"

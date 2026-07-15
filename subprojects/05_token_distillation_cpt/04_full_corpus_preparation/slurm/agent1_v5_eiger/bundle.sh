#!/usr/bin/env bash
set -euo pipefail

: "${TASK_COUNT:?TASK_COUNT is required}"
: "${TASKS_PER_NODE:?TASKS_PER_NODE is required}"
: "${TASK_CONCURRENCY:?TASK_CONCURRENCY is required}"
: "${SLURM_ARRAY_TASK_ID:?bundle must run as a Slurm array}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"

START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_NODE))
END=$((START + TASKS_PER_NODE))
if (( END > TASK_COUNT )); then END="${TASK_COUNT}"; fi
mkdir -p "${RUN_ROOT}/slurm"

status=0
for ((wave_start=START; wave_start<END; wave_start+=TASK_CONCURRENCY)); do
  wave_end=$((wave_start + TASK_CONCURRENCY))
  if (( wave_end > END )); then wave_end="${END}"; fi
  pids=()
  for ((task=wave_start; task<wave_end; task++)); do
    TASK_INDEX="${task}" "${PIPELINE_ROOT}/slurm/agent1_v5_eiger/stage.sh" \
      > "${RUN_ROOT}/slurm/task-${STAGE}-${task}.out" \
      2> "${RUN_ROOT}/slurm/task-${STAGE}-${task}.err" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
done
exit "${status}"

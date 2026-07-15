#!/usr/bin/env bash
set -euo pipefail

: "${TASK_COUNT:?TASK_COUNT is required}"
: "${TASKS_PER_NODE:?TASKS_PER_NODE is required}"
: "${SLURM_ARRAY_TASK_ID:?bundle must run as a Slurm array}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"

START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_NODE))
END=$((START + TASKS_PER_NODE))
if (( END > TASK_COUNT )); then END="${TASK_COUNT}"; fi

status=0
for ((task=START; task<END; task++)); do
  TASK_INDEX="${task}" "${PIPELINE_ROOT}/slurm/agent1_v5_eiger/stage.sh" > "${RUN_ROOT}/slurm/task-${STAGE}-${task}.out" 2> "${RUN_ROOT}/slurm/task-${STAGE}-${task}.err" &
  while (( $(jobs -pr | wc -l) >= TASKS_PER_NODE )); do
    if ! wait -n; then status=1; fi
  done
done
while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then status=1; fi
done
exit "${status}"

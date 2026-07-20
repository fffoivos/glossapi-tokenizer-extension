#!/usr/bin/env bash
set -eEuo pipefail

: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${PAIR_CAPACITY_RECEIPT:?PAIR_CAPACITY_RECEIPT is required}"

work_parent="${SLURM_TMPDIR:-/tmp}"
work_directory="$work_parent/agent1-v5-pair-capacity-${SLURM_JOB_ID:-local}"
python="${VENV_ROOT:?VENV_ROOT is required}/bin/python"

exec uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$python" "$PIPELINE_ROOT/scripts/agent1_v5_pair_capacity_canary.py" \
  --work-directory "$work_directory" \
  --output "$PAIR_CAPACITY_RECEIPT" \
  --required-free-bytes "${PAIR_REQUIRED_FREE_BYTES:-68719476736}" \
  --probe-bytes "${PAIR_PROBE_BYTES:-268435456}" \
  --sqlite-rows "${PAIR_SQLITE_ROWS:-200000}"

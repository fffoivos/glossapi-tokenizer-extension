#!/usr/bin/env bash
# Run the receipt-bound Agent-1 v5 candidate quality audit on a compute node.
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${VENV_ROOT:?VENV_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != "normal" ]]; then
  echo "quality audit must run on the normal partition" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "immutable quality audit already exists: $OUTPUT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=4

uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$VENV_ROOT/bin/python" \
  "$PIPELINE_ROOT/scripts/audit_agent1_v5_release_quality.py" \
  --run-root "$RUN_ROOT" \
  --output "$OUTPUT" \
  --workers "${AUDIT_WORKERS:-16}" \
  --samples-per-source "${AUDIT_SAMPLES_PER_SOURCE:-3}"

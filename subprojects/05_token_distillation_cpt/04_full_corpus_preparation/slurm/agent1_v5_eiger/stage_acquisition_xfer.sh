#!/usr/bin/env bash
set -euo pipefail

: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${CONFIG:?CONFIG is required}"
: "${SOURCE_ACQUISITION_RECEIPT:?SOURCE_ACQUISITION_RECEIPT is required}"
: "${STAGED_INPUT_ROOT:?STAGED_INPUT_ROOT is required}"
: "${STAGED_ACQUISITION_RECEIPT:?STAGED_ACQUISITION_RECEIPT is required}"

python3 "${PIPELINE_ROOT}/scripts/stage_agent1_v5_acquisition.py" \
  --config "${CONFIG}" \
  --acquisition-receipt "${SOURCE_ACQUISITION_RECEIPT}" \
  --output-root "${STAGED_INPUT_ROOT}" \
  --output-receipt "${STAGED_ACQUISITION_RECEIPT}"

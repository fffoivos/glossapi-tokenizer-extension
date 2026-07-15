#!/usr/bin/env bash
set -euo pipefail

: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"

exec uenv run pytorch/v2.6.0:v1 --view=default -- \
  "${PIPELINE_ROOT}/slurm/agent1_v5_eiger/stage.sh"

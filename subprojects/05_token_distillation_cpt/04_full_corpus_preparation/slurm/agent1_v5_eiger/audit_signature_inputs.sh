#!/usr/bin/env bash
# One full immutable payload audit for the accelerated signature path.
set -eEuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"

coord="$(dirname "$RUN_ROOT")/.${RUN_ROOT##*/}.coord"
venv="$coord/runtime/venv/bin/python"
config="$PIPELINE_ROOT/configs/agent1_v5_eiger_pipeline.json"
dedup="$PIPELINE_ROOT/scripts/agent1_v5_datatrove.py"
output="$RUN_ROOT/dedup_full_input_audit.json"
[[ ! -e "$output" ]] || { echo "full input audit already exists: $output" >&2; exit 2; }

uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$venv" "$dedup" audit-signature-inputs --config "$config" \
  --contract "$RUN_ROOT/run_contract.json" \
  --combined-manifest "$RUN_ROOT/release-pre-dedup/manifests/combined_manifest.json" \
  --runtime-receipt "$RUN_ROOT/datatrove_runtime.json" --output "$output"

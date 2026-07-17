#!/usr/bin/env bash
# Run one full-file MinHash signature rank, verify its durable receipt, then
# submit exactly one successor.  This keeps the Clariden debug QOS to one
# running and one queued job without trusting a login-host watchdog.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_DIR COORD_DIR PIPELINE_ROOT TASK_INDEX LAST_TASK" >&2
  exit 2
fi

run=$1
coord=$2
pipeline=$3
task=$4
last=$5
config="$pipeline/configs/agent1_v5_eiger_pipeline.json"
self=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")

uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$coord/runtime/venv/bin/python" "$pipeline/scripts/agent1_v5_datatrove.py" \
  signature-task --config "$config" --contract "$run/run_contract.json" \
  --combined-manifest "$run/release-pre-dedup/manifests/combined_manifest.json" \
  --runtime-receipt "$run/datatrove_runtime.json" --task-index "$task"

receipt="$run/60-dedup/minhash-signatures/receipts/$(printf '%06d' "$task").json"
python3 - "$receipt" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    assert json.load(handle).get("status") == "passed"
PY

if [[ "$task" -ge "$last" ]]; then
  exit 0
fi

next=$((task + 1))
command="$self $run $coord $pipeline $next $last"
sbatch --parsable --uenv-passthrough=ignore \
  --job-name="a1v5-signature-chain-r${next}" --partition=debug --time=01:25:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
  --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" --wrap "$command"

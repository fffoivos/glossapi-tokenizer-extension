#!/usr/bin/env bash
# One-time checksum-bound successor of run_signature_task_chain.sh.
#
# It is installed atomically only while the preceding legacy rank is running.
# With no takeover request it preserves the original serial behaviour.  With a
# valid request it stops after exactly the requested rank, after validating the
# normal signature receipt and all 32 output files.
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
tool="$(dirname -- "$self")/agent1_v5_signature_takeover.py"
request="$run/dedup_acceleration_takeover_request.json"
stop_receipt="$run/dedup_acceleration_sentinel_stop.json"

[[ -f "$tool" ]] || { echo "missing signature takeover verifier: $tool" >&2; exit 1; }

run_takeover_tool() {
  uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
    "$coord/runtime/venv/bin/python" "$tool" "$@"
}

uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$coord/runtime/venv/bin/python" "$pipeline/scripts/agent1_v5_datatrove.py" \
  signature-task --config "$config" --contract "$run/run_contract.json" \
  --combined-manifest "$run/release-pre-dedup/manifests/combined_manifest.json" \
  --runtime-receipt "$run/datatrove_runtime.json" --task-index "$task"

receipt="$run/60-dedup/minhash-signatures/receipts/$(printf '%06d' "$task").json"
uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$coord/runtime/venv/bin/python" - "$pipeline" "$run" "$receipt" "$task" <<'PY'
import json
import sys
from pathlib import Path

pipeline, run, receipt = map(Path, sys.argv[1:4])
task = int(sys.argv[4])
sys.path.insert(0, str(pipeline / "scripts"))
import agent1_v5_datatrove as dedup

value = json.loads(receipt.read_text(encoding="utf-8"))
assert value.get("schema_version") == dedup.SIGNATURE_RECEIPT_SCHEMA
assert value.get("status") == "passed"
assert int(value.get("task_index", -1)) == int(task)
outputs = value.get("outputs")
assert isinstance(outputs, list) and len(outputs) == 32
for output in outputs:
    dedup.validate_file_receipt(output, root=run)
PY

# A present request is intentionally fail-closed.  It is checked after the
# rank's durable output closure and before any successor submission.
if [[ -e "$request" ]]; then
  run_takeover_tool validate-request --request "$request" --run-root "$run" \
    --coord-root "$coord" --legacy-pipeline-root "$pipeline" \
    --active-helper "$self" --takeover-tool "$tool" --task-index "$task"
  run_takeover_tool write-stop --request "$request" --run-root "$run" \
    --coord-root "$coord" --legacy-pipeline-root "$pipeline" \
    --active-helper "$self" --takeover-tool "$tool" --task-index "$task" \
    --signature-receipt "$receipt" --output "$stop_receipt"
  exit 0
fi

if [[ "$task" -ge "$last" ]]; then
  exit 0
fi

next=$((task + 1))
command="$self $run $coord $pipeline $next $last"
sbatch --parsable --uenv-passthrough=ignore \
  --job-name="a1v5-signature-chain-r${next}" --partition=debug --time=01:25:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
  --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" --wrap "$command"

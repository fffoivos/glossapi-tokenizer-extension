#!/usr/bin/env bash
# Execute one receipt-checked MinHash signature row group and, only after its
# passed receipt exists, enqueue one successor from inside the Slurm job.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RUN_DIR COORD_DIR CODE_DIR TASK_INDEX ROW_GROUP MAX_ROW_GROUP" >&2
  exit 2
fi

run=$1
coord=$2
code=$3
task_index=$4
group=$5
max_group=$6
pipeline="$code/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
config="$pipeline/configs/agent1_v5_eiger_pipeline.json"
self=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")

uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$coord/runtime/venv/bin/python" "$pipeline/scripts/agent1_v5_datatrove.py" \
  signature-row-group-task --config "$config" --contract "$run/run_contract.json" \
  --combined-manifest "$run/release-pre-dedup/manifests/combined_manifest.json" \
  --runtime-receipt "$run/datatrove_runtime.json" --task-index "$task_index" --row-group "$group"

receipt="$run/60-dedup/minhash-signatures/partial-receipts/$(printf '%05d' "$task_index")/group-$(printf '%03d' "$group").json"
python3 - "$receipt" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    assert json.load(handle).get("status") == "passed"
PY

if [[ "$group" -ge "$max_group" ]]; then
  exit 0
fi

next=$((group + 1))
command="$self $run $coord $code $task_index $next $max_group"
sbatch --parsable --uenv-passthrough=ignore \
  --job-name="a1v5-signature-r${task_index}g${next}" --partition=debug --time=01:25:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
  --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" --wrap "$command"

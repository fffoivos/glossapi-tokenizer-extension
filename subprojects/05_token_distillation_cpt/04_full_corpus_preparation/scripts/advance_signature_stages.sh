#!/usr/bin/env bash
# Conservative hand-off for the two oversized MinHash signature shards.  It
# only proceeds when every required partial receipt has passed and exits on a
# terminal merge job that lacks its corresponding passed receipt.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR COORD_DIR CODE_DIR" >&2
  exit 2
fi

run=$1
coord=$2
code=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pipeline="$code/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
config="$pipeline/configs/agent1_v5_eiger_pipeline.json"
log="$coord/signature-stage-advance.log"

partial_passed() {
  local rank=$1 group=$2
  local receipt="$run/60-dedup/minhash-signatures/partial-receipts/$(printf '%05d' "$rank")/group-$(printf '%03d' "$group").json"
  [[ -s "$receipt" ]] && python3 - "$receipt" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    raise SystemExit(0 if json.load(handle).get("status") == "passed" else 1)
PY
}

all_partials_passed() {
  local rank=$1 max_group=$2 group
  for group in $(seq 0 "$max_group"); do
    partial_passed "$rank" "$group" || return 1
  done
}

merge_passed() {
  local rank=$1
  local receipt="$run/60-dedup/minhash-signatures/receipts/$(printf '%06d' "$rank").json"
  [[ -s "$receipt" ]] && python3 - "$receipt" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    raise SystemExit(0 if json.load(handle).get("status") == "passed" else 1)
PY
}

active_merge() {
  local rank=$1
  squeue -h -u fffoivos -n "a1v5-merge-r${rank}" -o '%A' | head -n 1
}

terminal_merge_state() {
  local rank=$1
  sacct -X --noheader --parsable2 --user=fffoivos --name="a1v5-merge-r${rank}" \
    --format=State --starttime=2026-07-15 2>/dev/null | tail -n 1 | cut -d '|' -f 1
}

submit_merge() {
  local rank=$1 command
  command="uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME $coord/runtime/venv/bin/python $pipeline/scripts/agent1_v5_datatrove.py merge-signature-row-groups --config $config --contract $run/run_contract.json --combined-manifest $run/release-pre-dedup/manifests/combined_manifest.json --runtime-receipt $run/datatrove_runtime.json --task-index $rank"
  sbatch --parsable --uenv-passthrough=ignore --job-name="a1v5-merge-r${rank}" \
    --partition=debug --time=01:25:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
    --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" --wrap "$command"
}

wait_and_merge() {
  local rank=$1 max_group=$2
  until all_partials_passed "$rank" "$max_group"; do
    sleep 30
  done
  while ! merge_passed "$rank"; do
    if [[ -n $(active_merge "$rank") ]]; then
      sleep 30
      continue
    fi
    state=$(terminal_merge_state "$rank")
    if [[ -n "$state" ]]; then
      echo "$(date -Is) stopping: rank $rank merge terminal state $state without passed receipt" >> "$log"
      exit 1
    fi
    job=$(submit_merge "$rank")
    echo "$(date -Is) queued rank $rank merge as job $job" >> "$log"
    sleep 30
  done
  echo "$(date -Is) rank $rank merge receipt passed" >> "$log"
}

echo "$(date -Is) starting conservative oversized-signature hand-off" >> "$log"
wait_and_merge 3 14

if ! pgrep -f "roll_signature_row_group_batches.sh $run" >/dev/null; then
  nohup "$script_dir/roll_signature_row_group_batches.sh" "$run" "$coord" "$code" 6 96 8 \
    > "$coord/signature-r6-batch-roll.stdout" 2>&1 < /dev/null &
  echo "$(date -Is) started rank-6 batch queue keeper pid=$!" >> "$log"
fi

wait_and_merge 6 96
echo "$(date -Is) both oversized signature shards have passed merge receipts" >> "$log"

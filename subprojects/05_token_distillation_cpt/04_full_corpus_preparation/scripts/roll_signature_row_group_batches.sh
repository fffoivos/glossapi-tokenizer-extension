#!/usr/bin/env bash
# Maintain a one-job look-ahead queue for independently checkpointed MinHash
# row groups.  A batch is only a Slurm scheduling unit: each row group still
# produces and must pass its own receipt.  Stop on any unexplained terminal
# state rather than retrying or skipping it.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RUN_DIR COORD_DIR CODE_DIR TASK_INDEX MAX_ROW_GROUP BATCH_SIZE" >&2
  exit 2
fi

run=$1
coord=$2
code=$3
task_index=$4
max_group=$5
batch_size=$6
pipeline="$code/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
config="$pipeline/configs/agent1_v5_eiger_pipeline.json"
rank=$(printf '%05d' "$task_index")
receipt_dir="$run/60-dedup/minhash-signatures/partial-receipts/$rank"
log="$coord/signature-r${task_index}-batch-roll.log"

is_passed() {
  local group=$1
  local receipt="$receipt_dir/group-$(printf '%03d' "$group").json"
  [[ -s "$receipt" ]] && python3 - "$receipt" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    raise SystemExit(0 if json.load(handle).get("status") == "passed" else 1)
PY
}

batch_end() {
  local start=$1
  local end=$((start + batch_size - 1))
  if [[ "$end" -gt "$max_group" ]]; then
    end=$max_group
  fi
  echo "$end"
}

active_job_for() {
  local start=$1
  squeue -h -u fffoivos -n "a1v5-signature-r${task_index}-b${start}" -o '%A' | head -n 1
}

latest_terminal_state() {
  local start=$1
  sacct -X --noheader --parsable2 --user=fffoivos \
    --name="a1v5-signature-r${task_index}-b${start}" --format=State \
    --starttime=2026-07-15 2>/dev/null | tail -n 1 | cut -d '|' -f 1
}

submit_batch() {
  local start=$1
  local end
  end=$(batch_end "$start")
  local command="set -euo pipefail; for group in \$(seq $start $end); do uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME $coord/runtime/venv/bin/python $pipeline/scripts/agent1_v5_datatrove.py signature-row-group-task --config $config --contract $run/run_contract.json --combined-manifest $run/release-pre-dedup/manifests/combined_manifest.json --runtime-receipt $run/datatrove_runtime.json --task-index $task_index --row-group \$group; done"
  sbatch --parsable --uenv-passthrough=ignore \
    --job-name="a1v5-signature-r${task_index}-b${start}" --partition=debug --time=01:25:00 \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
    --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" \
    --wrap "$command"
}

first_missing_batch() {
  local group
  for group in $(seq 0 "$max_group"); do
    if ! is_passed "$group"; then
      echo $((group / batch_size * batch_size))
      return
    fi
  done
}

echo "$(date -Is) starting receipt-aware signature rank-$task_index batch queue keeper" >> "$log"
while :; do
  start=$(first_missing_batch)
  if [[ -z "$start" ]]; then
    echo "$(date -Is) all rank-$task_index row-group receipts passed" >> "$log"
    exit 0
  fi

  if active=$(active_job_for "$start"); [[ -n "$active" ]]; then
    successor=$((start + batch_size))
    if [[ "$successor" -le "$max_group" ]] && [[ -z $(active_job_for "$successor") ]]; then
      state=$(latest_terminal_state "$successor")
      if [[ -n "$state" ]]; then
        echo "$(date -Is) stopping: successor batch $successor has terminal state $state without passed receipts" >> "$log"
        exit 1
      fi
      job=$(submit_batch "$successor")
      echo "$(date -Is) queued successor batch $successor-$(batch_end "$successor") as job $job" >> "$log"
    fi
    sleep 30
    continue
  fi

  state=$(latest_terminal_state "$start")
  if [[ -n "$state" ]]; then
    echo "$(date -Is) stopping: batch $start has terminal state $state without passed receipts" >> "$log"
    exit 1
  fi
  job=$(submit_batch "$start")
  echo "$(date -Is) queued batch $start-$(batch_end "$start") as job $job" >> "$log"
  sleep 30
done

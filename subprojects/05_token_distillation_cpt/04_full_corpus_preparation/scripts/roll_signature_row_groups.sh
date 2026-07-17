#!/usr/bin/env bash
# Keep one additional, receipt-validated MinHash row-group job queued on CSCS.
# This intentionally stops on an unexplained terminal job; it never retries or
# advances past a missing receipt.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_DIR COORD_DIR CODE_DIR" >&2
  exit 2
fi

run=$1
coord=$2
code=$3
pipeline="$code/subprojects/05_token_distillation_cpt/04_full_corpus_preparation"
config="$pipeline/configs/agent1_v5_eiger_pipeline.json"
receipt_dir="$run/60-dedup/minhash-signatures/partial-receipts/00003"
log="$coord/signature-r3-roll.log"

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

active_job_for() {
  local group=$1
  squeue -h -u fffoivos -n "a1v5-signature-r3g${group}" -o '%A' | head -n 1
}

latest_terminal_state() {
  local group=$1
  sacct -X --noheader --parsable2 --user=fffoivos \
    --name="a1v5-signature-r3g${group}" --format=State \
    --starttime=2026-07-15 2>/dev/null | tail -n 1 | cut -d '|' -f 1
}

submit_group() {
  local group=$1
  local command
  command="uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME $coord/runtime/venv/bin/python $pipeline/scripts/agent1_v5_datatrove.py signature-row-group-task --config $config --contract $run/run_contract.json --combined-manifest $run/release-pre-dedup/manifests/combined_manifest.json --runtime-receipt $run/datatrove_runtime.json --task-index 3 --row-group $group"
  sbatch --parsable --uenv-passthrough=ignore \
    --job-name="a1v5-signature-r3g${group}" --partition=debug --time=01:25:00 \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --account=a0140 \
    --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" \
    --wrap "$command"
}

echo "$(date -Is) starting receipt-aware signature rank-3 queue keeper" >> "$log"
while :; do
  next=''
  for group in $(seq 0 14); do
    if ! is_passed "$group"; then
      next=$group
      break
    fi
  done

  if [[ -z "$next" ]]; then
    echo "$(date -Is) all rank-3 row-group receipts passed" >> "$log"
    exit 0
  fi

  if active=$(active_job_for "$next"); [[ -n "$active" ]]; then
    # The first missing group is running or pending.  Keep precisely one
    # successor queued as well, subject to the same receipt/failure rules.
    candidate=$((next + 1))
    if [[ "$candidate" -le 14 ]] && ! is_passed "$candidate"; then
      if [[ -z $(active_job_for "$candidate") ]]; then
        candidate_state=$(latest_terminal_state "$candidate")
        if [[ -n "$candidate_state" ]]; then
          echo "$(date -Is) stopping: successor group $candidate has terminal state $candidate_state without passed receipt" >> "$log"
          exit 1
        fi
        job=$(submit_group "$candidate")
        echo "$(date -Is) queued successor group $candidate as job $job" >> "$log"
      fi
    fi
    sleep 30
    continue
  fi

  state=$(latest_terminal_state "$next")
  if [[ -n "$state" && "$state" != "COMPLETED" ]]; then
    echo "$(date -Is) stopping: group $next terminal state $state without passed receipt" >> "$log"
    exit 1
  fi
  if [[ "$state" == "COMPLETED" ]]; then
    echo "$(date -Is) stopping: group $next completed without passed receipt" >> "$log"
    exit 1
  fi

  job=$(submit_group "$next")
  echo "$(date -Is) queued group $next as job $job" >> "$log"
  sleep 30
done

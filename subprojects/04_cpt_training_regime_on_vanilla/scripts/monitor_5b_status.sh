#!/usr/bin/env bash
# Lightweight home-side status monitor for the 04 Vanilla 5B run.
#
# Runs on home and polls Clariden over SSH. It does not submit or cancel jobs;
# it only records queue state and recent artifact/log evidence.

set -euo pipefail

RUN_TAG="${RUN_TAG:-04_vanilla_goldfish_5b_20260528T112539Z}"
DATASET_JOB_ID="${DATASET_JOB_ID:-2415688}"
SUBMITTER_JOB_ID="${SUBMITTER_JOB_ID:-2415828}"
HELDOUT_JOB_ID="${HELDOUT_JOB_ID:-}"
REMOTE_HOST="${REMOTE_HOST:-clariden}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
DATASET_RUN_TAG="${DATASET_RUN_TAG:-04_hplt_b1_dataset_5b_20260528T112539Z}"
LOG_PATH="${LOG_PATH:-/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/monitor_logs/${RUN_TAG}_status.log}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
SSH_TIMEOUT="${SSH_TIMEOUT:-30}"

mkdir -p "$(dirname "$LOG_PATH")"

remote_status_cmd() {
  local sacct_ids="$DATASET_JOB_ID,$SUBMITTER_JOB_ID"
  if [ -n "$HELDOUT_JOB_ID" ]; then
    sacct_ids="$sacct_ids,$HELDOUT_JOB_ID"
  fi
  cat <<REMOTE
set -euo pipefail
echo "squeue:"
squeue -u fffoivos -o "%.18i %.9P %.32j %.8T %.10M %.10l %.20R"
echo
echo "sacct:"
sacct -j "$sacct_ids" --format=JobID,JobName%32,Partition,State,ExitCode,Elapsed,AllocTRES%60 -P 2>/dev/null || true
echo
echo "dataset log tail:"
tail -n 8 "$RUN_ROOT/04_hplt_b1_build-${DATASET_JOB_ID}.out" 2>/dev/null || true
echo
echo "dataset artifacts:"
find "$RUN_ROOT/$DATASET_RUN_TAG" -maxdepth 3 \( -type f -o -type l \) 2>/dev/null | sort | tail -n 30 || true
echo
echo "code/math heldout artifacts:"
find /iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout -maxdepth 1 \( -name "cpt_code_heldout_200_20260528.jsonl" -o -name "cpt_math_heldout_200_20260528.jsonl" -o -name "cpt_code_math_heldout_20260528.manifest.json" \) -printf "%p %s bytes\n" 2>/dev/null | sort || true
echo
echo "training artifacts:"
find "$RUN_ROOT/$RUN_TAG" -maxdepth 4 \( -name run_metadata.json -o -name training_command.sh -o -name latest_checkpointed_iteration.txt -o -name ".metadata" \) 2>/dev/null | sort || true
echo
echo "submit state:"
find "$RUN_ROOT/${RUN_TAG}_submit_state" -maxdepth 2 -type f 2>/dev/null | sort || true
echo
echo "sidecar state:"
find "$RUN_ROOT/${RUN_TAG}_sidecar_watch" -maxdepth 2 -type f 2>/dev/null | sort || true
REMOTE
}

while true; do
  {
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    if ! ssh -o ConnectTimeout="$SSH_TIMEOUT" "$REMOTE_HOST" "$(remote_status_cmd)"; then
      echo "WARN: remote status poll failed"
    fi
    echo
  } >> "$LOG_PATH" 2>&1
  sleep "$SLEEP_SECONDS"
done

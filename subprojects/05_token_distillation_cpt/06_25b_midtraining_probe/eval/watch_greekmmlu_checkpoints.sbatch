#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_gmmlu_watch
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUN_TAG:?set RUN_TAG}"
: "${TRAIN_RUN_DIR:?set TRAIN_RUN_DIR}"
: "${TRAINING_ASSETS_RECEIPT:?set TRAINING_ASSETS_RECEIPT}"
: "${TRAIN_JOB_ID:?set TRAIN_JOB_ID}"
: "${START_ITERATION:?set START_ITERATION}"
: "${END_ITERATION:?set END_ITERATION}"

SUBPROJECT="$REPO_ROOT/subprojects/05_token_distillation_cpt/06_25b_midtraining_probe"
SUBMIT_SCRIPT="$SUBPROJECT/eval/submit_greekmmlu_checkpoint.sh"
STATE_DIR="${STATE_DIR:-$TRAIN_RUN_DIR/eval_watch_state}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
MAX_SECONDS="${MAX_SECONDS:-41400}"
mkdir -p "$STATE_DIR"
test -x "$SUBMIT_SCRIPT"

mapfile -t ITERATIONS < <(python3 - "$START_ITERATION" "$END_ITERATION" <<'PY'
import sys
start,end=map(int,sys.argv[1:])
boundaries={0:1785,1785:3570,3570:5960}
if boundaries.get(start) != end: raise SystemExit("invalid production segment")
values=[value for value in range(119,5951,119) if start < value <= end]
if end == 5960: values.append(5960)
print(*values,sep="\n")
PY
)
(( ${#ITERATIONS[@]} > 0 ))

start_epoch=$(date +%s)
while true; do
  pending=0
  for iteration in "${ITERATIONS[@]}"; do
    state="$STATE_DIR/iter_${iteration}.submitted"
    [[ -e "$state" ]] && continue
    pending=$((pending + 1))
    padded=$(printf '%07d' "$iteration")
    checkpoint="$TRAIN_RUN_DIR/checkpoints/iter_$padded"
    marker="$TRAIN_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
    latest=""
    [[ -s "$marker" ]] && latest=$(tr -d '[:space:]' < "$marker")
    if [[ -s "$checkpoint/.metadata" && "$latest" =~ ^[0-9]+$ ]] \
      && (( latest >= iteration )); then
      TRAINING_ASSETS_RECEIPT="$TRAINING_ASSETS_RECEIPT" \
      TRAIN_RUN_DIR="$TRAIN_RUN_DIR" RUN_TAG="$RUN_TAG" ITERATION="$iteration" \
        "$SUBMIT_SCRIPT" >"$STATE_DIR/iter_${iteration}.log" 2>&1
      mv "$STATE_DIR/iter_${iteration}.log" "$state"
      pending=$((pending - 1))
    fi
  done
  (( pending > 0 )) || exit 0
  train_state=$(sacct -j "$TRAIN_JOB_ID" -X -n -o State | head -n 1 | tr -d '[:space:]' | cut -d+ -f1)
  case "$train_state" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE)
      echo "training job $TRAIN_JOB_ID ended as $train_state with $pending evaluations missing" >&2
      exit 5
      ;;
  esac
  elapsed=$(( $(date +%s) - start_epoch ))
  if (( elapsed >= MAX_SECONDS )); then
    echo "watch window expired with $pending checkpoint evaluations pending" >&2
    exit 4
  fi
  sleep "$SLEEP_SECONDS"
done

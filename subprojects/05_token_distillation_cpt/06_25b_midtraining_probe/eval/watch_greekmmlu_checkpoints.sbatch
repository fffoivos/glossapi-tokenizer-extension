#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_gmmlu_watch
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

set -euo pipefail
: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUN_TAG:?set RUN_TAG}"
: "${TRAIN_RUN_DIR:?set TRAIN_RUN_DIR}"
: "${HF_TOKENIZER_DIR:?set HF_TOKENIZER_DIR}"

SUBPROJECT="$REPO_ROOT/subprojects/05_token_distillation_cpt"
SUBMIT_SCRIPT="$SUBPROJECT/scripts/submit_td_checkpoint_sidecars.sh"
STATE_DIR="${STATE_DIR:-$TRAIN_RUN_DIR/eval_watch_state}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_SECONDS="${MAX_SECONDS:-82800}"
mkdir -p "$STATE_DIR"
test -x "$SUBMIT_SCRIPT"

start_epoch=$(date +%s)
while true; do
  pending=0
  for iteration in $(seq 119 119 5950) 5960; do
    state="$STATE_DIR/iter_${iteration}.submitted"
    [[ -e "$state" ]] && continue
    pending=$((pending + 1))
    padded=$(printf '%07d' "$iteration")
    checkpoint="$TRAIN_RUN_DIR/checkpoints/iter_$padded"
    if [[ -d "$checkpoint" && -s "$checkpoint/.metadata" ]]; then
      tokens=$((iteration * 4194304))
      SUBMIT_NATIVE=1 SUBMIT_GREEK_NLP=0 SUBMIT_BPB=0 SUBMIT_RETENTION=0 SUBMIT_CHECKSUM=0 \
      NATIVE_BENCHMARKS=greekmmlu HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR" \
      "$SUBMIT_SCRIPT" \
        --train-run-dir "$TRAIN_RUN_DIR" --run-tag "$RUN_TAG" --arm td \
        --iteration "$iteration" --tokens "$tokens" \
        --checkpoint-label "Greek-CPT25B-L11-${tokens}tok" \
        >"$STATE_DIR/iter_${iteration}.log" 2>&1
      touch "$state"
      pending=$((pending - 1))
    fi
  done
  (( pending > 0 )) || exit 0
  elapsed=$(( $(date +%s) - start_epoch ))
  if (( elapsed >= MAX_SECONDS )); then
    echo "watch window expired with $pending checkpoint evaluations pending" >&2
    exit 4
  fi
  sleep "$SLEEP_SECONDS"
done

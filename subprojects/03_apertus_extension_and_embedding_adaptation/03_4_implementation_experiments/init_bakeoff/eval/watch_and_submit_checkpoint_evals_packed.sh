#!/usr/bin/env bash
# Watch until all requested arms have a complete checkpoint, then submit one
# packed conversion/eval chain. This avoids one whole-node eval allocation per
# arm on Clariden.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. bakeoff_1node_chain_20260522_005620}"
ITER="${ITER:-390}"
TASK_GROUP="${TASK_GROUP:-full}"
ARMS="${ARMS:-vanilla retok centroid}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"

if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ITER must be an integer, got: $ITER" >&2
    exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${RUN_TAG}_watch_iter_${ITER_PAD}_${TASK_GROUP}_packed}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/watch.log"

log() {
    printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

all_ready() {
    local arm ckpt_root ckpt_dir tracker latest
    for arm in $ARMS; do
        ckpt_root="$RUN_ROOT/${RUN_TAG}_${arm}/checkpoints"
        ckpt_dir="$ckpt_root/iter_$ITER_PAD"
        tracker="$ckpt_root/latest_checkpointed_iteration.txt"
        if [ ! -d "$ckpt_dir" ]; then
            log "$arm: waiting for $ckpt_dir"
            return 1
        fi
        if [ ! -f "$tracker" ]; then
            log "$arm: checkpoint dir exists but tracker is missing: $tracker"
            return 1
        fi
        latest="$(tr -dc '0-9' < "$tracker" || true)"
        if [ "$latest" != "$ITER" ]; then
            log "$arm: checkpoint dir exists but tracker says '${latest:-empty}', waiting for $ITER"
            return 1
        fi
    done
    return 0
}

start_epoch="$(date +%s)"
log "packed watch start RUN_TAG=$RUN_TAG ITER=$ITER TASK_GROUP=$TASK_GROUP ARMS=[$ARMS]"
log "RUN_ROOT=$RUN_ROOT OUT_ROOT=$OUT_ROOT STATE_DIR=$STATE_DIR"

while true; do
    if [ -f "$STATE_DIR/packed.submitted" ]; then
        log "packed eval already submitted; exiting"
        exit 0
    fi

    if all_ready; then
        log "all arms ready; submitting packed $TASK_GROUP eval for iter $ITER"
        submit_log="$STATE_DIR/packed.submit.log"
        (
            cd "$SCRIPT_DIR"
            RUN_TAG="$RUN_TAG" \
            RUN_ROOT="$RUN_ROOT" \
            OUT_ROOT="$OUT_ROOT" \
            STATE_DIR="$STATE_DIR/submit_state" \
            bash "$SCRIPT_DIR/submit_bakeoff_checkpoint_eval_packed.sh" "$ITER" "$TASK_GROUP" $ARMS
        ) 2>&1 | tee "$submit_log"
        touch "$STATE_DIR/packed.submitted"
        log "packed eval submitted; details in $submit_log"
        exit 0
    fi

    now="$(date +%s)"
    elapsed=$((now - start_epoch))
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        log "ERROR: timeout after ${elapsed}s before all arms ready"
        exit 4
    fi

    sleep "$POLL_SECONDS"
done

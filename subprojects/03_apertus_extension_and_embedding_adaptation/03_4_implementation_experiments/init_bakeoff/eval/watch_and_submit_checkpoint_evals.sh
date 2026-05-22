#!/usr/bin/env bash
# Watch for bakeoff Megatron checkpoints and submit conversion + eval once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. bakeoff_1node_chain_20260522_005620}"
ITER="${ITER:-65}"
TASK_GROUP="${TASK_GROUP:-greek_only}"
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
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${RUN_TAG}_watch_iter_${ITER_PAD}_${TASK_GROUP}}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/watch.log"

log() {
    printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

all_done() {
    local arm
    for arm in $ARMS; do
        [ -f "$STATE_DIR/${arm}.submitted" ] || return 1
    done
    return 0
}

start_epoch="$(date +%s)"
log "watch start RUN_TAG=$RUN_TAG ITER=$ITER TASK_GROUP=$TASK_GROUP ARMS=[$ARMS]"
log "RUN_ROOT=$RUN_ROOT OUT_ROOT=$OUT_ROOT STATE_DIR=$STATE_DIR"

while true; do
    for arm in $ARMS; do
        stamp="$STATE_DIR/${arm}.submitted"
        [ -f "$stamp" ] && continue

        ckpt_root="$RUN_ROOT/${RUN_TAG}_${arm}/checkpoints"
        ckpt_dir="$ckpt_root/iter_$ITER_PAD"
        tracker="$ckpt_root/latest_checkpointed_iteration.txt"
        if [ ! -d "$ckpt_dir" ]; then
            log "$arm: waiting for $ckpt_dir"
            continue
        fi
        if [ ! -f "$tracker" ]; then
            log "$arm: checkpoint dir exists but tracker is missing: $tracker"
            continue
        fi
        latest="$(tr -dc '0-9' < "$tracker" || true)"
        if [ "$latest" != "$ITER" ]; then
            log "$arm: checkpoint dir exists but tracker says '${latest:-empty}', waiting for $ITER"
            continue
        fi

        log "$arm: checkpoint exists and tracker=$latest; submitting $TASK_GROUP eval for iter $ITER"
        submit_log="$STATE_DIR/${arm}.submit.log"
        (
            cd "$SCRIPT_DIR"
            RUN_TAG="$RUN_TAG" \
            RUN_ROOT="$RUN_ROOT" \
            OUT_ROOT="$OUT_ROOT" \
            bash "$SCRIPT_DIR/submit_bakeoff_checkpoint_eval.sh" "$arm" "$ITER" "$TASK_GROUP"
        ) 2>&1 | tee "$submit_log"
        touch "$stamp"
        log "$arm: submitted; details in $submit_log"
    done

    if all_done; then
        log "all requested arms submitted"
        exit 0
    fi

    now="$(date +%s)"
    elapsed=$((now - start_epoch))
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        log "ERROR: timeout after ${elapsed}s before all arms submitted"
        exit 4
    fi

    sleep "$POLL_SECONDS"
done

#!/usr/bin/env bash
# Watch one TD layer11 checkpoint and submit conversion/eval exactly once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TAG="${RUN_TAG:-td_full25_layer11_2b_20260523T165038Z}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$RUN_ROOT/$RUN_TAG/checkpoints}"
ITER="${ITER:?ITER is required}"
TASK_GROUP="${TASK_GROUP:-full}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
EVAL_TAG="${EVAL_TAG:-$RUN_TAG}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-200000}"

if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ITER must be an integer, got: $ITER" >&2
    exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
STATE_DIR="${STATE_DIR:-$OUT_ROOT/${EVAL_TAG}_watch_iter_${ITER_PAD}_${TASK_GROUP}}"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/watch.log"

log() {
    printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

start_epoch="$(date +%s)"
stamp="$STATE_DIR/submitted"
ckpt_dir="$MEGATRON_CKPT_ROOT/iter_$ITER_PAD"
tracker="$MEGATRON_CKPT_ROOT/latest_checkpointed_iteration.txt"

log "watch start RUN_TAG=$RUN_TAG ITER=$ITER TASK_GROUP=$TASK_GROUP"
log "MEGATRON_CKPT_ROOT=$MEGATRON_CKPT_ROOT OUT_ROOT=$OUT_ROOT STATE_DIR=$STATE_DIR"

while true; do
    if [ -f "$stamp" ]; then
        log "already submitted; exiting"
        exit 0
    fi

    if [ ! -d "$ckpt_dir" ]; then
        log "waiting for $ckpt_dir"
    elif [ ! -f "$tracker" ]; then
        log "checkpoint dir exists but tracker is missing: $tracker"
    else
        latest="$(tr -dc '0-9' < "$tracker" || true)"
        if [ -z "$latest" ]; then
            log "checkpoint dir exists but tracker is empty"
        elif [ "$latest" -lt "$ITER" ]; then
            log "checkpoint dir exists but tracker says $latest, waiting for >= $ITER"
        else
            log "checkpoint ready: iter_$ITER_PAD tracker=$latest; submitting $TASK_GROUP"
            submit_log="$STATE_DIR/submit.log"
            (
                cd "$SCRIPT_DIR"
                RUN_TAG="$RUN_TAG" \
                RUN_ROOT="$RUN_ROOT" \
                MEGATRON_CKPT_ROOT="$MEGATRON_CKPT_ROOT" \
                OUT_ROOT="$OUT_ROOT" \
                EVAL_TAG="$EVAL_TAG" \
                SUBMIT_INTRINSIC="${SUBMIT_INTRINSIC:-0}" \
                EVAL_JSONL="${EVAL_JSONL:-}" \
                bash "$SCRIPT_DIR/submit_td_checkpoint_eval.sh" "$ITER" "$TASK_GROUP"
            ) 2>&1 | tee "$submit_log"
            touch "$stamp"
            log "submitted; details in $submit_log"
            exit 0
        fi
    fi

    now="$(date +%s)"
    elapsed=$((now - start_epoch))
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        log "ERROR: timeout after ${elapsed}s"
        exit 4
    fi

    sleep "$POLL_SECONDS"
done

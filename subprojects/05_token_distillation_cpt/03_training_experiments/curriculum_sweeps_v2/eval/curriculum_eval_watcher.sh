#!/usr/bin/env bash
# GreekMMLU-ONLY eval watcher for a curriculum run. Wraps the deployed TD sidecar watcher, threading
# NATIVE_BENCHMARKS=greekmmlu + SUBMIT_GREEK_NLP=0 + SUBMIT_RETENTION=0 + SUBMIT_BPB=0 so ONLY GreekMMLU
# runs (forgetting is read from the in-training extra-valid loss, not benchmarks). Requires UPSTREAM
# EDIT 3 so these survive the watcher's ~23h self-resubmit.
#   Usage:  RUN_TAG=<the run tag> EVAL_ARM=td bash curriculum_eval_watcher.sh
set -euo pipefail
source "$(dirname "$0")/../paths.env"
RUN_TAG="${RUN_TAG:?set RUN_TAG to the curriculum run tag}"
EVAL_ARM="${EVAL_ARM:-td}"
TRAIN_RUN_DIR="$RUN_ROOT/$RUN_TAG"
HF_TOK="$EXT_HF_TOK"; [ "$EVAL_ARM" = vanilla ] && HF_TOK="$BASE_TOK"
WATCHER="$SUB/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch"
test -f "$WATCHER"; test -f "$V2/eval/cadence_curriculum.tsv"

sbatch --parsable --account=a0140 --partition=xfer --cpus-per-task=1 --mem=4G --time=24:00:00 \
  --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err" \
  --export="ALL,RUN_TAG=$RUN_TAG,RUN_ROOT=$RUN_ROOT,TRAIN_RUN_DIR=$TRAIN_RUN_DIR,CHECKPOINTS_FILE=$V2/eval/cadence_curriculum.tsv,HF_TOKENIZER_DIR=$HF_TOK,EVAL_ARM=$EVAL_ARM,NATIVE_BENCHMARKS=greekmmlu,SUBMIT_GREEK_NLP=0,SUBMIT_RETENTION=0,SUBMIT_BPB=0,REQUIRE_CODE_MATH_HELDOUTS=0,WATCHER_PARTITION=xfer,CPU_ONLY_PARTITION=xfer" \
  "$WATCHER"
echo "GreekMMLU-only eval watcher submitted for $RUN_TAG (EVAL_ARM=$EVAL_ARM)"

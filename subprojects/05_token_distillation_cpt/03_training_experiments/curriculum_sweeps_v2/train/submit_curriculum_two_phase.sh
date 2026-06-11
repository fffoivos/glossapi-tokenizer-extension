#!/usr/bin/env bash
# THE two-phase curriculum driver. One continuous afterok chain:
#   phase-1 segments  (TRAIN_CONFIG_OVERRIDE=phase1_hplt.env)     iters 0 .. PHASE1_EXIT_ITER
#   phase-2 segments  (TRAIN_CONFIG_OVERRIDE=phase2_glossapi.env) iters PHASE1_EXIT_ITER .. TOTAL_ITER
# Every segment passes the SAME full-run TRAIN_TOKENS (the WSD anchor). AdEMAMix carries via
# RESUME_TRAINING=1 (no --no-load-optim). The FIRST phase-2 segment exports RESET_DATA_INDEX=1 so the
# train dataloader restarts at the GlossAPI binary's index 0 while the scheduler/optimizer continue.
# --exit-interval is the ABSOLUTE exit iteration of each segment (Megatron checks iteration % == 0 and
# the while iteration<train_iters bound, so the final segment stops at TOTAL_ITER).
#
# Env knobs (set by the sweep launchers): ARM (td|vanilla), R (replay blend weight), LR_PEAK,
#   PHASE1_EXIT_ITER, RUN_TAG. Smoke-friendly overrides: TOTAL_ITER, SEG, SAVE_INTERVAL,
#   NODES, GPUS_PER_NODE, TIME_LIMIT. Live submission requires DRY_RUN=0 CONFIRM_LAUNCH=1.
# Requires the two bakeoff_train.sbatch edits in UPSTREAM_EDITS.md.
set -euo pipefail
source "$(dirname "$0")/../paths.env"

ARM="${ARM:-td}"
R="${R:-0.35}"
LR_PEAK="${LR_PEAK:-5.5e-5}"
RUN_TAG="${RUN_TAG:-curr_${ARM}_r${R}_lr${LR_PEAK}}"
OUTPUT_DIR="$RUN_ROOT/$RUN_TAG"
DRY_RUN="${DRY_RUN:-0}"
CONFIRM_LAUNCH="${CONFIRM_LAUNCH:-0}"

TRAIN_TOKENS="${TRAIN_TOKENS:-13500000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-119}"
TOTAL_ITER="${TOTAL_ITER:-3218}"       # 13.5B / 4.194M
PHASE1_EXIT_ITER="${PHASE1_EXIT_ITER:-2261}"   # pin from realized token counts after Stage B
NODES="${NODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
SEG="${SEG:-952}"                      # 8*119, iters per walltime segment
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN=$DRY_RUN not recognized (expected 0|1)" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = "0" ] && [ "$CONFIRM_LAUNCH" != "1" ]; then
  echo "ERROR: live submission requires CONFIRM_LAUNCH=1 (or set DRY_RUN=1)" >&2
  exit 2
fi
(( PHASE1_EXIT_ITER % SAVE_INTERVAL == 0 )) || { echo "ERROR: PHASE1_EXIT_ITER=$PHASE1_EXIT_ITER must be a multiple of $SAVE_INTERVAL" >&2; exit 2; }
(( PHASE1_EXIT_ITER > 0 && PHASE1_EXIT_ITER < TOTAL_ITER )) || { echo "ERROR: PHASE1_EXIT_ITER out of range" >&2; exit 2; }
[ "$DRY_RUN" = "0" ] && mkdir -p "$OUTPUT_DIR"

init="$INIT_TD"; [ "$ARM" = vanilla ] && init="$INIT_VANILLA"
dep=""; segn=0; resume=0

submit_block(){ # $1=config  $2=from_iter  $3=to_iter  $4=reset_first(0|1)
  local cfg="$1" from="$2" to="$3" reset="$4" cur="$2"
  while (( cur < to )); do
    local nxt=$(( cur + SEG )); (( nxt > to )) && nxt=$to
    (( to - nxt > 0 && to - nxt < SAVE_INTERVAL )) && nxt=$to
    segn=$((segn+1))
    local extra=""
    [ "$reset" = 1 ] && (( cur == from )) && extra=",RESET_DATA_INDEX=1"
    local dep_args=(); [ -n "$dep" ] && dep_args=(--dependency=afterok:$dep)
    local cmd=(sbatch --parsable "${dep_args[@]}"
      --job-name="${RUN_TAG}_s${segn}" --account=a0140 --partition=normal
      --nodes=$NODES --ntasks-per-node=1 --gpus-per-node=$GPUS_PER_NODE --gres=gpu:$GPUS_PER_NODE
      --cpus-per-task=288 --mem=460G --time="$TIME_LIMIT"
      --output="$OUTPUT_DIR/%x-%j.out" --error="$OUTPUT_DIR/%x-%j.err"
      --export="ALL,ARM=$ARM,INIT_CKPT=$init,OUTPUT_DIR=$OUTPUT_DIR,SCRIPT_DIR_OVERRIDE=$TRAIN_DIR,TRAIN_CONFIG_OVERRIDE=$cfg,TRAIN_TOKENS=$TRAIN_TOKENS,RESUME_TRAINING=$resume,DISABLE_SAVE=0,SAVE_INTERVAL=$SAVE_INTERVAL,EXIT_INTERVAL=$nxt,ACCOUNT=a0140,PARTITION=normal,NODES=$NODES,GPUS_PER_NODE=$GPUS_PER_NODE,LAUNCH_MODE=torchrun,TIME_LIMIT=$TIME_LIMIT,R=$R,LR_PEAK=$LR_PEAK${extra}"
      "$TRAIN_SCRIPT")
    echo "+ seg $segn  cfg=$(basename "$cfg")  iters $cur..$nxt  resume=$resume  reset=${extra:-0}"
    if [ "$DRY_RUN" = 1 ]; then
      printf '  DRY:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      dep="DRY${segn}"
    else
      dep="$("${cmd[@]}")"
    fi
    resume=1; init="$OUTPUT_DIR/checkpoints"; cur=$nxt
  done
}

echo "=== curriculum chain: RUN_TAG=$RUN_TAG ARM=$ARM R=$R LR_PEAK=$LR_PEAK TOTAL_ITER=$TOTAL_ITER PHASE1_EXIT_ITER=$PHASE1_EXIT_ITER ==="
submit_block "$V2/train/phase1_hplt.env"      0                  "$PHASE1_EXIT_ITER" 0
submit_block "$V2/train/phase2_glossapi.env"  "$PHASE1_EXIT_ITER" "$TOTAL_ITER"       1
echo "final segment job: $dep   (output: $OUTPUT_DIR)"

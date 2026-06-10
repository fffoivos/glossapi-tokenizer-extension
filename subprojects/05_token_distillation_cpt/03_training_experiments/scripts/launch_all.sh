#!/usr/bin/env bash
# ============================================================================
# Launch BOTH arms of the 13.5B CPT experiment IN PARALLEL:
#   - arm1 vanilla (base 131072)   - arm2 td (modern-greek 148480 + TD init)
# Each arm = (a) its own full-run WSD training chain  +  (b) an eval-sidecar
# watcher that auto-benchmarks every checkpoint at the chosen cadence. The two
# arms are independent Slurm job chains, so they run concurrently.
#
#   bash launch_all.sh                       # dry-run (prints everything)
#   DRY_RUN=0 CONFIRM_LAUNCH=1 bash launch_all.sh
#
# Reuses the ESTABLISHED machinery — training: submit_two_arm_full_run.sh ->
# bakeoff_train.sbatch; benchmarks: the 04/05 checkpoint-sidecar watchers (their
# checkpoint list is now CHECKPOINTS_FILE-overridable) -> submit_*_sidecars.sh
# -> lm-eval (greekmmlu=dascim, ilsp×2, plutus), Greek-NLP, heldout-Greek/code/
# math BPB, multilingual retention. Loss/grad/LR/throughput -> tensorboard +
# .out every iteration (parse with collect_metrics.py).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_LAUNCH="${CONFIRM_LAUNCH:-0}"

SUBMIT_WATCHERS="${SUBMIT_WATCHERS:-1}"
RUN_ARTIFACT_GATE="${RUN_ARTIFACT_GATE:-1}"
# Watchers are lightweight 24h CPU pollers. xfer is the cheap home for them but
# is in maintenance till 2026-06-11; `normal` works but bills a whole GH200 node
# for a poller. Prefer xfer once it returns, else run the poll loop on a login
# node. See ../../RUNBOOK.md.
WATCHER_PARTITION="${WATCHER_PARTITION:-xfer}"
BENCH_EVERY_ITERS="${BENCH_EVERY_ITERS:-238}"   # ~1B tokens between benchmarks

BASE_TOK="${BASE_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
EXT_TOK="${EXT_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
VANILLA_WATCHER="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch"
TD_WATCHER="$REPO_ROOT/subprojects/05_token_distillation_cpt/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch"

# total iters + global-batch tokens from the single source of truth
# shellcheck disable=SC1091
source "$ROOT/configs/common_cpt.env"
TOTAL_ITERS="$TRAIN_ITERS"; GBT="$GLOBAL_BATCH_TOKENS"

gen_checkpoints_file() {   # $1=label $2=outfile  -> "iter tokens label" per line
  local label="$1" out="$2" it tok tokB
  : > "$out"
  for (( it=BENCH_EVERY_ITERS; it<TOTAL_ITERS; it+=BENCH_EVERY_ITERS )); do
    tok=$(( it * GBT )); tokB=$(awk "BEGIN{printf \"%.1f\", $tok/1e9}")
    printf "%d %d %s-%sB\n" "$it" "$tok" "$label" "$tokB" >> "$out"
  done
  tok=$(( TOTAL_ITERS * GBT ))
  printf "%d %d %s-final\n" "$TOTAL_ITERS" "$tok" "$label" >> "$out"
}

launch_arm() {   # $1=arm $2=tokenizer $3=watcher $4=eval_arm $5=label
  local arm="$1" tok_dir="$2" watcher="$3" eval_arm="$4" label="$5"
  local run_tag="cpt13b_${arm}_${STAMP}"
  local output_dir="$RUN_ROOT/$run_tag"
  local state_dir="$RUN_ROOT/${run_tag}_orch"
  mkdir -p "$state_dir"
  local ckpts="$state_dir/benchmark_checkpoints.tsv"
  gen_checkpoints_file "$label" "$ckpts"
  echo "=== arm=$arm  run_tag=$run_tag ==="
  echo "  benchmark cadence: $(wc -l < "$ckpts") checkpoints (every ${BENCH_EVERY_ITERS} iters + final) -> $ckpts"

  # (a) training chain — independent per arm => the two arms run in parallel
  RUN_TAG="$run_tag" RUN_ROOT="$RUN_ROOT" DRY_RUN="$DRY_RUN" CONFIRM_LAUNCH="$CONFIRM_LAUNCH" \
    bash "$ROOT/scripts/submit_two_arm_full_run.sh" "$arm"

  # (b) eval-sidecar watcher — auto-benchmarks each checkpoint
  if [ "$SUBMIT_WATCHERS" = "1" ]; then
    local cmd=( sbatch --parsable --partition="$WATCHER_PARTITION"
      --output="$RUN_ROOT/%x-%j.out" --error="$RUN_ROOT/%x-%j.err"
      --export=ALL,RUN_TAG="$run_tag",RUN_ROOT="$RUN_ROOT",TRAIN_RUN_DIR="$output_dir",CHECKPOINTS_FILE="$ckpts",HF_TOKENIZER_DIR="$tok_dir",EVAL_ARM="$eval_arm"
      "$watcher" )
    if [ "$DRY_RUN" = "1" ]; then printf '  [dry-run] watcher:\n    %s\n' "${cmd[*]}"
    else echo "  watcher job: $("${cmd[@]}")"; fi
  fi
  echo
}

echo "=== launch_all.sh  (STAMP=$STAMP  DRY_RUN=$DRY_RUN) ==="
echo "total iters=$TOTAL_ITERS  global-batch tokens=$GBT  watcher partition=$WATCHER_PARTITION"
echo
if [ "$DRY_RUN" = "0" ] && [ "$RUN_ARTIFACT_GATE" = "1" ]; then
  echo "=== prelaunch artifact gate ==="
  bash "$ROOT/scripts/gate_cpt2arm_artifacts.sh"
  echo
fi
launch_arm vanilla "$BASE_TOK" "$VANILLA_WATCHER" vanilla Vanilla
launch_arm td      "$EXT_TOK"  "$TD_WATCHER"      td      ModernGreekTD

echo "Monitor loss/throughput across BOTH arms:"
echo "  python3 $ROOT/scripts/collect_metrics.py \\"
echo "    --arm vanilla:'$RUN_ROOT/cpt13b_vanilla_${STAMP}/*.out' \\"
echo "    --arm td:'$RUN_ROOT/cpt13b_td_${STAMP}/*.out' --out $RUN_ROOT/metrics_${STAMP}.csv"
if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "DRY RUN — set DRY_RUN=0 CONFIRM_LAUNCH=1 to submit both arms."
fi

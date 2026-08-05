#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen stage root}"
: "${FULL8_INITIAL_MEGATRON:?set canonical initialization}"
: "${FULL8_SMOKE_RUN_ROOT:?set new smoke root}"
DRY_RUN=${DRY_RUN:-1}
if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$FULL8_SMOKE_RUN_ROOT" ]] || { echo "smoke root exists" >&2; exit 2; }
  mkdir -p "$FULL8_SMOKE_RUN_ROOT/logs"
fi
submit() { if [[ "$DRY_RUN" == 1 ]]; then { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY_JOB; else sbatch --parsable "$@"; fi; }
common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_SMOKE_RUN_ROOT=$FULL8_SMOKE_RUN_ROOT"
train=$(submit --nodes=16 --time=00:30:00 \
  --output="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_RUN_ROOT=$FULL8_SMOKE_RUN_ROOT,FULL8_START_ITERATION=0,FULL8_END_ITERATION=288,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_BENCHMARK_MODE=1" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
signaler=$(submit --dependency="after:$train" \
  --output="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_SMOKE_TRAIN_JOB=$train" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/signal_graceful_stop_smoke.sbatch")
resume=$(submit --dependency="afterany:$train" \
  --output="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_SMOKE_RUN_ROOT/logs/%x-%j.err" \
  --export="$common" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resume_graceful_stop_smoke.sbatch")
printf '{"train":"%s","signaler":"%s","resume_coordinator":"%s"}\n' "$train" "$signaler" "$resume"

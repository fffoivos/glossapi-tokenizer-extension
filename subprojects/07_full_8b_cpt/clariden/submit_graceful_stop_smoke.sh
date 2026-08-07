#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen stage root}"
: "${FULL8_INITIAL_MEGATRON:?set canonical initialization}"
: "${FULL8_SMOKE_RUN_ROOT:?set new smoke root}"
: "${FULL8_SELECTED_PROFILE:?set selected execution profile receipt}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
readarray -t profile < <(python3 - "$FULL8_SELECTED_PROFILE" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x["selection"]["profile_id"]); print(x["selection"]["nodes"])
PY
)
profile_id=${profile[0]}
nodes=${profile[1]}
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" "$nodes")
train_placement=(--switches=1 --exclude="$train_exclude")
DRY_RUN=${DRY_RUN:-1}
if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$FULL8_SMOKE_RUN_ROOT" ]] || { echo "smoke root exists" >&2; exit 2; }
  mkdir -p "$FULL8_SMOKE_RUN_ROOT/logs"
fi
submit() { if [[ "$DRY_RUN" == 1 ]]; then { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY_JOB; else sbatch --parsable "$@"; fi; }
common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_SMOKE_RUN_ROOT=$FULL8_SMOKE_RUN_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$profile_id,FULL8_SMOKE_NODES=$nodes,FULL8_TRAIN_LEAF_SWITCH=$FULL8_TRAIN_LEAF_SWITCH"
train=$(submit "${train_placement[@]}" --nodes="$nodes" --time=00:30:00 \
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

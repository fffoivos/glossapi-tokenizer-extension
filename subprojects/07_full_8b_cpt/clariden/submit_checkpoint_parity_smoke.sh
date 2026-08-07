#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen full-8B data root}"
: "${FULL8_PARITY_ROOT:?set new checkpoint-parity root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_DP32_LEAF_SWITCH:?set pinned Clariden leaf switch for DP32}"
FULL8_RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
FULL8_PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
dp32_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_DP32_LEAF_SWITCH" 16)
dp32_placement=(--switches=1 --exclude="$dp32_exclude")

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${CONFIRM_CHECKPOINT_PARITY_SMOKE:-}" == "APERTUS8B_SYNC_DP32_RESTART" ]] || {
    echo "set CONFIRM_CHECKPOINT_PARITY_SMOKE=APERTUS8B_SYNC_DP32_RESTART" >&2; exit 2;
  }
  [[ ! -e "$FULL8_PARITY_ROOT" ]] || { echo "parity root exists" >&2; exit 2; }
  mkdir -p "$FULL8_PARITY_ROOT/logs" "$FULL8_PARITY_ROOT/control_dp32" "$FULL8_PARITY_ROOT/control_dp32_repeat"
fi

train="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"
common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_RECIPE=$FULL8_RECIPE,FULL8_PROFILES=$FULL8_PROFILES,FULL8_BENCHMARK_MODE=1"
control=$(submit "${dp32_placement[@]}" --nodes=16 --time=01:00:00 --job-name=full8b-sync-parity-control \
  --output="$FULL8_PARITY_ROOT/logs/%x-%j.out" --error="$FULL8_PARITY_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=160,FULL8_RUN_ROOT=$FULL8_PARITY_ROOT/control_dp32,FULL8_EXECUTION_PROFILE=dp32_16node,FULL8_START_ITERATION=0,FULL8_END_ITERATION=162,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON" \
  "$train")
restart=$(submit "${dp32_placement[@]}" --nodes=16 --time=00:20:00 --job-name=full8b-sync-parity-r1 --dependency="afterok:$control" \
  --output="$FULL8_PARITY_ROOT/logs/%x-%j.out" --error="$FULL8_PARITY_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=,FULL8_EXACT_LOAD_ITERATION=160,FULL8_RUN_ROOT=$FULL8_PARITY_ROOT/control_dp32,FULL8_EXECUTION_PROFILE=dp32_16node,FULL8_START_ITERATION=160,FULL8_END_ITERATION=161,FULL8_LOAD_CHECKPOINT=$FULL8_PARITY_ROOT/control_dp32/checkpoints" \
  "$train")
repeat=$(submit "${dp32_placement[@]}" --nodes=16 --time=00:20:00 --job-name=full8b-sync-parity-r2 --dependency="afterok:$control" \
  --output="$FULL8_PARITY_ROOT/logs/%x-%j.out" --error="$FULL8_PARITY_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_BENCHMARK_SAVE_ITERATIONS=,FULL8_EXACT_LOAD_ITERATION=160,FULL8_RUN_ROOT=$FULL8_PARITY_ROOT/control_dp32_repeat,FULL8_EXECUTION_PROFILE=dp32_16node,FULL8_START_ITERATION=160,FULL8_END_ITERATION=161,FULL8_LOAD_CHECKPOINT=$FULL8_PARITY_ROOT/control_dp32/checkpoints" \
  "$train")
gate=$(submit --dependency="afterok:$restart:$repeat" \
  --output="$FULL8_PARITY_ROOT/logs/%x-%j.out" --error="$FULL8_PARITY_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_PARITY_ROOT=$FULL8_PARITY_ROOT,FULL8_PROFILES=$FULL8_PROFILES" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_checkpoint_parity_smoke.sbatch")

graph=$(python3 - "$control" "$restart" "$repeat" "$gate" <<'PY'
import json,sys
print(json.dumps(dict(zip(("control","restart","restart_repeat","gate"),sys.argv[1:])),separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$FULL8_PARITY_ROOT/submission.json" "$graph" <<'PY'
import datetime,json,os,sys,tempfile
out,raw=sys.argv[1:]
value={"schema_version":"apertus_full_8b_checkpoint_parity_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":json.loads(raw)}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,out)
PY
fi
printf '%s\n' "$graph"

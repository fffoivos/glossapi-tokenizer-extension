#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_STAGE_ROOT:?set new immutable data stage root}"
: "${FULL8_VALIDATION_PYTHON:?set Python with numpy, pyarrow and tokenizers}"
: "${FULL8_SOURCE_ROOT:?set sanitized source root}"
: "${FULL8_SANITIZED_BRIDGE_RECEIPT:?set sanitized bridge receipt}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    echo DRY_RUN_ID
  else sbatch --parsable "$@"
  fi
}

export_arg="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_VALIDATION_PYTHON=$FULL8_VALIDATION_PYTHON,FULL8_SOURCE_ROOT=$FULL8_SOURCE_ROOT,FULL8_SANITIZED_BRIDGE_RECEIPT=$FULL8_SANITIZED_BRIDGE_RECEIPT"
if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$FULL8_STAGE_ROOT" ]] || { echo "data stage root exists" >&2; exit 2; }
  mkdir -p "$FULL8_STAGE_ROOT/logs"
fi
freeze=$(submit --output="$FULL8_STAGE_ROOT/logs/%x-%j.out" --error="$FULL8_STAGE_ROOT/logs/%x-%j.err" --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/freeze_data_inventory.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then freeze=DRY_RUN_FREEZE; fi
pack=$(submit --dependency="afterok:$freeze" --output="$FULL8_STAGE_ROOT/logs/%x-%A_%a.out" --error="$FULL8_STAGE_ROOT/logs/%x-%A_%a.err" --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/pack_full_data.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then pack=DRY_RUN_PACK; fi
finalize=$(submit --dependency="afterok:$pack" --output="$FULL8_STAGE_ROOT/logs/%x-%j.out" --error="$FULL8_STAGE_ROOT/logs/%x-%j.err" --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_data.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then finalize=DRY_RUN_FINALIZE; fi
graph=$(printf '{"dry_run":%s,"freeze":"%s","pack":"%s","finalize":"%s"}' "$DRY_RUN" "$freeze" "$pack" "$finalize")
if [[ "$DRY_RUN" == 0 ]]; then printf '%s\n' "$graph" >"$FULL8_STAGE_ROOT/submission.json"; fi
printf '%s\n' "$graph"

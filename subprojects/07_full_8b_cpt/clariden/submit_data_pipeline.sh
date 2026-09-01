#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_STAGE_ROOT:?set new immutable data stage root}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    echo DRY_RUN_ID
  else sbatch --parsable "$@"
  fi
}

export_arg="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT"
freeze=$(submit --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/freeze_data_inventory.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then freeze=DRY_RUN_FREEZE; fi
pack=$(submit --dependency="afterok:$freeze" --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/pack_full_data.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then pack=DRY_RUN_PACK; fi
finalize=$(submit --dependency="afterok:$pack" --export="$export_arg" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_data.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then finalize=DRY_RUN_FINALIZE; fi
printf '{"dry_run":%s,"freeze":"%s","pack":"%s","finalize":"%s"}\n' "$DRY_RUN" "$freeze" "$pack" "$finalize"

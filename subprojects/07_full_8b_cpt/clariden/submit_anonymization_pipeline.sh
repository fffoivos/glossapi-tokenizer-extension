#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set new anonymized derivative root}"
: "${FULL8_PARENT_SOURCE_ROOT:?set frozen parent binary stage}"
: "${FULL8_VALIDATION_MANIFEST:?set corrected 13-panel validation manifest}"
TASK_COUNT=${FULL8_ANON_TASK_COUNT:-1457}
GROUP_SIZE=${FULL8_ANON_GROUP_SIZE:-4}
MAX_CONCURRENT=${FULL8_ANON_MAX_CONCURRENT:-16}
DRY_RUN=${DRY_RUN:-1}
((TASK_COUNT > 0 && GROUP_SIZE > 0 && MAX_CONCURRENT > 0))
groups=$(((TASK_COUNT + GROUP_SIZE - 1) / GROUP_SIZE))
last_group=$((groups - 1))
export_values="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_ANON_STAGE_ROOT=$FULL8_ANON_STAGE_ROOT,FULL8_PARENT_SOURCE_ROOT=$FULL8_PARENT_SOURCE_ROOT,FULL8_VALIDATION_MANIFEST=$FULL8_VALIDATION_MANIFEST,FULL8_ANON_TASK_COUNT=$TASK_COUNT,FULL8_ANON_GROUP_SIZE=$GROUP_SIZE"

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    echo DRY_RUN
  else
    sbatch --parsable "$@"
  fi
}
freeze=$(submit --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/freeze_anonymization_overlay.sbatch")
freeze=${freeze%%;*}
smoke=$(submit --dependency="afterok:$freeze" --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/smoke_anonymization_pipeline.sbatch")
smoke=${smoke%%;*}
inventory=$(submit --dependency="afterok:$smoke" --array="0-${last_group}%${MAX_CONCURRENT}" --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/build_anonymization_inventory_group.sbatch")
inventory=${inventory%%;*}
dedup=$(submit --dependency="afterok:$inventory" --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_postmask_dedup.sbatch")
dedup=${dedup%%;*}
binary=$(submit --dependency="afterok:$dedup" --array="0-${last_group}%${MAX_CONCURRENT}" --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/build_sanitized_binary_group.sbatch")
binary=${binary%%;*}
finalize=$(submit --dependency="afterok:$binary" --export="$export_values" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_sanitized_bridge.sbatch")
finalize=${finalize%%;*}
printf '{"dry_run":%s,"tasks":%s,"groups":%s,"max_concurrent":%s,"freeze":"%s","smoke":"%s","inventory":"%s","dedup":"%s","binary":"%s","finalize":"%s"}\n' \
  "$DRY_RUN" "$TASK_COUNT" "$groups" "$MAX_CONCURRENT" "$freeze" "$smoke" "$inventory" "$dedup" "$binary" "$finalize"

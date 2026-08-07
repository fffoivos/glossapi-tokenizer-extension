#!/usr/bin/env bash
set -euo pipefail

: "${FULL8_CODE_ROOT:?set immutable selection code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable selection code receipt}"
: "${FULL8_PARITY_CODE_ROOT:?set immutable code root used by parity jobs}"
: "${FULL8_PARITY_CODE_BUNDLE_RECEIPT:?set immutable parity code receipt}"
: "${FULL8_STAGE_ROOT:?set completed sanitized D0 stage}"
: "${FULL8_PARITY_ROOT:?set synchronous DP32 parity root}"
: "${FULL8_BENCHMARK_ROOT:?set new DP32 selection evidence root}"
: "${FULL8_SELECTION_LOG_ROOT:?set separate coordinator log root}"
: "${FULL8_PARITY_GATE_JOB_ID:?set the queued parity-finalizer job id}"
FULL8_RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
FULL8_PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}

[[ ! -e "$FULL8_BENCHMARK_ROOT" ]] || {
  echo "DP32 selection root already exists: $FULL8_BENCHMARK_ROOT" >&2
  exit 2
}

submit=(sbatch --parsable --dependency="afterok:$FULL8_PARITY_GATE_JOB_ID"
  --output="$FULL8_SELECTION_LOG_ROOT/%x-%j.out"
  --error="$FULL8_SELECTION_LOG_ROOT/%x-%j.err"
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_PARITY_CODE_ROOT=$FULL8_PARITY_CODE_ROOT,FULL8_PARITY_CODE_BUNDLE_RECEIPT=$FULL8_PARITY_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_PARITY_ROOT=$FULL8_PARITY_ROOT,FULL8_BENCHMARK_ROOT=$FULL8_BENCHMARK_ROOT,FULL8_RECIPE=$FULL8_RECIPE,FULL8_PROFILES=$FULL8_PROFILES"
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_dp32_fallback_selection.sbatch")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'mkdir -p %q\n' "$FULL8_SELECTION_LOG_ROOT" >&2
  printf '%q ' "${submit[@]}" >&2
  printf '\n' >&2
  printf '{"dry_run":true,"selection_gate":"DRY_RUN","dependency":"%s"}\n' \
    "$FULL8_PARITY_GATE_JOB_ID"
  exit 0
fi

[[ "${CONFIRM_DP32_FALLBACK_SELECTION:-}" == "APERTUS8B_SELECT_PROVEN_DP32" ]] || {
  echo "set CONFIRM_DP32_FALLBACK_SELECTION=APERTUS8B_SELECT_PROVEN_DP32" >&2
  exit 2
}
mkdir -p "$FULL8_SELECTION_LOG_ROOT"
job_id=$("${submit[@]}")
printf '{"dry_run":false,"selection_gate":"%s","dependency":"%s"}\n' \
  "$job_id" "$FULL8_PARITY_GATE_JOB_ID"

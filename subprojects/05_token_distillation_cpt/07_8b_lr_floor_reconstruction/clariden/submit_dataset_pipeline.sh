#!/usr/bin/env bash
# Submit the minimal replay supplement overlay and immutable freeze on Bristen.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LR13_CODE_ROOT="${LR13_CODE_ROOT:-$(cd "$HERE/.." && pwd)}"
source "$HERE/paths.env"
cluster=$(scontrol show config 2>/dev/null | sed -n 's/^ClusterName *= *//p' | head -1)
[[ "$cluster" == bristen ]] || { echo "ERROR: CPU data pipeline must be submitted from Bristen (got ${cluster:-unknown})" >&2; exit 2; }
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_DATA_BUILD="${CONFIRM_DATA_BUILD:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$DRY_RUN" == 1 || "$CONFIRM_DATA_BUILD" == 1 ]] || { echo "ERROR: live build requires CONFIRM_DATA_BUILD=1" >&2; exit 2; }
mkdir -p "$LR13_RUN_ROOT" "$LR13_DATA_ROOT"
submit() {
  local dry_id="$1"
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY:' >&2; printf ' %q' sbatch --parsable "$@" >&2; printf '\n' >&2
    echo "$dry_id"
  else
    sbatch --parsable "$@"
  fi
}
common="ALL,LR13_CODE_ROOT=$LR13_CODE_ROOT,LR13_RUN_ID=$LR13_RUN_ID"
runtime=$(submit DRY_RUNTIME --account=a0140 --partition=normal --output="$LR13_RUN_ROOT/%x-%j.out" --error="$LR13_RUN_ROOT/%x-%j.err" --export="$common" "$HERE/bootstrap_bristen_runtime.sbatch")
download=$(submit DRY_DOWNLOAD --account=a0140 --partition=normal --dependency="afterok:$runtime" --output="$LR13_RUN_ROOT/%x-%j.out" --error="$LR13_RUN_ROOT/%x-%j.err" --export="$common" "$HERE/download_replay_supplements.sbatch")
plan=$(submit DRY_PLAN --account=a0140 --partition=normal --dependency="afterok:$download" --output="$LR13_RUN_ROOT/%x-%j.out" --error="$LR13_RUN_ROOT/%x-%j.err" --export="$common" "$HERE/prepare_replay_supplements.sbatch")
bins=$(submit DRY_BINARIES --account=a0140 --partition=normal --dependency="afterok:$plan" --array="0-7%8" --output="$LR13_RUN_ROOT/%x-%A_%a.out" --error="$LR13_RUN_ROOT/%x-%A_%a.err" --export="$common" "$HERE/build_replay_supplements.sbatch")
freeze=$(submit DRY_FREEZE --account=a0140 --partition=normal --dependency="afterok:$bins" --output="$LR13_RUN_ROOT/%x-%j.out" --error="$LR13_RUN_ROOT/%x-%j.err" --export="$common" "$HERE/freeze_dataset.sbatch")
printf 'runtime=%s\ndownload=%s\nplan=%s\nbinaries=%s\nfreeze=%s\n' "$runtime" "$download" "$plan" "$bins" "$freeze"

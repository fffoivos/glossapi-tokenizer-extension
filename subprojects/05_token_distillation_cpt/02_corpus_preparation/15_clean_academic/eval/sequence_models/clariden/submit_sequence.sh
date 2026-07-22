#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(git -C "$HERE" rev-parse --show-toplevel)}
ACTION=${1:-}
case "$ACTION" in
  packets) SCRIPT="$HERE/40_build_human_gold_packets.sbatch" ;;
  import-gold) SCRIPT="$HERE/50_import_human_gold.sbatch" ;;
  fit-feature) SCRIPT="$HERE/60_fit_feature_sequence.sbatch" ;;
  *) echo "usage: $0 {packets|import-gold|fit-feature}" >&2; exit 2 ;;
esac

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: Clariden handoff requires a clean exact commit" >&2
  exit 3
fi
COMMAND=(sbatch --export="ALL,SEQUENCE_CLARIDEN_DIR=$HERE,PHASE04_EXPECTED_COMMIT=$COMMIT,REPO_ROOT=$REPO_ROOT" "$SCRIPT")
printf 'DRY_RUN:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${CONFIRM_LAUNCH:-0}" == "1" ]]; then
  "${COMMAND[@]}"
else
  echo "Set CONFIRM_LAUNCH=1 to submit. No job was launched."
fi

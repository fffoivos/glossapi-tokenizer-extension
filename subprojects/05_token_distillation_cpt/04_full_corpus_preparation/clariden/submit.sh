#!/usr/bin/env bash
# Dry-run-first Phase-04 submission helper.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
target=${1:-}
shift || true

case "$target" in
    bootstrap-runtime) script="$HERE/04_bootstrap_runtime.sbatch" ;;
    build-detector) script="$HERE/05_build_detector.sbatch" ;;
    acquire) script="$HERE/00_acquire_sources.sbatch" ;;
    quality) script="$HERE/10_quality_audit.sbatch" ;;
    structural-detect) script="$HERE/20_structural_detect.sbatch" ;;
    structural-token-loss) script="$HERE/30_structural_token_loss.sbatch" ;;
    *)
        echo "usage: $0 bootstrap-runtime|build-detector|acquire|quality|structural-detect|structural-token-loss [sbatch args...]" >&2
        exit 2
        ;;
esac

PHASE04_DIR=$(cd "$HERE/.." && pwd)
REPO_ROOT=$(git -C "$PHASE04_DIR" rev-parse --show-toplevel)
ACADEMIC_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic"
PHASE04_EXPECTED_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
if [[ "${CONFIRM_LAUNCH:-0}" == "1" && -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: live Phase-04 submission requires a clean exact checkout: $REPO_ROOT" >&2
    git -C "$REPO_ROOT" status --short >&2
    exit 3
fi

command=(
    sbatch
    "$@"
    --export="ALL,REPO_ROOT=$REPO_ROOT,PHASE04_DIR=$PHASE04_DIR,ACADEMIC_DIR=$ACADEMIC_DIR,PHASE04_CLARIDEN_DIR=$HERE,PHASE04_EXPECTED_COMMIT=$PHASE04_EXPECTED_COMMIT"
    "$script"
)
printf 'COMMAND:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${CONFIRM_LAUNCH:-0}" != "1" ]]; then
    echo "DRY RUN: set CONFIRM_LAUNCH=1 to submit."
    exit 0
fi

exec "${command[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=cpt4_acquire_mdc
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "Mozilla Data Collective acquisition"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
[[ -n "${MOZILLA_DATA_COLLECTIVE_API_KEY:-}" ]] || {
    echo "ERROR: MOZILLA_DATA_COLLECTIVE_API_KEY is absent; inject it only in the submission environment" >&2
    exit 81
}

DESTINATION=${MDC_DESTINATION:-$DATA_ROOT/mdc}
OUTPUT=${MDC_ACQUISITION_RECEIPT:-$RUN_ROOT/source_locks/mdc_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}.receipt.json}
mkdir -p "$DESTINATION" "$(dirname "$OUTPUT")"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 82; }
exec 9>"$DESTINATION/.phase04-mdc-acquisition.lock"
flock -n 9 || { echo "ERROR: another MDC acquisition is active" >&2; exit 83; }

source_args=()
for source_id in ${SOURCES:-}; do
    source_args+=(--source "$source_id")
done
CODE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/acquire_mdc_sources.py" \
    --sources "$SOURCE_CONFIG" \
    --destination "$DESTINATION" \
    --output "$OUTPUT" \
    --code-commit "$CODE_COMMIT" \
    "${source_args[@]}"

unset MOZILLA_DATA_COLLECTIVE_API_KEY
echo "MDC_ACQUISITION_RECEIPT=$OUTPUT"
echo "No API key, download token, or presigned URL is stored in this output."

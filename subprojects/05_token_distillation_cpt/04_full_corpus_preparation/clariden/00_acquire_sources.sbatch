#!/usr/bin/env bash
# External source acquisition only. This is a CPU-only workload on a production
# compute node; xfer is reserved for transfers between CSCS filesystems.
#SBATCH --job-name=cpt4_acquire
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh to export the original script directory}
source "$HERE/paths.env"
source "$HERE/common.sh"

phase04_require_compute_partition "external Hugging Face acquisition"
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: export HF_TOKEN at submission; never store it in the repo" >&2; exit 81; }
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

LOCK=${LOCK:-$RUN_ROOT/source_locks/sources_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}.lock.json}
DESTINATION=${DESTINATION:-$DATA_ROOT/hf}
case "$LOCK" in
    *.lock.json) RECEIPT_STEM=${LOCK%.lock.json} ;;
    *.json) RECEIPT_STEM=${LOCK%.json} ;;
    *) RECEIPT_STEM=$LOCK ;;
esac
DOWNLOAD_MANIFEST=${DOWNLOAD_MANIFEST:-${RECEIPT_STEM}.download.json}
SCHEMA_AUDIT=${SCHEMA_AUDIT:-${RECEIPT_STEM}.schemas.json}
ACQUISITION_RECEIPT=${ACQUISITION_RECEIPT:-${RECEIPT_STEM}.receipt.json}
mkdir -p \
    "$(dirname "$LOCK")" \
    "$(dirname "$DOWNLOAD_MANIFEST")" \
    "$(dirname "$SCHEMA_AUDIT")" \
    "$(dirname "$ACQUISITION_RECEIPT")" \
    "$DESTINATION"
command -v flock >/dev/null || { echo "ERROR: flock is required to serialize acquisitions" >&2; exit 82; }
ACQUISITION_LOCK_FILE=${ACQUISITION_LOCK_FILE:-$DESTINATION/.phase04-acquisition.lock}
exec 9>"$ACQUISITION_LOCK_FILE"
flock -n 9 || {
    echo "ERROR: another Phase-04 acquisition is writing under $DESTINATION" >&2
    exit 83
}

source_args=()
for source_id in ${SOURCES:-}; do
    source_args+=(--source "$source_id")
done

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/resolve_sources.py" \
    --sources "$SOURCE_CONFIG" --output "$LOCK" "${source_args[@]}"

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/download_locked_sources.py" \
    --lock "$LOCK" --destination "$DESTINATION" --manifest "$DOWNLOAD_MANIFEST" \
    --workers 12 "${source_args[@]}"

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/verify_staged_schemas.py" \
    --sources "$SOURCE_CONFIG" --lock "$LOCK" --destination "$DESTINATION" --output "$SCHEMA_AUDIT"

CODE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/finalize_acquisition.py" \
    --sources "$SOURCE_CONFIG" \
    --lock "$LOCK" \
    --download-manifest "$DOWNLOAD_MANIFEST" \
    --schema-audit "$SCHEMA_AUDIT" \
    --destination "$DESTINATION" \
    --code-commit "$CODE_COMMIT" \
    --output "$ACQUISITION_RECEIPT"

echo "LOCK=$LOCK"
echo "DESTINATION=$DESTINATION"
echo "DOWNLOAD_MANIFEST=$DOWNLOAD_MANIFEST"
echo "SCHEMA_AUDIT=$SCHEMA_AUDIT"
echo "ACQUISITION_RECEIPT=$ACQUISITION_RECEIPT"

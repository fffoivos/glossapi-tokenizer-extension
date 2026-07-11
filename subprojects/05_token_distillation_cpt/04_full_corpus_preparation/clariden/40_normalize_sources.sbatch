#!/usr/bin/env bash
#SBATCH --job-name=cpt4_normalize
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "full-corpus normalization"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "${ACQUISITION_RECEIPT:?set ACQUISITION_RECEIPT to the passed receipt from job 2735235}"
phase04_require_file "$NORMALIZE_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "10-normalize"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_add_input acquisition_receipt "$ACQUISITION_RECEIPT"

args=(
    --sources "$SOURCE_CONFIG"
    --acquisition-receipt "$ACQUISITION_RECEIPT"
    --output "$PHASE04_STAGE_DIR/canonical"
    --manifest "$PHASE04_STAGE_DIR/normalization_manifest.json"
    --rows-per-shard "${ROWS_PER_SHARD:-50000}"
    --progress-every "${PROGRESS_EVERY:-100000}"
    --workers "${NORMALIZE_WORKERS:-32}"
    --temporary-directory "${NORMALIZE_TEMP_DIRECTORY:-$DATA_ROOT/work/$PIPELINE_RUN_ID/normalize}"
    --duckdb-memory-limit "${NORMALIZE_DUCKDB_MEMORY_LIMIT:-300GB}"
    --duckdb-threads "${NORMALIZE_DUCKDB_THREADS:-32}"
)
for source_id in ${SOURCES:-}; do
    args+=(--source "$source_id")
done
if [[ -n "${MAX_ROWS_PER_SOURCE:-}" ]]; then
    args+=(--max-rows-per-source "$MAX_ROWS_PER_SOURCE")
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SECTION_SQLITE_CACHE_MB="${SECTION_SQLITE_CACHE_MB:-512}"
if [[ ! -s "$PHASE04_STAGE_DIR/normalization_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$NORMALIZE_SCRIPT" "${args[@]}"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/normalization_manifest.json" \
    --exact-parquet-root "$PHASE04_STAGE_DIR/canonical" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/normalization_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json"

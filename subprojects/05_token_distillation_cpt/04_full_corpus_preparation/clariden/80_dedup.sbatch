#!/usr/bin/env bash
#SBATCH --job-name=cpt4_dedup
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "post-clean exact and near deduplication"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$DEDUP_WRAPPER_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "70-dedup"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "60-greekmmlu-decontam"

DECONTAM_DIR="$PIPELINE_RUN_ROOT/stages/60-greekmmlu-decontam"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4
DEDUP_MANIFEST="$PHASE04_STAGE_DIR/dedup_manifest.json"
DEDUP_STATUS="missing"
if [[ -s "$DEDUP_MANIFEST" ]]; then
    DEDUP_STATUS=$(phase04_contract_python get-json-field \
        --path "$DEDUP_MANIFEST" --field status)
fi
if [[ "$DEDUP_STATUS" != "completed" ]]; then
    wrapper_args=(
        --input "$DECONTAM_DIR/corpus"
        --staged-input "$PHASE04_STAGE_DIR/staged-input"
        --state-root "$PHASE04_STAGE_DIR/state"
        --run-root "$PHASE04_STAGE_DIR/run"
        --manifest "$DEDUP_MANIFEST"
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp"
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}"
        --workers "${DEDUP_WORKERS:-240}"
        --duckdb-threads "${DEDUP_DUCKDB_THREADS:-32}"
        --staging-workers "${DEDUP_STAGING_WORKERS:-8}"
        --greek-diacritic-policy preserve
        --minhash-threshold 0.85
        --num-perm 128
        --bands 32
        --rows-per-band 4
        --shingle-mode token
        --shingle-size 5
        --max-bucket-size 5000
    )
    if [[ "$DEDUP_STATUS" == "staged" ]]; then
        if [[ -s "$PHASE04_STAGE_DIR/run/run_config.json" ]]; then
            wrapper_args+=(--resume)
        else
            wrapper_args+=(--reuse-staged)
        fi
    fi
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DEDUP_WRAPPER_SCRIPT" "${wrapper_args[@]}"
fi
[[ "$(phase04_contract_python get-json-field --path "$DEDUP_MANIFEST" --field status)" == "completed" ]] || {
    echo "ERROR: dedup wrapper did not produce a completed manifest" >&2
    exit 120
}
BOUND_DECISIONS=$(phase04_contract_python get-json-field \
    --path "$DEDUP_MANIFEST" --field dedup_output.content_bound_decisions.path)
RUN_SUMMARY=$(phase04_contract_python get-json-field \
    --path "$DEDUP_MANIFEST" --field dedup_output.summary_path)

phase04_contract_python validate-inventory \
    --manifest "$DEDUP_MANIFEST" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$DEDUP_MANIFEST" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$RUN_SUMMARY" \
    "$BOUND_DECISIONS"

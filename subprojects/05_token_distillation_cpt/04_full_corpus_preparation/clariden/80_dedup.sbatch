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
if [[ ! -s "$PHASE04_STAGE_DIR/staging_manifest.json" ]]; then
    # The adapter is run stage-only.  A failed, unpublished staging attempt is
    # retried into a new job-specific directory; no partial input is reused.
    STAGED_INPUT="$PHASE04_STAGE_DIR/staged-input-${SLURM_JOB_ID}"
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DEDUP_WRAPPER_SCRIPT" \
        --input "$DECONTAM_DIR/corpus" \
        --staged-input "$STAGED_INPUT" \
        --state-root "$PHASE04_STAGE_DIR/state" \
        --run-root "$PHASE04_STAGE_DIR/run" \
        --manifest "$PHASE04_STAGE_DIR/staging_manifest.json" \
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --workers "${DEDUP_WORKERS:-240}" \
        --greek-diacritic-policy preserve \
        --minhash-threshold 0.85 \
        --num-perm 128 \
        --bands 32 \
        --rows-per-band 4 \
        --shingle-mode token \
        --shingle-size 5 \
        --max-bucket-size 5000 \
        --stage-only
fi
STAGED_INPUT=$(phase04_contract_python get-json-field \
    --path "$PHASE04_STAGE_DIR/staging_manifest.json" --field staged_input)

if [[ ! -s "$PHASE04_STAGE_DIR/run/final/run_summary.json" || \
      ! -s "$PHASE04_STAGE_DIR/run/final/dedup_decisions.parquet" || \
      ! -s "$PHASE04_STAGE_DIR/dedup_result.json" ]]; then
    cli_args=(
        -m glossapi_corpus_cli.cli dedup-text run
        --input-root "$STAGED_INPUT"
        --state-root "$PHASE04_STAGE_DIR/state"
        --run-root "$PHASE04_STAGE_DIR/run"
        --max-workers "${DEDUP_WORKERS:-240}"
        --greek-diacritic-policy preserve
        --minhash-threshold 0.85
        --num-perm 128
        --bands 32
        --rows-per-band 4
        --shingle-mode token
        --shingle-size 5
        --max-bucket-size 5000
    )
    [[ -s "$PHASE04_STAGE_DIR/run/run_config.json" ]] && cli_args+=(--resume)
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "${cli_args[@]}" \
        | tee "$PHASE04_STAGE_DIR/dedup_result.json"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/staging_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/staging_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$PHASE04_STAGE_DIR/dedup_result.json" \
    "$PHASE04_STAGE_DIR/run/final/run_summary.json" \
    "$PHASE04_STAGE_DIR/run/final/dedup_decisions.parquet"

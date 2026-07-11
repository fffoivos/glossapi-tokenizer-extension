#!/usr/bin/env bash
#SBATCH --job-name=cpt4_materialize
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
phase04_require_compute_partition "release materialization and validation"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$MATERIALIZE_SCRIPT"
phase04_require_file "$VALIDATE_RELEASE_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "80-materialize-validate"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "70-dedup"

DECONTAM_DIR="$PIPELINE_RUN_ROOT/stages/60-greekmmlu-decontam"
DEDUP_DIR="$PIPELINE_RUN_ROOT/stages/70-dedup"
phase04_require_file "$DECONTAM_DIR/final_clean_stage.txt"
FINAL_CLEAN_STAGE=$(<"$DECONTAM_DIR/final_clean_stage.txt")
case "$FINAL_CLEAN_STAGE" in
    50-clean|58-final-clean) ;;
    *) echo "ERROR: invalid recorded final cleaning stage: $FINAL_CLEAN_STAGE" >&2; exit 108 ;;
esac
phase04_stage_require_upstream "$FINAL_CLEAN_STAGE"
CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/$FINAL_CLEAN_STAGE"
DECISIONS="$DEDUP_DIR/run/final/dedup_decisions.parquet"
phase04_stage_add_input dedup_decisions "$DECISIONS"
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/release_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$MATERIALIZE_SCRIPT" \
        --input "$DECONTAM_DIR/corpus" \
        --dedup-decisions "$DECISIONS" \
        --cleaning-ledger "$CLEAN_DIR/ledger" \
        --decontam-ledger "$DECONTAM_DIR/ledger" \
        --output "$PHASE04_STAGE_DIR/release" \
        --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
        --token-waterfall "$PHASE04_STAGE_DIR/token_waterfall.json" \
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --threads "${DUCKDB_THREADS:-128}"
fi

if [[ ! -s "$PHASE04_STAGE_DIR/validation.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$VALIDATE_RELEASE_SCRIPT" \
        --release "$PHASE04_STAGE_DIR/release" \
        --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
        --dedup-decisions "$DECISIONS" \
        --output "$PHASE04_STAGE_DIR/validation.json" \
        --temporary-directory "$PHASE04_STAGE_DIR/duckdb-tmp" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --threads "${DUCKDB_THREADS:-128}"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/release_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/release_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$PHASE04_STAGE_DIR/token_waterfall.json" \
    "$PHASE04_STAGE_DIR/validation.json"
echo "LOCAL_RELEASE_VALIDATED=$PHASE04_STAGE_DIR/release"
echo "Publication remains a separate, explicitly confirmed stage."

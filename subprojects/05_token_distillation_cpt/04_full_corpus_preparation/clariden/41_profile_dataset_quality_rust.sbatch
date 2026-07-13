#!/usr/bin/env bash
#SBATCH --job-name=cpt4_rust_quality
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
export PYTHONPATH="$GLOSSAPI_QUALITY_MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
phase04_require_compute_partition "receipt-bound GlossAPI Rust quality diagnostics"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$DATASET_QUALITY_SCRIPT"
phase04_require_file "$DATASET_REVIEW_SAMPLE_EXPORT_SCRIPT"
phase04_require_file "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT"
phase04_require_file "$POST_DECEMBER_INVENTORY"
phase04_init_pipeline_run

# The default is the exact Stage30 representative sample.  It is intentionally
# a later canonical stage.  The optional raw full scan can start after Stage10
# and resumes independently under its own stage name.
QUALITY_MODE=${QUALITY_MODE:-review_sample}
case "$QUALITY_MODE" in
    review_sample) QUALITY_STAGE=35-dataset-quality-sample ;;
    full_scan) QUALITY_STAGE=15-dataset-quality-full ;;
    *) echo "ERROR: QUALITY_MODE must be review_sample or full_scan" >&2; exit 111 ;;
esac
phase04_stage_begin "$QUALITY_STAGE"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "10-normalize"
if [[ "$QUALITY_MODE" == "review_sample" ]]; then
    phase04_stage_require_upstream "30-review-packet"
fi

NORMALIZE_DIR="$PIPELINE_RUN_ROOT/stages/10-normalize"
NORMALIZATION_MANIFEST="$NORMALIZE_DIR/normalization_manifest.json"
CANONICAL_ROOT="$NORMALIZE_DIR/canonical"
BATCH_SIZE=4096
QUANTILE_SAMPLE_SIZE=${QUANTILE_SAMPLE_SIZE:-8192}
THREADS=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is unset}
QUALITY_OUT="$PHASE04_STAGE_DIR/quality"
QUALITY_SITE_HANDOFF="$PHASE04_STAGE_DIR/dataset_quality_site_handoff.json"
SCRATCH_ROOT=${SLURM_TMPDIR:-$DATA_ROOT/tmp/$PIPELINE_RUN_ID/dataset-quality/$SLURM_JOB_ID}

phase04_stage_add_input normalization_manifest "$NORMALIZATION_MANIFEST"
phase04_stage_add_input rust_quality_build_receipt "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT"
phase04_stage_add_input post_december_inventory "$POST_DECEMBER_INVENTORY"
phase04_stage_bind_parameter scan_mode "$QUALITY_MODE"
phase04_stage_bind_parameter batch_size "$BATCH_SIZE"
phase04_stage_bind_parameter quantile_sample_size "$QUANTILE_SAMPLE_SIZE"
phase04_stage_bind_parameter rust_threads "$THREADS"
phase04_contract_python validate-inventory \
    --manifest "$NORMALIZATION_MANIFEST" \
    --exact-parquet-root "$CANONICAL_ROOT" \
    --output "$PHASE04_STAGE_DIR/normalization_inventory_validation.json"

mkdir -p "$SCRATCH_ROOT"
export CUDA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export TMPDIR="$SCRATCH_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4

args=(
    run
    --scan-mode "$QUALITY_MODE"
    --normalization-manifest "$NORMALIZATION_MANIFEST"
    --canonical-root "$CANONICAL_ROOT"
    --build-receipt "$GLOSSAPI_RUST_QUALITY_BUILD_RECEIPT"
    --expected-commit "$GLOSSAPI_QUALITY_COMMIT"
    --batch-size "$BATCH_SIZE"
    --quantile-sample-size "$QUANTILE_SAMPLE_SIZE"
    --threads "$THREADS"
    --scratch-dir "$SCRATCH_ROOT"
    --output-dir "$QUALITY_OUT"
    --site-handoff "$QUALITY_SITE_HANDOFF"
)

if [[ "$QUALITY_MODE" == "review_sample" ]]; then
    REVIEW_REQUESTS=${REVIEW_REQUESTS:-$PIPELINE_RUN_ROOT/stages/30-review-packet/requests.jsonl}
    SAMPLE_PACKET="$PHASE04_STAGE_DIR/complete_review_samples.jsonl"
    SAMPLE_RECEIPT="$PHASE04_STAGE_DIR/complete_review_samples_receipt.json"
    SAMPLE_ATTESTATION="$PHASE04_STAGE_DIR/complete_review_samples_site_attestation.json"
    phase04_stage_add_input review_requests "$REVIEW_REQUESTS"
    export_args=(
        --normalization-manifest "$NORMALIZATION_MANIFEST"
        --canonical-root "$CANONICAL_ROOT"
        --review-requests "$REVIEW_REQUESTS"
        --scratch-dir "$SCRATCH_ROOT"
        --checkpoint-dir "$PHASE04_STAGE_DIR/sample-export-checkpoints"
        --output "$SAMPLE_PACKET"
        --receipt "$SAMPLE_RECEIPT"
        --site-attestation "$SAMPLE_ATTESTATION"
    )
    [[ "${RESUME_STAGE:-0}" == "1" ]] && export_args+=(--resume)
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DATASET_REVIEW_SAMPLE_EXPORT_SCRIPT" "${export_args[@]}"
    phase04_stage_add_input complete_review_samples_receipt "$SAMPLE_RECEIPT"
    args+=(
        --review-sample-packet "$SAMPLE_PACKET"
        --review-sample-receipt "$SAMPLE_RECEIPT"
        --review-sample-attestation "$SAMPLE_ATTESTATION"
        --review-requests "$REVIEW_REQUESTS"
    )
else
    [[ "${INCLUDE_NANOCHAT_BASE:-0}" == "0" || "${INCLUDE_NANOCHAT_BASE:-0}" == "1" ]] || {
        echo "ERROR: INCLUDE_NANOCHAT_BASE must be 0 or 1" >&2
        exit 112
    }
    phase04_stage_bind_parameter include_nanochat_base "${INCLUDE_NANOCHAT_BASE:-0}"
    [[ "${INCLUDE_NANOCHAT_BASE:-0}" == "1" ]] && args+=(--include-base)
fi
[[ "${RESUME_STAGE:-0}" == "1" ]] && args+=(--resume)

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$DATASET_QUALITY_SCRIPT" "${args[@]}"

SUMMARY="$QUALITY_OUT/dataset_quality_summary_v2.json"
DOCUMENTS="$QUALITY_OUT/dataset_quality_document_v2.parquet"
outputs=(
    "$PHASE04_STAGE_DIR/normalization_inventory_validation.json"
    "$QUALITY_OUT/contract.json"
    "$SUMMARY"
    "$DOCUMENTS"
    "$QUALITY_SITE_HANDOFF"
)
if [[ "$QUALITY_MODE" == "review_sample" ]]; then
    outputs+=("$SAMPLE_PACKET" "$SAMPLE_RECEIPT" "$SAMPLE_ATTESTATION")
fi
phase04_stage_finish "${outputs[@]}"
echo "DATASET_QUALITY_COMPLETE=$SUMMARY"
echo "DATASET_QUALITY_MODE=$QUALITY_MODE"
echo "No cleaned corpus text was persisted by this diagnostic stage."

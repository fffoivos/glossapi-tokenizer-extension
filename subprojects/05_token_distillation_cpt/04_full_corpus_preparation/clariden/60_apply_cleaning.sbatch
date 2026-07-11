#!/usr/bin/env bash
#SBATCH --job-name=cpt4_clean
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
phase04_require_compute_partition "approved source cleaning and PII masking"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$CLEAN_SCRIPT"
phase04_require_file "$TOKENIZER_JSON"
phase04_require_file "${SOURCE_ADMISSION:?set SOURCE_ADMISSION after reviewing the aggregate}"
[[ -n "${CONFIRM_ADMISSION_SHA256:-}" ]] || {
    echo "ERROR: set CONFIRM_ADMISSION_SHA256 to the inspected admission file's sha256." >&2
    exit 101
}
phase04_init_pipeline_run
phase04_stage_begin "50-clean"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "40-review-aggregate"
phase04_contract_python validate-admission \
    --path "$SOURCE_ADMISSION" --expected-sha256 "$CONFIRM_ADMISSION_SHA256"
phase04_stage_add_input manually_confirmed_admission "$SOURCE_ADMISSION"
phase04_stage_add_input tokenizer "$TOKENIZER_JSON"
phase04_stage_add_input eligibility_policy "$TRAINING_ELIGIBILITY_POLICY"

NORMALIZE_DIR="$PIPELINE_RUN_ROOT/stages/10-normalize"
args=(
    --input "$NORMALIZE_DIR/canonical"
    --output "$PHASE04_STAGE_DIR/corpus"
    --quarantine "$PHASE04_STAGE_DIR/quarantine"
    --ledger "$PHASE04_STAGE_DIR/ledger"
    --manifest "$PHASE04_STAGE_DIR/cleaning_manifest.json"
    --source-admission "$SOURCE_ADMISSION"
    --eligibility-policy "$TRAINING_ELIGIBILITY_POLICY"
    --cleaning-policy "$CLEANING_POLICY"
    --tokenizer-json "$TOKENIZER_JSON"
)
if [[ -n "${DOCUMENT_ACTIONS:-}" ]]; then
    IFS=: read -r -a action_paths <<<"$DOCUMENT_ACTIONS"
    for path in "${action_paths[@]}"; do
        phase04_stage_add_input "document_actions:$(basename "$path")" "$path"
        args+=(--document-actions "$path")
    done
fi
if [[ "${APPLY_STRUCTURAL:-0}" == "1" ]]; then
    phase04_require_file "${STRUCTURAL_MODEL_RECEIPT:?set STRUCTURAL_MODEL_RECEIPT to the passed classifier receipt}"
    [[ -n "${STRUCTURAL_SPANS:-}" ]] || { echo "ERROR: STRUCTURAL_SPANS is required" >&2; exit 102; }
    phase04_stage_add_input structural_model_receipt "$STRUCTURAL_MODEL_RECEIPT"
    IFS=: read -r -a span_paths <<<"$STRUCTURAL_SPANS"
    for path in "${span_paths[@]}"; do
        phase04_stage_add_input "structural_spans:$(basename "$path")" "$path"
        args+=(--structural-spans "$path")
    done
    args+=(--apply-structural)
fi
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/cleaning_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$CLEAN_SCRIPT" "${args[@]}"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/cleaning_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/cleaning_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json"

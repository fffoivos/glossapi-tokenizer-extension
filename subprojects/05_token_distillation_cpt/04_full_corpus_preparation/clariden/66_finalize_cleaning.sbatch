#!/usr/bin/env bash
#SBATCH --job-name=cpt4_final_clean
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
phase04_require_compute_partition "terminal admission plus structural-last finalization"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$FINALIZE_CLEAN_SCRIPT"
phase04_require_file "$TOKENIZER_JSON"
phase04_require_file "${FINAL_SOURCE_ADMISSION:?set FINAL_SOURCE_ADMISSION after post-clean review}"
[[ -n "${CONFIRM_FINAL_ADMISSION_SHA256:-}" ]] || {
    echo "ERROR: set CONFIRM_FINAL_ADMISSION_SHA256 after inspecting final admission." >&2
    exit 105
}
phase04_init_pipeline_run
phase04_stage_begin "58-final-clean"
STAGE50_DIR="$PIPELINE_RUN_ROOT/stages/50-clean"
STAGE50_MANIFEST="$STAGE50_DIR/cleaning_manifest.json"

[[ "${APPLY_STRUCTURAL+x}" == "x" ]] || {
    echo "ERROR: Stage58 requires an explicit APPLY_STRUCTURAL=0 or APPLY_STRUCTURAL=1." >&2
    exit 108
}
case "$APPLY_STRUCTURAL" in
    0) STRUCTURAL_FINALIZATION_MODE=no_op ;;
    1) STRUCTURAL_FINALIZATION_MODE=apply ;;
    *)
        echo "ERROR: APPLY_STRUCTURAL must be exactly 0 or 1." >&2
        exit 108
        ;;
esac

decision_args=(
    validate-structural-model
    --requested-mode "$STRUCTURAL_FINALIZATION_MODE"
    --stage50-cleaning-manifest "$STAGE50_MANIFEST"
    --cleaning-policy "$CLEANING_POLICY"
    --output "$PHASE04_STAGE_DIR/structural_application_decision.json"
)
request_args=(
    freeze-structural-request
    --requested-mode "$STRUCTURAL_FINALIZATION_MODE"
    --output "$PHASE04_STAGE_DIR/structural_finalization_request.json"
)
if [[ "$APPLY_STRUCTURAL" == "1" ]]; then
    PROMOTED_DIR="$PIPELINE_RUN_ROOT/stages/54-structural-promote"
    STRUCTURAL_MODEL_RECEIPT="${STRUCTURAL_MODEL_RECEIPT:-$PROMOTED_DIR/academic_structural_model_receipt.json}"
    STRUCTURAL_SPANS="${STRUCTURAL_SPANS:-$PROMOTED_DIR/structural_spans.jsonl}"
    phase04_require_file "$STRUCTURAL_MODEL_RECEIPT"
    phase04_require_file "$STRUCTURAL_SPANS"
    [[ -n "${CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256:-}" ]] || {
        echo "ERROR: structural apply requires CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256." >&2
        exit 108
    }
    [[ "$(sha256sum "$STRUCTURAL_MODEL_RECEIPT" | awk '{print $1}')" == \
        "$CONFIRM_STRUCTURAL_MODEL_RECEIPT_SHA256" ]] || {
        echo "ERROR: structural model receipt confirmation hash mismatch." >&2
        exit 108
    }
    [[ "$(realpath "$STRUCTURAL_MODEL_RECEIPT")" == "$(realpath "$PROMOTED_DIR/academic_structural_model_receipt.json")" ]] || {
        echo "ERROR: Stage58 only accepts the receipt emitted by Stage54." >&2
        exit 108
    }
    [[ "$(realpath "$STRUCTURAL_SPANS")" == "$(realpath "$PROMOTED_DIR/structural_spans.jsonl")" ]] || {
        echo "ERROR: Stage58 only accepts the spans emitted by Stage54." >&2
        exit 108
    }
    phase04_contract_python validate-stage \
        --stage-dir "$PROMOTED_DIR" \
        --stage "54-structural-promote" \
        --run-id "$PIPELINE_RUN_ID" \
        --code-commit "$PHASE04_EXPECTED_COMMIT"
    request_args+=(--receipt "$STRUCTURAL_MODEL_RECEIPT")
    decision_args+=(--receipt "$STRUCTURAL_MODEL_RECEIPT")
else
    [[ "${CONFIRM_STRUCTURAL_NOOP:-0}" == "1" ]] || {
        echo "ERROR: structural no-op requires CONFIRM_STRUCTURAL_NOOP=1." >&2
        exit 108
    }
    [[ -z "${STRUCTURAL_MODEL_RECEIPT:-}" && -z "${STRUCTURAL_SPANS:-}" ]] || {
        echo "ERROR: APPLY_STRUCTURAL=0 must not carry STRUCTURAL_MODEL_RECEIPT or STRUCTURAL_SPANS." >&2
        exit 108
    }
fi

# Freeze and revalidate the operator choice even when Stage58 is resumed or is
# already complete. A changed mode or promoted receipt can never reuse this
# stage directory.
phase04_contract_python "${request_args[@]}"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0

phase04_stage_add_input structural_finalization_request \
    "$PHASE04_STAGE_DIR/structural_finalization_request.json"
phase04_stage_require_upstream "50-clean"
if [[ "${REUSE_STAGE50_ADMISSION:-0}" == "1" ]]; then
    STAGE50_ADMISSION=$(phase04_contract_python get-input-path \
        --receipt "$PIPELINE_RUN_ROOT/stages/50-clean/stage_receipt.json" \
        --name manually_confirmed_admission)
    [[ "$(realpath "$FINAL_SOURCE_ADMISSION")" == "$(realpath "$STAGE50_ADMISSION")" ]] || {
        echo "ERROR: reused terminal admission must be the exact Stage50 admission." >&2
        exit 107
    }
else
    phase04_stage_require_upstream "56-post-clean-review-aggregate"
fi
phase04_contract_python validate-admission \
    --path "$FINAL_SOURCE_ADMISSION" \
    --expected-sha256 "$CONFIRM_FINAL_ADMISSION_SHA256" \
    --require-terminal
phase04_stage_add_input manually_confirmed_admission "$FINAL_SOURCE_ADMISSION"
phase04_stage_add_input tokenizer "$TOKENIZER_JSON"
phase04_stage_add_input eligibility_policy "$TRAINING_ELIGIBILITY_POLICY"
phase04_stage_add_input source_config "$SOURCE_CONFIG"
phase04_stage_add_input source_license_adjudication "$SOURCE_LICENSE_ADJUDICATION"
phase04_stage_add_input cleaning_policy "$CLEANING_POLICY"
phase04_stage_add_input stage50_cleaning_manifest "$STAGE50_MANIFEST"
if [[ "$APPLY_STRUCTURAL" == "1" ]]; then
    phase04_stage_require_upstream "54-structural-promote"
    phase04_stage_add_input structural_model_receipt "$STRUCTURAL_MODEL_RECEIPT"
fi
phase04_contract_python "${decision_args[@]}"

args=(
    --input "$STAGE50_DIR/corpus"
    --input-ledger "$STAGE50_DIR/ledger"
    --input-quarantine "$STAGE50_DIR/quarantine"
    --input-cleaning-manifest "$STAGE50_MANIFEST"
    --output "$PHASE04_STAGE_DIR/corpus"
    --quarantine "$PHASE04_STAGE_DIR/quarantine"
    --ledger "$PHASE04_STAGE_DIR/ledger"
    --manifest "$PHASE04_STAGE_DIR/cleaning_manifest.json"
    --source-admission "$FINAL_SOURCE_ADMISSION"
    --source-config "$SOURCE_CONFIG"
    --license-adjudication "$SOURCE_LICENSE_ADJUDICATION"
    --eligibility-policy "$TRAINING_ELIGIBILITY_POLICY"
    --cleaning-policy "$CLEANING_POLICY"
    --tokenizer-json "$TOKENIZER_JSON"
    --structural-decision "$PHASE04_STAGE_DIR/structural_application_decision.json"
    --work-dir "$PHASE04_STAGE_DIR/work"
    --workers "${CLEANING_WORKERS:-0}"
    --tokenizer-threads "${TOKENIZER_THREADS_PER_WORKER:-0}"
    --batch-rows "${CLEANING_BATCH_ROWS:-2048}"
)
structural_pass=$(phase04_contract_python get-json-field \
    --path "$PHASE04_STAGE_DIR/structural_application_decision.json" \
    --field apply_structural)
if [[ "$structural_pass" == "True" ]]; then
    [[ -n "${STRUCTURAL_SPANS:-}" ]] || {
        echo "ERROR: passed structural receipt requires STRUCTURAL_SPANS." >&2
        exit 106
    }
    IFS=: read -r -a span_paths <<<"$STRUCTURAL_SPANS"
    for path in "${span_paths[@]}"; do
        phase04_stage_add_input "structural_spans:$(basename "$path")" "$path"
        args+=(--structural-spans "$path")
    done
else
    echo "STRUCTURAL_NO_OP=$PHASE04_STAGE_DIR/structural_application_decision.json"
fi

# Stage 58 may change only terminal admission and append reviewed structural
# artifacts. The eligibility ceiling and deterministic policy inputs must be
# byte-identical to the reviewed Stage 50 pass.
phase04_contract_python validate-cleaning-replay \
    --reference-receipt "$STAGE50_DIR/stage_receipt.json" \
    --current-inputs "$PHASE04_STAGE_DIR/stage_inputs.json" \
    --finalizer \
    --output "$PHASE04_STAGE_DIR/cleaning_replay_validation.json"

export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/cleaning_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$FINALIZE_CLEAN_SCRIPT" "${args[@]}"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/cleaning_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/structural_finalization_request.json" \
    "$PHASE04_STAGE_DIR/cleaning_replay_validation.json" \
    "$PHASE04_STAGE_DIR/structural_application_decision.json" \
    "$PHASE04_STAGE_DIR/cleaning_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json"

#!/usr/bin/env bash
#SBATCH --job-name=cpt4_struct_promote
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "manual-audit validation and structural-span promotion"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT"
phase04_require_file "${STRUCTURAL_MANUAL_AUDIT_RECEIPT:?set STRUCTURAL_MANUAL_AUDIT_RECEIPT}"
phase04_require_file "${STRUCTURAL_AUDIT_ANNOTATIONS:?set STRUCTURAL_AUDIT_ANNOTATIONS}"
phase04_require_file "${STRUCTURAL_SILVER_RECEIPT:?set STRUCTURAL_SILVER_RECEIPT}"
phase04_require_file "${STRUCTURAL_SILVER_SPLIT:?set STRUCTURAL_SILVER_SPLIT}"
phase04_require_file "${STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT:?set the passed C0 classifier-selection receipt after the joint ladder}"
phase04_require_file "${STRUCTURAL_PARITY_RECEIPT:?set STRUCTURAL_PARITY_RECEIPT after joint LLM-silver recovery}"
phase04_init_pipeline_run
phase04_stage_begin "54-structural-promote"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "52-structural-detect"
phase04_stage_require_upstream "53-structural-review-packet"

RAW_MANIFEST="$PIPELINE_RUN_ROOT/stages/52-structural-detect/raw_predictions_manifest.json"
PACKET_MANIFEST="$PIPELINE_RUN_ROOT/stages/53-structural-review-packet/review_packet_manifest.json"
phase04_stage_add_input raw_predictions_manifest "$RAW_MANIFEST"
phase04_stage_add_input review_packet_manifest "$PACKET_MANIFEST"
phase04_stage_add_input manual_audit_receipt "$STRUCTURAL_MANUAL_AUDIT_RECEIPT"
phase04_stage_add_input manual_audit_annotations "$STRUCTURAL_AUDIT_ANNOTATIONS"
phase04_stage_add_input llm_silver_receipt "$STRUCTURAL_SILVER_RECEIPT"
phase04_stage_add_input llm_silver_split "$STRUCTURAL_SILVER_SPLIT"
phase04_stage_add_input classifier_selection_receipt "$STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT"
phase04_stage_add_input detector_parity_receipt "$STRUCTURAL_PARITY_RECEIPT"
phase04_stage_add_input cleaning_policy "$CLEANING_POLICY"

if [[ ! -s "$PHASE04_STAGE_DIR/audit_validation.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" validate-audit \
        --packet-manifest "$PACKET_MANIFEST" \
        --manual-receipt "$STRUCTURAL_MANUAL_AUDIT_RECEIPT" \
        --annotations "$STRUCTURAL_AUDIT_ANNOTATIONS" \
        --cleaning-policy "$CLEANING_POLICY" \
        --output "$PHASE04_STAGE_DIR/audit_validation.json"
fi
if [[ ! -s "$PHASE04_STAGE_DIR/academic_structural_model_receipt.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" build-model-receipt \
        --raw-manifest "$RAW_MANIFEST" \
        --audit-validation "$PHASE04_STAGE_DIR/audit_validation.json" \
        --silver-receipt "$STRUCTURAL_SILVER_RECEIPT" \
        --silver-split-manifest "$STRUCTURAL_SILVER_SPLIT" \
        --classifier-selection-receipt "$STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT" \
        --parity-receipt "$STRUCTURAL_PARITY_RECEIPT" \
        --output "$PHASE04_STAGE_DIR/academic_structural_model_receipt.json"
fi
if [[ ! -s "$PHASE04_STAGE_DIR/structural_spans_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" rebind \
        --raw-manifest "$RAW_MANIFEST" \
        --model-receipt "$PHASE04_STAGE_DIR/academic_structural_model_receipt.json" \
        --output "$PHASE04_STAGE_DIR/structural_spans.jsonl" \
        --manifest "$PHASE04_STAGE_DIR/structural_spans_manifest.json" \
        --work-dir "$PHASE04_STAGE_DIR/work"
fi
MODEL_VALIDATION="$PHASE04_STAGE_DIR/model_receipt_validation.json"
MODEL_VALIDATION_PARTIAL="${MODEL_VALIDATION}.partial-${SLURM_JOB_ID}"
rm -f "$MODEL_VALIDATION_PARTIAL"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" validate-model-receipt \
    --receipt "$PHASE04_STAGE_DIR/academic_structural_model_receipt.json" \
    --raw-manifest "$RAW_MANIFEST" \
    > "$MODEL_VALIDATION_PARTIAL"
if [[ -s "$MODEL_VALIDATION" ]]; then
    cmp -s "$MODEL_VALIDATION_PARTIAL" "$MODEL_VALIDATION" || {
        echo "ERROR: resumed model-receipt validation output drifted." >&2
        exit 109
    }
    rm -f "$MODEL_VALIDATION_PARTIAL"
else
    mv "$MODEL_VALIDATION_PARTIAL" "$MODEL_VALIDATION"
fi
phase04_require_file "$MODEL_VALIDATION"

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/audit_validation.json" \
    "$PHASE04_STAGE_DIR/academic_structural_model_receipt.json" \
    "$PHASE04_STAGE_DIR/structural_spans.jsonl" \
    "$PHASE04_STAGE_DIR/structural_spans_manifest.json" \
    "$MODEL_VALIDATION"
echo "STRUCTURAL_MODEL_RECEIPT=$PHASE04_STAGE_DIR/academic_structural_model_receipt.json"
echo "STRUCTURAL_SPANS=$PHASE04_STAGE_DIR/structural_spans.jsonl"
echo "Stage58 remains no-op unless APPLY_STRUCTURAL=1 and the tracked policy gates also pass."

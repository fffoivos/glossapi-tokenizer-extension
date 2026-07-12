#!/usr/bin/env bash
#SBATCH --job-name=cpt4_struct_stage50
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
phase04_require_compute_partition "Stage50 academic structural-span detection"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT"
phase04_require_file "$STRUCTURAL_CLASSIFIER_SELECTION_SCRIPT"
phase04_require_file "${REFERENCE_BIN:?set REFERENCE_BIN to the immutable Clariden detector binary}"
phase04_require_file "${DETECTOR_BUILD_RECEIPT:?set DETECTOR_BUILD_RECEIPT}"
phase04_require_file "${STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT:?run the joint ladder and C0 bridge first}"
phase04_require_file "${STRUCTURAL_SILVER_RECEIPT:?set the imported joint source receipt}"
phase04_require_file "${STRUCTURAL_SILVER_SPLIT:?set the imported joint split manifest}"
phase04_init_pipeline_run
phase04_stage_begin "52-structural-detect"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "50-clean"

STAGE50_MANIFEST="$PIPELINE_RUN_ROOT/stages/50-clean/cleaning_manifest.json"
phase04_stage_add_input stage50_cleaning_manifest "$STAGE50_MANIFEST"
phase04_stage_add_input cleaning_policy "$CLEANING_POLICY"
phase04_stage_add_input detector_binary "$REFERENCE_BIN"
phase04_stage_add_input detector_build_receipt "$DETECTOR_BUILD_RECEIPT"
phase04_stage_add_input classifier_selection_receipt "$STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT"
phase04_stage_add_input llm_silver_receipt "$STRUCTURAL_SILVER_RECEIPT"
phase04_stage_add_input llm_silver_split "$STRUCTURAL_SILVER_SPLIT"
phase04_stage_add_input bibliography_model "$STRUCTURAL_BIB_MODEL"
phase04_stage_add_input toc_model "$STRUCTURAL_TOC_MODEL"
phase04_stage_add_input smoother "$STRUCTURAL_SMOOTHER"
phase04_stage_add_input sequence_config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG"
phase04_stage_add_input bibliography_rust_code "$STRUCTURAL_BIB_RUST_CODE"
phase04_stage_add_input toc_rust_code "$STRUCTURAL_TOC_RUST_CODE"

parity_args=()
if [[ -n "${STRUCTURAL_PARITY_RECEIPT:-}" ]]; then
    phase04_require_file "$STRUCTURAL_PARITY_RECEIPT"
    phase04_stage_add_input detector_parity_receipt "$STRUCTURAL_PARITY_RECEIPT"
    parity_args+=(--parity-receipt "$STRUCTURAL_PARITY_RECEIPT")
fi

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$STRUCTURAL_CLASSIFIER_SELECTION_SCRIPT" validate \
    --receipt "$STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT" \
    --source-receipt "$STRUCTURAL_SILVER_RECEIPT" \
    --source-split-manifest "$STRUCTURAL_SILVER_SPLIT" \
    --sequence-config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" \
    --bib-model "$STRUCTURAL_BIB_MODEL" \
    --toc-model "$STRUCTURAL_TOC_MODEL" \
    --smoother "$STRUCTURAL_SMOOTHER" >/dev/null

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" detect \
    --stage50-cleaning-manifest "$STAGE50_MANIFEST" \
    --cleaning-policy "$CLEANING_POLICY" \
    --detector-binary "$REFERENCE_BIN" \
    --detector-build-receipt "$DETECTOR_BUILD_RECEIPT" \
    "${parity_args[@]}" \
    --model-code "$STRUCTURAL_BIB_RUST_CODE" \
    --model-code "$STRUCTURAL_TOC_RUST_CODE" \
    --sequence-config "$STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" \
    --smoother "$STRUCTURAL_SMOOTHER" \
    --bib-model "$STRUCTURAL_BIB_MODEL" \
    --toc-model "$STRUCTURAL_TOC_MODEL" \
    --code-commit "$PHASE04_EXPECTED_COMMIT" \
    --output-dir "$PHASE04_STAGE_DIR/raw" \
    --manifest "$PHASE04_STAGE_DIR/raw_predictions_manifest.json" \
    --batch-rows "${STRUCTURAL_DETECT_BATCH_ROWS:-2048}" \
    --rayon-threads "${STRUCTURAL_RAYON_THREADS:-240}"

phase04_stage_finish "$PHASE04_STAGE_DIR/raw_predictions_manifest.json"
echo "STRUCTURAL_RAW_PREDICTIONS=$PHASE04_STAGE_DIR/raw_predictions_manifest.json"

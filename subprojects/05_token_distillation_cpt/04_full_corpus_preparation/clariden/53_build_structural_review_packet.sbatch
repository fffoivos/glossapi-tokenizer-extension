#!/usr/bin/env bash
#SBATCH --job-name=cpt4_struct_packet
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "structural false-deletion review packet"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "53-structural-review-packet"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "52-structural-detect"
phase04_stage_require_upstream "50-clean"

RAW_MANIFEST="$PIPELINE_RUN_ROOT/stages/52-structural-detect/raw_predictions_manifest.json"
phase04_stage_add_input raw_predictions_manifest "$RAW_MANIFEST"
phase04_stage_add_input tokenizer "$TOKENIZER_JSON"
if [[ ! -s "$PHASE04_STAGE_DIR/token_loss_report.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" token-loss \
        --raw-manifest "$RAW_MANIFEST" \
        --tokenizer-json "$TOKENIZER_JSON" \
        --per-document "$PHASE04_STAGE_DIR/token_loss_per_document.parquet" \
        --report "$PHASE04_STAGE_DIR/token_loss_report.json" \
        --work-dir "$PHASE04_STAGE_DIR/token-loss-work" \
        --batch-rows "${STRUCTURAL_LOSS_BATCH_ROWS:-2048}" \
        --tokenizer-batch-docs "${STRUCTURAL_LOSS_TOKENIZER_BATCH_DOCS:-64}"
fi
if [[ ! -s "$PHASE04_STAGE_DIR/review_packet_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$STRUCTURAL_SPAN_PRODUCTION_SCRIPT" build-review-packet \
        --raw-manifest "$RAW_MANIFEST" \
        --packet "$PHASE04_STAGE_DIR/review_cases.jsonl" \
        --manifest "$PHASE04_STAGE_DIR/review_packet_manifest.json" \
        --work-dir "$PHASE04_STAGE_DIR/work" \
        --cases 100 \
        --context-chars "${STRUCTURAL_REVIEW_CONTEXT_CHARS:-600}" \
        --batch-rows "${STRUCTURAL_REVIEW_BATCH_ROWS:-2048}"
fi

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/token_loss_per_document.parquet" \
    "$PHASE04_STAGE_DIR/token_loss_report.json" \
    "$PHASE04_STAGE_DIR/review_cases.jsonl" \
    "$PHASE04_STAGE_DIR/review_packet_manifest.json"
echo "MANUAL_STRUCTURAL_AUDIT_REQUIRED=$PHASE04_STAGE_DIR/review_cases.jsonl"
echo "STRUCTURAL_TOKEN_LOSS_REPORT=$PHASE04_STAGE_DIR/token_loss_report.json"
echo "AUTOMATIC_ADJUDICATION_FORBIDDEN=1"

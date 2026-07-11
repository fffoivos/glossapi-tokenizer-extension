#!/usr/bin/env bash
#SBATCH --job-name=cpt4_postclean_gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "post-clean review aggregation gate"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$REVIEW_AGGREGATE_SCRIPT"
phase04_require_file "$MERGE_ADMISSIONS_SCRIPT"
phase04_require_file "${POST_CLEAN_REVIEWS_JSONL:?set POST_CLEAN_REVIEWS_JSONL to completed responses}"
phase04_init_pipeline_run
phase04_stage_begin "56-post-clean-review-aggregate"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "55-post-clean-review-packet"
phase04_stage_require_upstream "50-clean"

PACKET_DIR="$PIPELINE_RUN_ROOT/stages/55-post-clean-review-packet"
PRE_CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/50-clean"
PRE_CLEAN_ADMISSION=$(phase04_contract_python get-input-path \
    --receipt "$PRE_CLEAN_DIR/stage_receipt.json" --name manually_confirmed_admission)
phase04_stage_add_input reviewer_responses "$POST_CLEAN_REVIEWS_JSONL"
phase04_stage_add_input pre_clean_admission "$PRE_CLEAN_ADMISSION"
if [[ ! -s "$PHASE04_STAGE_DIR/post_clean_admission.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$REVIEW_AGGREGATE_SCRIPT" \
        --requests "$PACKET_DIR/requests.jsonl" \
        --packet-summary "$PACKET_DIR/summary.json" \
        --reviews "$POST_CLEAN_REVIEWS_JSONL" \
        --review-policy "$SOURCE_REVIEW_POLICY" \
        --output "$PHASE04_STAGE_DIR/post_clean_admission.json"
fi
if [[ ! -s "$PHASE04_STAGE_DIR/final_admission_candidate.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$MERGE_ADMISSIONS_SCRIPT" \
        --preclean "$PRE_CLEAN_ADMISSION" \
        --postclean "$PHASE04_STAGE_DIR/post_clean_admission.json" \
        --output "$PHASE04_STAGE_DIR/final_admission_candidate.json"
fi

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/post_clean_admission.json" \
    "$PHASE04_STAGE_DIR/final_admission_candidate.json"
candidate_sha=$(sha256sum "$PHASE04_STAGE_DIR/final_admission_candidate.json" | awk '{print $1}')
echo "MANUAL_FINAL_ADMISSION_REVIEW_REQUIRED=$PHASE04_STAGE_DIR/final_admission_candidate.json"
echo "CONFIRM_FINAL_ADMISSION_SHA256=$candidate_sha"

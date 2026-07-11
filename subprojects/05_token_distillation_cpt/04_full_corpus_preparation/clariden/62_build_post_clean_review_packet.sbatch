#!/usr/bin/env bash
#SBATCH --job-name=cpt4_postclean_packet
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
phase04_require_compute_partition "post-clean source-review packet"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$REVIEW_PACKET_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "55-post-clean-review-packet"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "50-clean"

CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/50-clean"
PRE_CLEAN_ADMISSION=$(phase04_contract_python get-input-path \
    --receipt "$CLEAN_DIR/stage_receipt.json" --name manually_confirmed_admission)
phase04_stage_add_input pre_clean_admission "$PRE_CLEAN_ADMISSION"
phase04_stage_add_input review_policy "$SOURCE_REVIEW_POLICY"
if [[ ! -s "$PHASE04_STAGE_DIR/summary.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$REVIEW_PACKET_SCRIPT" \
        --candidate-input "$CLEAN_DIR/corpus" \
        --source-admission "$PRE_CLEAN_ADMISSION" \
        --decision include_after_cleaning \
        --review-policy "$SOURCE_REVIEW_POLICY" \
        --review-phase post_clean \
        --requests-out "$PHASE04_STAGE_DIR/requests.jsonl" \
        --summary-out "$PHASE04_STAGE_DIR/summary.json"
fi

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/requests.jsonl" \
    "$PHASE04_STAGE_DIR/summary.json"
echo "MANUAL_POST_CLEAN_REVIEW_REQUIRED=$PHASE04_STAGE_DIR/requests.jsonl"

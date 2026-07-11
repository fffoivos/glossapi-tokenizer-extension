#!/usr/bin/env bash
#SBATCH --job-name=cpt4_review_gate
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
phase04_require_compute_partition "source-review aggregation gate"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$REVIEW_AGGREGATE_SCRIPT"
phase04_require_file "${REVIEWS_JSONL:?set REVIEWS_JSONL to completed, schema-valid reviewer responses}"
phase04_init_pipeline_run
phase04_stage_begin "40-review-aggregate"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "30-review-packet"

PACKET_DIR="$PIPELINE_RUN_ROOT/stages/30-review-packet"
phase04_stage_add_input reviewer_responses "$REVIEWS_JSONL"
args=(
    --requests "$PACKET_DIR/requests.jsonl"
    --packet-summary "$PACKET_DIR/summary.json"
    --reviews "$REVIEWS_JSONL"
    --review-policy "$SOURCE_REVIEW_POLICY"
    --output "$PHASE04_STAGE_DIR/admission_candidate.json"
)
if [[ -n "${NOVELTY_SUMMARY:-}" ]]; then
    phase04_stage_add_input novelty_summary "$NOVELTY_SUMMARY"
    args+=(--novelty-summary "$NOVELTY_SUMMARY")
fi
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$REVIEW_AGGREGATE_SCRIPT" "${args[@]}"

phase04_stage_finish "$PHASE04_STAGE_DIR/admission_candidate.json"
candidate_sha=$(sha256sum "$PHASE04_STAGE_DIR/admission_candidate.json" | awk '{print $1}')
echo "MANUAL_ADMISSION_REVIEW_REQUIRED=$PHASE04_STAGE_DIR/admission_candidate.json"
echo "CONFIRM_ADMISSION_SHA256=$candidate_sha"
echo "Cleaning is not submitted automatically. Inspect every source decision first."

#!/usr/bin/env bash
#SBATCH --job-name=cpt4_review_packet
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
phase04_require_compute_partition "source-review packet construction"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$REVIEW_PACKET_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "30-review-packet"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "20-lineage"
phase04_stage_require_upstream "10-normalize"

NORMALIZE_DIR="$PIPELINE_RUN_ROOT/stages/10-normalize"
phase04_stage_add_input review_policy "$SOURCE_REVIEW_POLICY"
phase04_stage_add_input normalization_manifest "$NORMALIZE_DIR/normalization_manifest.json"
phase04_contract_python validate-inventory \
    --manifest "$NORMALIZE_DIR/normalization_manifest.json" \
    --exact-parquet-root "$NORMALIZE_DIR/canonical" \
    --output "$PHASE04_STAGE_DIR/normalization_inventory_validation.json"
candidate_args=()
while IFS= read -r -d '' candidate_dir; do
    [[ "$(basename "$candidate_dir")" == "nanochat_base" ]] && continue
    [[ -n "$(find "$candidate_dir" -type f -name '*.parquet' -print -quit)" ]] || continue
    candidate_args+=(--candidate-input "$candidate_dir")
done < <(find "$NORMALIZE_DIR/canonical" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
[[ ${#candidate_args[@]} -gt 0 ]] || { echo "ERROR: no normalized candidate roots" >&2; exit 110; }
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$REVIEW_PACKET_SCRIPT" \
    "${candidate_args[@]}" \
    --review-policy "$SOURCE_REVIEW_POLICY" \
    --review-phase pre_clean \
    --requests-out "$PHASE04_STAGE_DIR/requests.jsonl" \
    --summary-out "$PHASE04_STAGE_DIR/summary.json"

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/normalization_inventory_validation.json" \
    "$PHASE04_STAGE_DIR/requests.jsonl" \
    "$PHASE04_STAGE_DIR/summary.json"
echo "MANUAL_REVIEW_REQUIRED=$PHASE04_STAGE_DIR/requests.jsonl"
echo "No review model is invoked by this job and no admission is approved."

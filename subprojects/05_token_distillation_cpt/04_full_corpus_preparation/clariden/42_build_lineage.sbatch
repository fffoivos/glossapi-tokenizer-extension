#!/usr/bin/env bash
#SBATCH --job-name=cpt4_lineage
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
phase04_require_compute_partition "source-lineage graph"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$LINEAGE_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "20-lineage"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "10-normalize"

NORMALIZE_DIR="$PIPELINE_RUN_ROOT/stages/10-normalize"
mkdir -p "$PHASE04_STAGE_DIR/work"
if [[ ! -s "$PHASE04_STAGE_DIR/summary.json" ]]; then
    BASE_INPUT="$NORMALIZE_DIR/canonical/nanochat_base"
    [[ -d "$BASE_INPUT" ]] || { echo "ERROR: normalized nanochat_base is missing" >&2; exit 109; }
    candidate_args=()
    while IFS= read -r -d '' candidate_dir; do
        [[ "$(basename "$candidate_dir")" == "nanochat_base" ]] && continue
        [[ -n "$(find "$candidate_dir" -type f -name '*.parquet' -print -quit)" ]] || continue
        candidate_args+=(--candidate-input "$candidate_dir")
    done < <(find "$NORMALIZE_DIR/canonical" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    [[ ${#candidate_args[@]} -gt 0 ]] || { echo "ERROR: no normalized candidate roots" >&2; exit 110; }
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$LINEAGE_SCRIPT" rows \
        --base-input "$BASE_INPUT" \
        "${candidate_args[@]}" \
        --registry-manifest-out "$PHASE04_STAGE_DIR/registry.json" \
        --rows-out "$PHASE04_STAGE_DIR/rows.jsonl" \
        --relationships-out "$PHASE04_STAGE_DIR/relationships.jsonl" \
        --summary-out "$PHASE04_STAGE_DIR/summary.json" \
        --sqlite-work-path "$PHASE04_STAGE_DIR/work/lineage-${SLURM_JOB_ID}.sqlite"
fi

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/registry.json" \
    "$PHASE04_STAGE_DIR/summary.json"

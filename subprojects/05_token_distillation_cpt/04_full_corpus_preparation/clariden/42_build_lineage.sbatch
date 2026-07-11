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
phase04_require_file "$APERTUS_OVERLAP_ACTIONS_SCRIPT"
phase04_init_pipeline_run
phase04_stage_begin "20-lineage"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "10-normalize"

NORMALIZE_DIR="$PIPELINE_RUN_ROOT/stages/10-normalize"
phase04_stage_add_input normalization_manifest "$NORMALIZE_DIR/normalization_manifest.json"
phase04_contract_python validate-inventory \
    --manifest "$NORMALIZE_DIR/normalization_manifest.json" \
    --exact-parquet-root "$NORMALIZE_DIR/canonical" \
    --output "$PHASE04_STAGE_DIR/normalization_inventory_validation.json"
LINEAGE_WORK_DIR="${LINEAGE_WORK_DIR:-$DATA_ROOT/work/$PIPELINE_RUN_ID/lineage}"
mkdir -p "$LINEAGE_WORK_DIR/sqlite-tmp"
ACQUISITION_RECEIPT=$(phase04_contract_python get-input-path \
    --receipt "$NORMALIZE_DIR/stage_receipt.json" --name acquisition_receipt)
mkdir -p "$PHASE04_STAGE_DIR/work"
debug_args=()
debug_outputs=()
LINEAGE_DEBUG_EXPORTS_MODE=0
[[ "${LINEAGE_DEBUG_EXPORTS:-0}" != "1" ]] || LINEAGE_DEBUG_EXPORTS_MODE=1
phase04_stage_bind_parameter lineage_debug_exports "$LINEAGE_DEBUG_EXPORTS_MODE"
if [[ "$LINEAGE_DEBUG_EXPORTS_MODE" == "1" ]]; then
    debug_args+=(
        --rows-out "$PHASE04_STAGE_DIR/rows.jsonl"
        --relationships-out "$PHASE04_STAGE_DIR/relationships.jsonl"
    )
    debug_outputs+=(
        "$PHASE04_STAGE_DIR/rows.jsonl"
        "$PHASE04_STAGE_DIR/relationships.jsonl"
    )
fi
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
        "${debug_args[@]}" \
        --registry-manifest-out "$PHASE04_STAGE_DIR/registry.json" \
        --actions-out "$PHASE04_STAGE_DIR/document_actions.jsonl" \
        --novelty-out "$PHASE04_STAGE_DIR/source_novelty.json" \
        --summary-out "$PHASE04_STAGE_DIR/summary.json" \
        --normalization-manifest "$NORMALIZE_DIR/normalization_manifest.json" \
        --normalization-inventory-validation "$PHASE04_STAGE_DIR/normalization_inventory_validation.json" \
        --sqlite-work-path "$LINEAGE_WORK_DIR/lineage.sqlite" \
        --sqlite-temp-directory "$LINEAGE_WORK_DIR/sqlite-tmp" \
        --sqlite-cache-mb "${LINEAGE_SQLITE_CACHE_MB:-32768}" \
        --input-workers "${LINEAGE_INPUT_WORKERS:-16}" \
        --input-spool-directory "$LINEAGE_WORK_DIR/input-fragments" \
        --canonical-verification-interval "${LINEAGE_CANONICAL_VERIFY_EVERY:-100000}" \
        --resume
fi

if [[ ! -s "$PHASE04_STAGE_DIR/apertus_overlap_actions_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$APERTUS_OVERLAP_ACTIONS_SCRIPT" \
        --canonical-root "$NORMALIZE_DIR/canonical" \
        --normalization-manifest "$NORMALIZE_DIR/normalization_manifest.json" \
        --acquisition-receipt "$ACQUISITION_RECEIPT" \
        --sources "$SOURCE_CONFIG" \
        --output "$PHASE04_STAGE_DIR/apertus_overlap_actions.parquet" \
        --manifest "$PHASE04_STAGE_DIR/apertus_overlap_actions_manifest.json" \
        --temporary-directory "$PHASE04_STAGE_DIR/work/apertus-overlay-duckdb" \
        --memory-limit "${DUCKDB_MEMORY_LIMIT:-350GB}" \
        --threads "${DUCKDB_THREADS:-64}"
fi
phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/apertus_overlap_actions_manifest.json" \
    --output "$PHASE04_STAGE_DIR/apertus_overlap_inventory_validation.json"

phase04_stage_finish \
    "$PHASE04_STAGE_DIR/normalization_inventory_validation.json" \
    "$PHASE04_STAGE_DIR/registry.json" \
    "$PHASE04_STAGE_DIR/document_actions.jsonl" \
    "$PHASE04_STAGE_DIR/source_novelty.json" \
    "$PHASE04_STAGE_DIR/apertus_overlap_actions.parquet" \
    "$PHASE04_STAGE_DIR/apertus_overlap_actions_manifest.json" \
    "$PHASE04_STAGE_DIR/apertus_overlap_inventory_validation.json" \
    "$PHASE04_STAGE_DIR/summary.json" \
    "${debug_outputs[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=cpt4_decontam
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
phase04_require_compute_partition "GreekMMLU decontamination"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$DECONTAM_SCRIPT"
phase04_require_file "${GREEKMMLU_QUERIES_JSONL:?set GREEKMMLU_QUERIES_JSONL to immutable benchmark queries}"
phase04_require_file "${GREEKMMLU_BENCHMARK_MANIFEST:?set GREEKMMLU_BENCHMARK_MANIFEST to the frozen query manifest}"
phase04_init_pipeline_run
phase04_stage_begin "60-greekmmlu-decontam"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
FINAL_CLEAN_STAGE=${FINAL_CLEAN_STAGE:?set FINAL_CLEAN_STAGE to 50-clean or 58-final-clean}
case "$FINAL_CLEAN_STAGE" in
    50-clean|58-final-clean) ;;
    *) echo "ERROR: unsupported FINAL_CLEAN_STAGE=$FINAL_CLEAN_STAGE" >&2; exit 107 ;;
esac
phase04_stage_require_upstream "$FINAL_CLEAN_STAGE"
phase04_stage_add_input greekmmlu_queries "$GREEKMMLU_QUERIES_JSONL"
phase04_stage_add_input greekmmlu_benchmark_manifest "$GREEKMMLU_BENCHMARK_MANIFEST"

CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/$FINAL_CLEAN_STAGE"
printf '%s\n' "$FINAL_CLEAN_STAGE" > "$PHASE04_STAGE_DIR/final_clean_stage.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/decontamination_manifest.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DECONTAM_SCRIPT" \
        --input "$CLEAN_DIR/corpus" \
        --output "$PHASE04_STAGE_DIR/corpus" \
        --dropped "$PHASE04_STAGE_DIR/dropped" \
        --ledger "$PHASE04_STAGE_DIR/ledger" \
        --manifest "$PHASE04_STAGE_DIR/decontamination_manifest.json" \
        --queries-jsonl "$GREEKMMLU_QUERIES_JSONL" \
        --benchmark-manifest "$GREEKMMLU_BENCHMARK_MANIFEST" \
        --workers "${DECONTAM_WORKERS:-240}"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/decontamination_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/decontamination_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$PHASE04_STAGE_DIR/final_clean_stage.txt"

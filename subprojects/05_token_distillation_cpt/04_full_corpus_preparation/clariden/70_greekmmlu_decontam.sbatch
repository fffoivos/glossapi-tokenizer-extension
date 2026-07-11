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
phase04_init_pipeline_run
phase04_stage_begin "60-greekmmlu-decontam"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
FINAL_CLEAN_STAGE=${FINAL_CLEAN_STAGE:?set FINAL_CLEAN_STAGE to 50-clean or 58-final-clean}
case "$FINAL_CLEAN_STAGE" in
    50-clean|58-final-clean) ;;
    *) echo "ERROR: unsupported FINAL_CLEAN_STAGE=$FINAL_CLEAN_STAGE" >&2; exit 107 ;;
esac
phase04_stage_require_upstream "$FINAL_CLEAN_STAGE"
phase04_stage_require_upstream "59-greekmmlu-freeze"
GREEKMMLU_DIR="$PIPELINE_RUN_ROOT/stages/59-greekmmlu-freeze"
GREEKMMLU_QUERIES_JSONL="$GREEKMMLU_DIR/queries.jsonl"
GREEKMMLU_BENCHMARK_MANIFEST="$GREEKMMLU_DIR/query_manifest.json"
phase04_stage_add_input greekmmlu_queries "$GREEKMMLU_QUERIES_JSONL"
phase04_stage_add_input greekmmlu_benchmark_manifest "$GREEKMMLU_BENCHMARK_MANIFEST"

CLEAN_DIR="$PIPELINE_RUN_ROOT/stages/$FINAL_CLEAN_STAGE"
printf '%s\n' "$FINAL_CLEAN_STAGE" > "$PHASE04_STAGE_DIR/final_clean_stage.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -s "$PHASE04_STAGE_DIR/decontamination_manifest.json" ]]; then
    ATTEMPT_DIR="$PHASE04_STAGE_DIR/attempts/${SLURM_JOB_ID}"
    mkdir -p "$ATTEMPT_DIR"
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$DECONTAM_SCRIPT" \
        --input "$CLEAN_DIR/corpus" \
        --output "$ATTEMPT_DIR/corpus" \
        --dropped "$ATTEMPT_DIR/dropped" \
        --ledger "$ATTEMPT_DIR/ledger" \
        --manifest "$ATTEMPT_DIR/decontamination_manifest.json" \
        --queries-jsonl "$GREEKMMLU_QUERIES_JSONL" \
        --benchmark-manifest "$GREEKMMLU_BENCHMARK_MANIFEST" \
        --workers "${DECONTAM_WORKERS:-240}"
    # Publish only a fully completed attempt. Each symlink replacement is
    # atomic; the manifest link is last, so an interrupted run is never
    # mistaken for a completed canonical stage on resume.
    for name in corpus dropped ledger; do
        [[ ! -e "$PHASE04_STAGE_DIR/$name" || -L "$PHASE04_STAGE_DIR/$name" ]] || {
            echo "ERROR: refusing to replace legacy non-symlink partial output: $name" >&2
            exit 121
        }
        relative="attempts/${SLURM_JOB_ID}/$name"
        ln -s "$relative" "$PHASE04_STAGE_DIR/.${name}.${SLURM_JOB_ID}.link"
        mv -f "$PHASE04_STAGE_DIR/.${name}.${SLURM_JOB_ID}.link" "$PHASE04_STAGE_DIR/$name"
    done
    ln -s "attempts/${SLURM_JOB_ID}/decontamination_manifest.json" \
        "$PHASE04_STAGE_DIR/.decontamination_manifest.${SLURM_JOB_ID}.link"
    mv -f "$PHASE04_STAGE_DIR/.decontamination_manifest.${SLURM_JOB_ID}.link" \
        "$PHASE04_STAGE_DIR/decontamination_manifest.json"
fi

phase04_contract_python validate-inventory \
    --manifest "$PHASE04_STAGE_DIR/decontamination_manifest.json" \
    --output "$PHASE04_STAGE_DIR/inventory_validation.json"
phase04_stage_finish \
    "$PHASE04_STAGE_DIR/decontamination_manifest.json" \
    "$PHASE04_STAGE_DIR/inventory_validation.json" \
    "$PHASE04_STAGE_DIR/final_clean_stage.txt"

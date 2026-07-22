#!/usr/bin/env bash
# Restore source text for a source/label-balanced sample of selected gap rows.
#SBATCH --job-name=bib_gap_present
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned sklearn dependency root}
TABLE_DIR=${TABLE_DIR:?set immutable candidate table root}
SOURCE_JSONL=${SOURCE_JSONL:?set immutable Struct2K source JSONL}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable presentation packet root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
PRESENTATION_PER_SOURCE_LABEL=${PRESENTATION_PER_SOURCE_LABEL:-30}
CONTEXT_LINES=${CONTEXT_LINES:-4}
MAXIMUM_DISPLAY_GAP_LINES=${MAXIMUM_DISPLAY_GAP_LINES:-16}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_gap_dataset_presentation \
    --table-dir "$TABLE_DIR" --source-jsonl "$SOURCE_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --per-source-label "$PRESENTATION_PER_SOURCE_LABEL" \
    --context-lines "$CONTEXT_LINES" \
    --maximum-display-gap-lines "$MAXIMUM_DISPLAY_GAP_LINES" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_GAP_PRESENTATION=$OUTPUT_DIR"

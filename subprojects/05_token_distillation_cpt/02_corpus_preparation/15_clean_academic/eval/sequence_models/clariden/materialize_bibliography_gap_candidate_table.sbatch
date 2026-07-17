#!/usr/bin/env bash
# Materialize the multi-regime gap candidate universe on a Clariden CPU node.
#SBATCH --job-name=bib_gap_candidates
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned sklearn dependency root}
SOURCE=${SOURCE:?set provenance-bound source JSONL}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen base table root}
POSITIONAL_TABLE_DIR=${POSITIONAL_TABLE_DIR:?set frozen positional table root}
ENTRY_OOF_DIR=${ENTRY_OOF_DIR:?set frozen P0D OOF root}
HEADING_OOF_DIR=${HEADING_OOF_DIR:?set frozen heading OOF root}
CONNECTOR_OOF_DIR=${CONNECTOR_OOF_DIR:?set frozen connector OOF root}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable candidate table root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_gap_candidate_table \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" \
    --positional-table-dir "$POSITIONAL_TABLE_DIR" --entry-oof-dir "$ENTRY_OOF_DIR" \
    --heading-oof-dir "$HEADING_OOF_DIR" --connector-oof-dir "$CONNECTOR_OOF_DIR" \
    --output-dir "$OUTPUT_DIR" --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_GAP_CANDIDATE_TABLE=$OUTPUT_DIR"

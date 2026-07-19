#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-table
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
SOURCE=${SOURCE:?set train source JSONL}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set entry feature table}
POSITIONAL_TABLE_DIR=${POSITIONAL_TABLE_DIR:?set positional feature table}
ENTRY_OOF_DIR=${ENTRY_OOF_DIR:?set P0D OOF root}
HEADING_OOF_DIR=${HEADING_OOF_DIR:?set heading OOF root}
CONNECTOR_OOF_DIR=${CONNECTOR_OOF_DIR:?set connector OOF root}
CONTINUATION_OOF_DIR=${CONTINUATION_OOF_DIR:?set continuation-head root}
SIGNAL_PROBABILITY=${SIGNAL_PROBABILITY:?set signal-TCN OOF probability}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable table root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_table \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" \
    --positional-table-dir "$POSITIONAL_TABLE_DIR" --entry-oof-dir "$ENTRY_OOF_DIR" \
    --heading-oof-dir "$HEADING_OOF_DIR" --connector-oof-dir "$CONNECTOR_OOF_DIR" \
    --continuation-oof-dir "$CONTINUATION_OOF_DIR" \
    --signal-probability "$SIGNAL_PROBABILITY" --output-dir "$OUTPUT_DIR" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_TABLE=$OUTPUT_DIR"

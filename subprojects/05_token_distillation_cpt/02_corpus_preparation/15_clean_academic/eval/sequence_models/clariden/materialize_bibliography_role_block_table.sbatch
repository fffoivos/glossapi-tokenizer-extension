#!/usr/bin/env bash
# Build fully reviewed block sequences from OOF expert probabilities.
#SBATCH --job-name=bib_role_block_table
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
SOURCE=${SOURCE:?set STRUCT-2K source}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
OVERLAY=${OVERLAY:?set combined trusted role overlay}
HEADING_OVERLAY=${HEADING_OVERLAY:?set typed heading overlay}
ENTRY_OOF_DIR=${ENTRY_OOF_DIR:?set frozen P0D result root}
HEADING_OOF_DIR=${HEADING_OOF_DIR:?set heading OOF result root}
CONNECTOR_OOF_DIR=${CONNECTOR_OOF_DIR:?set connector OOF result root}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable block table root}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_role_tables block \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" --overlay "$OVERLAY" \
    --heading-overlay "$HEADING_OVERLAY" --entry-oof-dir "$ENTRY_OOF_DIR" \
    --heading-oof-dir "$HEADING_OOF_DIR" --connector-oof-dir "$CONNECTOR_OOF_DIR" \
    --output-dir "$OUTPUT_DIR"
trap - EXIT
echo "BIBLIOGRAPHY_ROLE_BLOCK_TABLE=$OUTPUT_DIR"

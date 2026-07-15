#!/usr/bin/env bash
# Materialize either the heading or connector role-expert table on Clariden CPU.
#SBATCH --job-name=bib_role_table
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
COMMAND=${COMMAND:?set heading or connector}
SOURCE=${SOURCE:?set STRUCT-2K source}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
OVERLAY=${OVERLAY:?set combined trusted role overlay}
HEADING_OVERLAY=${HEADING_OVERLAY:?set typed heading overlay}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable table root}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ "$COMMAND" == heading || "$COMMAND" == connector ]] || exit 91
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
if [[ "$COMMAND" == heading ]]; then
  ENTRY_OOF=${ENTRY_OOF:?set frozen P0D OOF probability}
  INVENTORY_PROVENANCE=${INVENTORY_PROVENANCE:?set heading inventory provenance}
  uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
    python3 -m sequence_models.bibliography_role_tables heading \
      --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" --overlay "$OVERLAY" \
      --heading-overlay "$HEADING_OVERLAY" --entry-oof "$ENTRY_OOF" \
      --inventory-provenance "$INVENTORY_PROVENANCE" --output-dir "$OUTPUT_DIR"
else
  POSITIONAL_TABLE_DIR=${POSITIONAL_TABLE_DIR:?set positional table}
  ENTRY_OOF_DIR=${ENTRY_OOF_DIR:?set frozen P0D result root}
  HEADING_OOF_DIR=${HEADING_OOF_DIR:?set heading OOF result root}
  uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
    python3 -m sequence_models.bibliography_role_tables connector \
      --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" --overlay "$OVERLAY" \
      --heading-overlay "$HEADING_OVERLAY" --positional-table-dir "$POSITIONAL_TABLE_DIR" \
      --entry-oof-dir "$ENTRY_OOF_DIR" --heading-oof-dir "$HEADING_OOF_DIR" \
      --entry-threshold 0.25 --output-dir "$OUTPUT_DIR"
fi
trap - EXIT
echo "BIBLIOGRAPHY_ROLE_EXPERT_TABLE=$OUTPUT_DIR"

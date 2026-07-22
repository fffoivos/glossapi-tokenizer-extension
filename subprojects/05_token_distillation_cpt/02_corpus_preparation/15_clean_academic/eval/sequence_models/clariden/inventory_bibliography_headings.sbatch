#!/usr/bin/env bash
# Mine the document-wide, prediction-blinded heading inventory on Clariden CPU.
#SBATCH --job-name=bib_heading_inventory
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
SOURCE=${SOURCE:?set STRUCT-2K source}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
ENTRY_OOF=${ENTRY_OOF:?set frozen P0D OOF probability}
OVERLAY=${OVERLAY:?set combined trusted role overlay}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable inventory root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
mkdir "$OUTPUT_DIR"
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_heading_review inventory \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" --entry-oof "$ENTRY_OOF" \
    --overlay "$OVERLAY" --split train \
    --packet-out "$OUTPUT_DIR/packet.json" \
    --provenance-out "$OUTPUT_DIR/provenance.json" \
    --receipt-out "$OUTPUT_DIR/receipt.json" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_HEADING_INVENTORY=$OUTPUT_DIR"

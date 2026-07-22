#!/usr/bin/env bash
# Apply the frozen bibliography decoder to source-held-out signal-TCN scores.
#SBATCH --job-name=bib_source_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set joint silver JSONL}
TABLE_DIR=${TABLE_DIR:?set source-held-out train table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen document-OOF entry model}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set source-held-out signal TCN}
FROZEN_RECALL_BLOCK_DIR=${FROZEN_RECALL_BLOCK_DIR:?set frozen grouped-OOF decoder report}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new source-held-out evaluation directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.6.0:v1}

for receipt in "$TABLE_DIR/receipt.json" "$LINE_OOF_DIR/receipt.json" "$SIGNAL_TCN_DIR/receipt.json" "$FROZEN_RECALL_BLOCK_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 91
[[ -f "$QUALITY_DECISIONS" && ! -L "$QUALITY_DECISIONS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_signal_source_holdout_eval \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --frozen-recall-block-dir "$FROZEN_RECALL_BLOCK_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SOURCE_HOLDOUT_EVAL=$OUTPUT_DIR"

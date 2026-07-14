#!/usr/bin/env bash
# Evaluate frozen signal-TCN candidates on the existing validation table.
#SBATCH --job-name=bib_signal_val
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set joint silver JSONL}
VALIDATION_TABLE_DIR=${VALIDATION_TABLE_DIR:?set validation feature table}
VALIDATION_LINE_PROBABILITY=${VALIDATION_LINE_PROBABILITY:?set frozen validation line probability}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set frozen train-OOF signal TCN}
TRAIN_RECALL_BLOCK_DIR=${TRAIN_RECALL_BLOCK_DIR:?set completed train recall-block grid}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set validation extraction-quality decisions}
POLICY=${POLICY:?set pre-validation frozen policy}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable validation output}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in "$VALIDATION_TABLE_DIR/receipt.json" "$SIGNAL_TCN_DIR/receipt.json" "$TRAIN_RECALL_BLOCK_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
for file in "$INPUT" "$VALIDATION_LINE_PROBABILITY" "$QUALITY_DECISIONS" "$POLICY"; do
  [[ -f "$file" && ! -L "$file" ]] || exit 91
done
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_signal_validation \
  --input "$INPUT" \
  --validation-table-dir "$VALIDATION_TABLE_DIR" \
  --validation-line-probability "$VALIDATION_LINE_PROBABILITY" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --train-recall-block-dir "$TRAIN_RECALL_BLOCK_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --policy "$POLICY" \
  --output-dir "$OUTPUT_DIR" \
  --workers 16 \
  --torch-threads 16 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SIGNAL_VALIDATION=$OUTPUT_DIR"

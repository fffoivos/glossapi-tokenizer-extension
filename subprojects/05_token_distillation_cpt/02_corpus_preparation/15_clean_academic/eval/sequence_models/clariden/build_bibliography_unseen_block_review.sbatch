#!/usr/bin/env bash
# Run the frozen bibliography classifier on unseen source-matched documents.
#SBATCH --job-name=bib_unseen_review
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
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
DOCUMENTS=${DOCUMENTS:?set source-matched documents JSONL}
SELECTION_MANIFEST=${SELECTION_MANIFEST:?set source-matched selection manifest}
LINE_MODEL=${LINE_MODEL:?set frozen final D1 line model}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set frozen train-fold signal TCN directory}
RECALL_BLOCK_DIR=${RECALL_BLOCK_DIR:?set frozen recall-block grid directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable output directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for file in "$DOCUMENTS" "$SELECTION_MANIFEST" "$LINE_MODEL"; do
  [[ -f "$file" && ! -L "$file" ]] || exit 90
done
for receipt in "$SIGNAL_TCN_DIR/receipt.json" "$RECALL_BLOCK_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 91
done
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_unseen_block_review \
  --documents "$DOCUMENTS" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --line-model "$LINE_MODEL" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --recall-block-dir "$RECALL_BLOCK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --candidate-per-source 30 \
  --documents-per-source 10 \
  --workers 16 \
  --batch-size 16 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_UNSEEN_REVIEW=$OUTPUT_DIR"

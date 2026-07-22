#!/usr/bin/env bash
# Train-only B1 bias and anchored-component coherence sweep.
#SBATCH --job-name=bib_entry_coherence
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned Python dependency directory}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set train-only line OOF directory}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set train-only B0/H0 directory}
B1_OOF_DIR=${B1_OOF_DIR:?set train-only B1 directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable coherence-sweep directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR" ]] || exit 90
[[ -f "$LINE_OOF_DIR/receipt.json" && ! -L "$LINE_OOF_DIR" ]] || exit 91
[[ -f "$BLOCK_OOF_DIR/receipt.json" && ! -L "$BLOCK_OOF_DIR" ]] || exit 92
[[ -f "$B1_OOF_DIR/receipt.json" && ! -L "$B1_OOF_DIR" ]] || exit 93
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 94

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_coherence \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --b1-oof-dir "$B1_OOF_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_COHERENCE=$OUTPUT_DIR"

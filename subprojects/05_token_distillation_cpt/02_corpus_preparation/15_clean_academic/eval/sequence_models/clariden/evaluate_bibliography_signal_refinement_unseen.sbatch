#!/usr/bin/env bash
# Diagnose frozen train-selected refinements against the unseen user review.
#SBATCH --job-name=bib_refine_unseen
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
PACKET=${PACKET:?set frozen unseen-document packet}
REVIEW=${REVIEW:?set archived user review JSON}
REFINEMENT_DIR=${REFINEMENT_DIR:?set completed train-OOF refinement artifact}
OUTPUT=${OUTPUT:?set new diagnostic JSON path}
UENV=${UENV:-pytorch/v2.9.1:v2}

for file in "$PACKET" "$REVIEW" "$REFINEMENT_DIR/receipt.json"; do
  [[ -f "$file" && ! -L "$file" ]] || exit 90
done
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 91
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 92
for module in line_lr.py span_signals.py span_seq_data.py; do
  [[ -f "$CODE_ROOT/$module" && ! -L "$CODE_ROOT/$module" ]] || exit 93
done

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" \
  python3 -m sequence_models.bibliography_signal_refinement_unseen \
  --packet "$PACKET" \
  --review "$REVIEW" \
  --refinement-dir "$REFINEMENT_DIR" \
  --output "$OUTPUT"

echo "BIBLIOGRAPHY_REFINEMENT_UNSEEN=$OUTPUT"

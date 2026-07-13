#!/usr/bin/env bash
# Freeze the train-only model choice before validation is opened.
#SBATCH --job-name=bib_entry_freeze
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
LINE_OOF_DIR=${LINE_OOF_DIR:?set completed line-OOF directory}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set completed B0/H0 directory}
B1_OOF_DIR=${B1_OOF_DIR:?set completed B1 directory}
B2_OOF_DIR=${B2_OOF_DIR:?set completed/skipped B2 directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable freeze directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in "$LINE_OOF_DIR/receipt.json" "$BLOCK_OOF_DIR/receipt.json" "$B1_OOF_DIR/receipt.json" "$B2_OOF_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 91
done
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 92

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_validation freeze \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --b1-oof-dir "$B1_OOF_DIR" \
  --b2-oof-dir "$B2_OOF_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_FREEZE=$OUTPUT_DIR"

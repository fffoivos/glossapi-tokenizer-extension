#!/usr/bin/env bash
# Open retrospective validation once, after the train-only selection is frozen.
#SBATCH --job-name=bib_entry_val
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?set pinned input SHA-256}
TRAIN_TABLE_DIR=${TRAIN_TABLE_DIR:?set completed train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set completed train-only line OOF directory}
FREEZE_DIR=${FREEZE_DIR:?set completed frozen selection directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable validation directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$TRAIN_TABLE_DIR/receipt.json" && ! -L "$TRAIN_TABLE_DIR" ]] || exit 91
[[ -f "$LINE_OOF_DIR/receipt.json" && ! -L "$LINE_OOF_DIR" ]] || exit 92
[[ -f "$FREEZE_DIR/receipt.json" && ! -L "$FREEZE_DIR" ]] || exit 93
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 94

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_validation validate \
  --input "$INPUT" \
  --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
  --train-table-dir "$TRAIN_TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --frozen-config "$FREEZE_DIR/frozen_config.json" \
  --output-dir "$OUTPUT_DIR" \
  --workers 16 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_VALIDATION=$OUTPUT_DIR"

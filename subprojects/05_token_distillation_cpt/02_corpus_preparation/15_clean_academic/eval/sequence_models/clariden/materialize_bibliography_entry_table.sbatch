#!/usr/bin/env bash
# Materialize the exact-mask bibliography entry feature table on Clariden CPU.
#SBATCH --job-name=bib_entry_table
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?set pinned input SHA-256}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable feature-table directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT/sequence_models" && ! -L "$CODE_ROOT" ]] || exit 90
[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 91
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 92

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$CODE_ROOT"

uenv run "$UENV" --view=default -- python3 -m sequence_models.bibliography_entry_dataset \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --split train \
  --workers "$SLURM_CPUS_PER_TASK" \
  --n-folds 5 \
  --fold-seed bibliography-entry-folds-v1 \
  --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_TABLE=$OUTPUT_DIR"

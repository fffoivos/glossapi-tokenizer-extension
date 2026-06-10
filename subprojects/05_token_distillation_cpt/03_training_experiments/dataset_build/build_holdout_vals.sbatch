#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_holdout_vals
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# Build 3 held-out val sets (~0.5B tok each) + val_holdout_ids.parquet (for train exclusion).
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
DB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build
STAGE=$SC/cpt_corpus/cpt_2arm_13b
SELECTED=$SC/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python
mkdir -p "$STAGE"
test -x "$PY"
"$PY" -u "$DB/build_holdout_vals.py" --selected "$SELECTED" --output-dir "$STAGE"
echo "DONE $(date -u)"
ls -lah "$STAGE"/val_*.jsonl "$STAGE/val_holdout_ids.parquet"

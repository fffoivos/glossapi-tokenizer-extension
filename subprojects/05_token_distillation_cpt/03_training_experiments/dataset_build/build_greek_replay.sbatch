#!/usr/bin/env bash
#SBATCH --job-name=greek_replay_build
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# CPU-only: nanochat ∩ apertus_overlap_drop -> greek_replay.parquet (5% replay source).
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
DB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build
DROP=$SC/cpt_corpus/dedup_audit/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python
mkdir -p "$SC/cpt_corpus/greek_replay" /capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b
test -f "$DROP"
test -x "$PY"
"$PY" -u "$DB/build_greek_replay.py" \
  --drop-list "$DROP" \
  --nanochat-glob "$SC/cpt_corpus/nanochat/data/*.parquet" \
  --output "$SC/cpt_corpus/greek_replay/greek_replay.parquet"
echo "DONE $(date -u)"

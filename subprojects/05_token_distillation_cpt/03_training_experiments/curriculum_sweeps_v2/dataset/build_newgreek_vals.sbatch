#!/usr/bin/env bash
#SBATCH --job-name=v2_newgreek_vals
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Build the NEW-GREEK held-outs with the deployed builder, but on a CPU Slurm
# node so pyarrow does not run on the Clariden login node. MUST run before
# mix_phase_binaries (hplt/glossapi recipes drop val_holdout_ids.parquet) and
# before tokenize_newgreek_vals.
set -euo pipefail
source "$(dirname "$0")/../paths.env"
mkdir -p "$STAGE"
test -x "$PY"
test -f "$THIS/dataset_build/build_holdout_vals.py"
"$PY" -u "$THIS/dataset_build/build_holdout_vals.py" --selected "$SELECTED" --output-dir "$STAGE" --char-budget 2000000000
echo "new-Greek vals DONE $(date -u)"
ls -lah "$STAGE"/val_*.jsonl "$STAGE/val_holdout_ids.parquet" 2>&1 | head -20

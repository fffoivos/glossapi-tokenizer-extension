#!/usr/bin/env bash
#SBATCH --job-name=v2_codeparrot_snap
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Mint a guaranteed-disjoint `code` forgetting set with a stable sha id. Run
# this BEFORE build_forgetting_vals.sbatch, which merges the code ids.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
mkdir -p "$STAGE"; test -x "$PY"
"$PY" -u "$V2/dataset/snapshot_codeparrot_heldout.py" --output-dir "$STAGE" --char-budget 2000000000 --skip-docs 2000000
echo "codeparrot snapshot DONE $(date -u)"

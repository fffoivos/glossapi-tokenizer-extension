#!/usr/bin/env bash
#SBATCH --job-name=v2_forget_vals
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Build the OLD-data forgetting held-outs. MUST run before mix_phase_binaries (replay_only drops
# these ids) and before tokenize_forgetting_vals. Run
# snapshot_codeparrot_heldout.sbatch FIRST so the `code` set is included.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
mkdir -p "$STAGE"
test -x "$PY"
"$PY" -u "$V2/dataset/build_forgetting_vals.py" --output-dir "$STAGE" --char-budget 2000000000
echo "forgetting-vals DONE $(date -u)"
ls -lah "$STAGE"/val_forget_*.jsonl "$STAGE/forget_holdout_ids.parquet" 2>&1 | head -20

#!/usr/bin/env bash
#SBATCH --job-name=v2_forget_vals
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Build OLD-data validation held-outs. MUST run before mix_phase_binaries
# (replay_only drops these ids) and before tokenize_forgetting_vals. Uses the
# absolute 2B-char target unless that exceeds MAX_VAL_FRACTION of the finite
# pool; default MAX_VAL_FRACTION=0.25.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
mkdir -p "$STAGE"
check_build_py
run_build_py -u "$V2/dataset/build_forgetting_vals.py" \
  --output-dir "$STAGE" \
  --char-budget "${FORGET_CHAR_BUDGET:-2000000000}" \
  --max-val-fraction "${MAX_VAL_FRACTION:-0.25}"
echo "forgetting-vals DONE $(date -u)"
ls -lah "$STAGE"/val_forget_*.jsonl "$STAGE/forget_holdout_ids.parquet" "$STAGE/forgetting_val_manifest.json" 2>&1 | head -20

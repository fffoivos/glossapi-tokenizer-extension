#!/usr/bin/env bash
#SBATCH --job-name=v2_mix_binaries
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-2 : 0=hplt_only 1=glossapi_only 2=replay_only. Build the 3 phase binaries ONCE;
# replay% is later a Megatron blend weight, so no per-replay rebuild.
# Mirrors mix_13b.sbatch (env SELECTED, PYTHONPATH=$CB:$SC/repo, mix_builder.py via the uenv venv).
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"

BINS=(hplt_only glossapi_only replay_only)
B=${BINS[${SLURM_ARRAY_TASK_ID:-0}]}
# token budgets: HPLT/OpenArchives are sized to cover the fixed-total 13.5B
# replay sweep down to R=0.15 while preserving the 70/30 phase split. replay is
# sized to cover the largest blend draw (R=0.35 of a 13.5B run = ~4.7B) plus
# headroom.
declare -A TGT=( [hplt_only]=8500000000 [glossapi_only]=3700000000 [replay_only]=5000000000 )

export SELECTED="$SELECTED"
export PYTHONPATH="$CB:$SC/repo:${PYTHONPATH:-}"
mkdir -p "$STAGE"
# both holdout-id parquets MUST pre-exist (the recipes' drop wiring reads them)
test -f "$STAGE/val_holdout_ids.parquet"
test -f "$STAGE/forget_holdout_ids.parquet"
test -x "$PY"
test -f "$V2/dataset/recipes/$B.json"

"$PY" -u "$MIX_BUILDER" \
  --recipe "$V2/dataset/recipes/$B.json" \
  --target-tokens "${TGT[$B]}" \
  --tokenizer "$EXT_TOK" \
  --output "$STAGE/${B}.jsonl" \
  --seed 20260611
echo "mix DONE $B $(date -u)"
ls -lah "$STAGE/${B}.jsonl"

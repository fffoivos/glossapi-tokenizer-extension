#!/usr/bin/env bash
#SBATCH --job-name=mix_13b
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%A_%a.err
# CPU-only array mix: SELECTED + replay + greek_replay -> bulk_mix_part_NN.jsonl
# Run as: sbatch --array=0-7%2 --dependency=afterok:<build_holdout_vals job> mix_13b.sbatch
# (depends on V: the recipe excludes val_holdout_ids.parquet, built by V)
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
REPO=$SC/repo/glossapi-tokenizer-extension
CB=$SC/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build  # legacy mirror has corpus_build
DB=$REPO/subprojects/05_token_distillation_cpt/03_training_experiments/dataset_build
RECIPE=$DB/bulk_13b.json
TOKENIZER=$SC/tokenizers/apertus_greek_modern_only_148480
export SELECTED=$SC/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
STAGE=$SC/cpt_corpus/cpt_2arm_13b
BADNESS_REPORT=$STAGE/selected_badness_validation.json
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python
mkdir -p "$STAGE"
TOTAL_TOKENS=13500000000
NSHARD="${SLURM_ARRAY_TASK_COUNT:-8}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PER=$(( TOTAL_TOKENS / NSHARD ))
SEED=$(( 20260609 + IDX ))
OUT="$STAGE/bulk_mix_part_$(printf '%02d' "$IDX").jsonl"
test -f "$SELECTED"; test -f "$RECIPE"; test -f "$SC/cpt_corpus/greek_replay/greek_replay.parquet"
test -f "$STAGE/val_holdout_ids.parquet"   # built by build_holdout_vals (dep V); recipe drop-keys
test -x "$PY"
"$PY" -u "$DB/validate_selected_badness.py" --selected "$SELECTED" --require-current-report "$BADNESS_REPORT"
echo "mix shard $IDX/$NSHARD per-shard-tokens=$PER seed=$SEED -> $OUT"
export SELECTED=$SELECTED
export PYTHONPATH=$CB:$SC/repo:${PYTHONPATH:-}
"$PY" -u "$CB/mix_builder.py" \
  --recipe "$RECIPE" \
  --target-tokens "$PER" \
  --tokenizer "$TOKENIZER" \
  --output "$OUT" \
  --seed "$SEED" \
  --source-shard-index "$IDX" \
  --source-shard-count "$NSHARD"
echo "DONE shard $IDX $(date -u)"

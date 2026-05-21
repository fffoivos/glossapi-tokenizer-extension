#!/usr/bin/env bash
#SBATCH --job-name=mix_builder_full
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_full_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_full_%j.err
#
# Full bulk-mix build for the 2B-token init bakeoff.
#
# Inputs:
#   $SELECTED from prepare_greek_pool.sh:
#     /iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
#   recipes/bulk.json:
#     70/24/4/2 Greek/replay/code/math
#
# Output:
#   /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl
#
# Compute justification:
#   * Parallelism: mix_builder.py is deterministic single-writer streaming.
#     PyArrow readers use internal threads; tokenizer.encode is called
#     row-at-a-time for deterministic budget tracking.
#   * CPU saturation: one socket (72 CPUs) is enough because the writer and
#     tokenizer loop are the bottleneck. Wider allocation mostly idles.
#   * Memory: streaming; peak is row groups + tokenizer + source iterator
#     buffers. 200G is conservative.
#   * Known gap: batching tokenizer.encode would be faster but would change the
#     deterministic streaming loop. Keep the production path identical to the
#     smoke job.

set -euo pipefail

STAGE_ROOT=/iopsstor/scratch/cscs/fffoivos/cpt_corpus
SELECTED=$STAGE_ROOT/cpt/selected_after_apertus_and_internal_dedup.parquet
REPO_DIR=/iopsstor/scratch/cscs/fffoivos/repo
RECIPE=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json
TOKENIZER=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480
OUTPUT=$STAGE_ROOT/bulk_mix.jsonl

if [ ! -f "$SELECTED" ]; then
    echo "ERROR: $SELECTED missing — run prepare_greek_pool first" >&2
    exit 2
fi
if [ ! -d "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER missing" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT")" /capstor/scratch/cscs/fffoivos/runs/preprocess

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:$REPO_DIR
export SELECTED=$SELECTED

python3 $REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py \
    --recipe $RECIPE \
    --target-tokens 7000000000 \
    --tokenizer $TOKENIZER \
    --output $OUTPUT \
    --seed 20260520
INNER

echo "=== full mix complete ==="
ls -lh "$OUTPUT"
wc -l "$OUTPUT"

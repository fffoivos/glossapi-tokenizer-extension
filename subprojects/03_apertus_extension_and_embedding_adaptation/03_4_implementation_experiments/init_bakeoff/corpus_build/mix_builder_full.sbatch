#!/usr/bin/env bash
#SBATCH --job-name=mix_builder_full
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_full_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_full_%j.err
#
# Full bulk-mix build for the 2B-token init bakeoff.
#
# Can be run either as one 7B-token job or as a Slurm array of smaller shards:
#
#   sbatch --array=0-6%2 --export=ALL,TARGET_TOKENS=1000000000 mix_builder_full.sbatch
#
# Array mode writes bulk_mix_part_XX.jsonl files, each with its own seed.
# In array mode, each task also receives a source-row shard index so the
# per-source rows are disjoint across shard files. Concatenate them with
# concat_bulk_mix.sbatch before preprocess_data.sbatch.
#
# Inputs:
#   $SELECTED from prepare_greek_pool.sh:
#     /iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
#   recipes/bulk.json:
#     70/24/4/2 Greek/replay/code/math for the 2026-05-21 CSCS run
#
# Output:
#   /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl
#
# Compute justification:
#   * Parallelism: mix_builder.py is deterministic single-writer streaming.
#     PyArrow readers use internal threads; tokenizer.encode is called
#     row-at-a-time for deterministic budget tracking.
#   * CPU saturation: one shard uses roughly one hot Python core. 16 CPUs leave
#     PyArrow/runtime headroom without pretending this is a full-node CPU job.
#   * Memory: streaming; peak is row groups + tokenizer + source iterator
#     buffers. 64G is conservative for this bounded streaming path.
#   * Known gap: batching tokenizer.encode would be faster but would change the
#     deterministic streaming loop. Keep the production path identical to the
#     smoke job. After the token-fair fix, 50M-token smoke measured ~160k tok/s.
#     The production run is therefore sharded, and the fastest safe mode is to
#     run two 1B-token shards concurrently on xfer. Source-row sharding prevents
#     duplicate per-source prefixes, but it also scans skipped rows; request the
#     xfer 24 h ceiling so a correct run is not lost near the end.
#     One shard uses roughly one hot Python core; additional cores inside a
#     single shard do not help without a batching/parallel-token-count rewrite.
#     Run on xfer; normal/debug/low are GPU-node partitions on Clariden.

set -euo pipefail

STAGE_ROOT=/iopsstor/scratch/cscs/fffoivos/cpt_corpus
SELECTED=$STAGE_ROOT/cpt/selected_after_apertus_and_internal_dedup.parquet
REPO_DIR=/iopsstor/scratch/cscs/fffoivos/repo
INIT_BAKEOFF_DIR=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff
source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "mix_builder_full"
RECIPE=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json
TOKENIZER=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480
TARGET_TOKENS="${TARGET_TOKENS:-7000000000}"
SEED="${SEED:-20260520}"
SHARD_PREFIX="${SHARD_PREFIX:-$STAGE_ROOT/bulk_mix_part_}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD_ID="$(printf "%02d" "$SLURM_ARRAY_TASK_ID")"
    OUTPUT="${OUTPUT:-${SHARD_PREFIX}${SHARD_ID}.jsonl}"
    SEED="$(( SEED + SLURM_ARRAY_TASK_ID ))"
    SOURCE_SHARD_INDEX="${SOURCE_SHARD_INDEX:-$SLURM_ARRAY_TASK_ID}"
    SOURCE_SHARD_COUNT="${SOURCE_SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-7}}"
else
    OUTPUT="${OUTPUT:-$STAGE_ROOT/bulk_mix.jsonl}"
    SOURCE_SHARD_INDEX="${SOURCE_SHARD_INDEX:-0}"
    SOURCE_SHARD_COUNT="${SOURCE_SHARD_COUNT:-1}"
fi

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

python3 -u $REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py \
    --recipe $RECIPE \
    --target-tokens $TARGET_TOKENS \
    --tokenizer $TOKENIZER \
    --output $OUTPUT \
    --seed $SEED \
    --source-shard-index $SOURCE_SHARD_INDEX \
    --source-shard-count $SOURCE_SHARD_COUNT
INNER

echo "=== full mix complete ==="
ls -lh "$OUTPUT"
wc -l "$OUTPUT"

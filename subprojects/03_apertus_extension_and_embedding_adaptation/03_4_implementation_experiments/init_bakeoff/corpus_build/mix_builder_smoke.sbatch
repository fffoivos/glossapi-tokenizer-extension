#!/usr/bin/env bash
#SBATCH --job-name=mix_builder_smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_smoke_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_builder_smoke_%j.err
#
# PF3 smoke: validate the local_parquet code path with 50M-token target.
# Reads from $SELECTED (output of prepare_greek_pool.sh) + replay parquets.
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#
# 1. Parallelism strategy.
#    mix_builder.py is a single Python writer (sequential by design for
#    bakeoff determinism — see its docstring "Determinism"). The hot
#    paths are:
#      (a) pyarrow parquet reader (auto-multithreaded, uses up to 64 io threads)
#      (b) HuggingFace fast tokenizer.encode (single text/single thread per call,
#          but GIL-free and called serially in the streaming loop)
#      (c) Python row dispatch (GIL-held; fixed overhead).
#    Real throughput at 50 M tokens is ~10 MB/s encode-limited; 50 M / 10 MB/s
#    on Greek ≈ 30-60 s of encode work, swamped by parquet IO + JSONL write.
#
# 2. CPU saturation.
#    --cpus-per-task=72 = one socket. Going wider doesn't help: the writer
#    is single-threaded and the readers are IO-bound. 72 cores leaves the
#    other 3 sockets for any concurrent jobs we'd schedule alongside.
#
# 3. Memory ceiling.
#    Streaming, so peak RAM is ~one parquet row group per source + the
#    tokenizer (~50 MB) + the per-source replay parquets (largest single
#    shard ≈ 1-2 GB). 200 G is generous headroom; could shrink to 64 G if
#    we wanted to share the node.
#
# 4. Known efficiency gaps.
#    Tokenizer is called row-at-a-time. `encode_batch` would give 10-20×
#    speedup but would change the determinism contract — the smoke test
#    intentionally exercises the production code path, not the faster one.

set -euo pipefail

STAGE_ROOT=/iopsstor/scratch/cscs/fffoivos/cpt_corpus
SELECTED=$STAGE_ROOT/cpt/selected_after_apertus_and_internal_dedup.parquet
REPO_DIR=/iopsstor/scratch/cscs/fffoivos/repo
RECIPE=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json
TOKENIZER=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480
OUTPUT=/capstor/scratch/cscs/fffoivos/runs/preprocess/mix_smoke_50M.jsonl

if [ ! -f "$SELECTED" ]; then
    echo "ERROR: $SELECTED missing — run prepare_greek_pool first" >&2
    exit 2
fi
if [ ! -d "$TOKENIZER" ]; then
    echo "ERROR: $TOKENIZER missing" >&2
    exit 2
fi

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:$REPO_DIR
export SELECTED=$SELECTED

python3 -u $REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py \
    --recipe $RECIPE \
    --target-tokens 50000000 \
    --tokenizer $TOKENIZER \
    --output $OUTPUT \
    --seed 20260521 \
    --progress-every-tokens 5000000
INNER

echo "=== smoke complete ==="
ls -la $OUTPUT
wc -l $OUTPUT
head -c 400 $OUTPUT
echo

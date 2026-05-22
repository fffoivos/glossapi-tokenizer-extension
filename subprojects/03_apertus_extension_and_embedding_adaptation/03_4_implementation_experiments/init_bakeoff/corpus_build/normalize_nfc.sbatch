#!/usr/bin/env bash
#SBATCH --job-name=normalize_nfc
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/normalize_nfc_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/normalize_nfc_%j.err
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#
# 1. Parallelism strategy.
#    verify_and_normalize_nfc.py uses Python multiprocessing — one worker
#    per parquet shard. Each worker reads, NFC-normalizes string columns
#    (idempotent: unicodedata.normalize("NFC", s)), and rewrites in place.
#    Shards are independent; no shared state past the worker pool.
#
# 2. CPU saturation.
#    --cpus-per-task=64 is passed to normalize_nfc.sh via WORKERS. Corpus has
#    ~300 shards, so this still keeps enough parallel file workers busy while
#    staying on xfer instead of a GPU-node partition.
#    Normalization is memcpy + Unicode-table-lookup, so it benefits from
#    high parallelism even though each shard's work is IO-light.
#
# 3. Memory ceiling.
#    Per-worker peak ≈ one decompressed parquet shard (≤ 2 GB for the
#    largest Greek nanochat shards). 288 × 2 GB = 576 GB worst case, but
#    in practice only ~64 simultaneously are mid-process (the rest are
#    waiting on IO); --mem=400G is a comfortable headroom margin.
#
# 4. Known efficiency gaps.
#    Within a single shard, NFC normalization is row-serial (no
#    intra-shard parallelism). If shard count drops below 288 in a
#    future corpus version, we'd want to re-shard upstream rather than
#    spawn many idle workers.

set -euo pipefail
mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess

REPO_DIR=/iopsstor/scratch/cscs/fffoivos/repo
SCRIPT_DIR=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build
source "$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "normalize_nfc"

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:$REPO_DIR
export WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
bash $SCRIPT_DIR/normalize_nfc.sh
INNER

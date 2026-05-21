#!/usr/bin/env bash
#SBATCH --job-name=prepare_greek_pool
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/prepare_greek_pool_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/prepare_greek_pool_%j.err
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#
# 1. Parallelism strategy.
#    Hot path is duckdb COPY queries inside materialize_*_mix_input
#    (pipeline.py). duckdb auto-uses all visible threads for analytical
#    queries (default `threads = num_visible_cpu`). Greek nanochat is 129 GB
#    of parquet → duckdb scans + dedup is the dominant cost; near-linear
#    speedup to ~64-128 cores, diminishing past that for this query shape.
#
# 2. CPU saturation.
#    --cpus-per-task=288 = full normal-partition node (4 sockets × 72 cores,
#    ThreadsPerCore=1, per `scontrol show node`). Slurm's OverSubscribe=EXCLUSIVE
#    means we own the node regardless of --cpus-per-task, but duckdb's
#    auto-detect reads the cgroup CPU mask — undersizing cgroup pins duckdb
#    to a subset of cores even though the rest are idle. Job 2334358 used
#    --cpus-per-task=64 and observed ~22% node CPU utilization.
#
# 3. Memory ceiling.
#    Node has 870 G RealMemory. --mem=800G leaves ~70 G for OS + slurmd.
#    Prior attempt at --mem=400G (job 2334358) hit duckdb OutOfMemoryException
#    "failed to pin block of size 256 KiB (305.1 GiB used)" in the apertus-drop
#    COPY query (pipeline.py:2486). duckdb auto-sets memory_limit to ~80% of
#    visible memory, so --mem=800G → duckdb memory_limit ≈ 640 G, comfortably
#    above the 305 GiB peak with room for the dedup-replay COPY that follows.
#
# 4. Known efficiency gaps.
#    The dedup-metadata-root scan reads 4 metadata parquets sequentially via
#    duckdb's parallel scanner — already saturating I/O. The Python wrapper
#    (cli.py → pipeline.materialize_streaming_mix_selected_input) is a thin
#    coordinator; no GIL-held hot path inside it.
#
# CD to scratch first — duckdb writes .tmp/duckdb_temp_storage_*.tmp relative
# to CWD and the home filesystem has a tiny quota that overflows on a 129 GB
# corpus. iopsstor scratch has ~300 TB free.
#
# Retry history:
#   2334246 (xfer)     — partition in maintenance till 2026-06-11
#   2334267 (400G/64)  — duckdb temp on home quota, exited "Disk quota exceeded"
#   2334358 (400G/64)  — duckdb OOM at 305 GiB during apertus-drop COPY
#   2334476 (800G/64)  — current

set -euo pipefail
mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess
WORK_TMP=/iopsstor/scratch/cscs/fffoivos/tmp/prepare_greek_pool_$SLURM_JOB_ID
mkdir -p "$WORK_TMP"
cd "$WORK_TMP"

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:/iopsstor/scratch/cscs/fffoivos/repo
export REPO_ROOT=/iopsstor/scratch/cscs/fffoivos/repo
export TMPDIR=$WORK_TMP
cd $WORK_TMP
bash /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/prepare_greek_pool.sh
INNER

echo "=== cleanup tmp ==="
rm -rf "$WORK_TMP"

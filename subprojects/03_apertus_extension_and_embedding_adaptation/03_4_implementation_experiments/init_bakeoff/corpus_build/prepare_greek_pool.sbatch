#!/usr/bin/env bash
#SBATCH --job-name=prepare_greek_pool
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
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
#    xfer is the only visible non-GPU Clariden partition. Keep this CPU-only
#    even if it is slower than the old full GPU-node allocation.
#
# 3. Memory ceiling.
#    xfer nodes expose 250 GB RAM. DuckDB is capped below the Slurm allocation
#    and spills to scratch via DUCKDB_TEMP_DIRECTORY.
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
#   2334246 (xfer)     — partition was in maintenance at the time
#   2334267 (400G/64)  — duckdb temp on home quota, exited "Disk quota exceeded"
#   2334358 (400G/64)  — duckdb OOM at 305 GiB during apertus-drop COPY
#   2334476 (800G/64)  — current

set -euo pipefail
mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess
REPO_DIR=/iopsstor/scratch/cscs/fffoivos/repo
INIT_BAKEOFF_DIR=$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff
source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "prepare_greek_pool"
WORK_TMP=/iopsstor/scratch/cscs/fffoivos/tmp/prepare_greek_pool_$SLURM_JOB_ID
mkdir -p "$WORK_TMP"
cd "$WORK_TMP"

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:/iopsstor/scratch/cscs/fffoivos/repo
export REPO_ROOT=/iopsstor/scratch/cscs/fffoivos/repo
export TMPDIR=$WORK_TMP
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-180GB}"
export DUCKDB_TEMP_DIRECTORY="${DUCKDB_TEMP_DIRECTORY:-$WORK_TMP/duckdb_tmp}"
export DUCKDB_THREADS="${DUCKDB_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
cd $WORK_TMP
bash /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/prepare_greek_pool.sh
INNER

echo "=== cleanup tmp ==="
rm -rf "$WORK_TMP"

#!/usr/bin/env bash
#SBATCH --job-name=cache_native_greek
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
CACHE_DIR="${CACHE_DIR:-/iopsstor/scratch/cscs/fffoivos/native_greek_eval/cache}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

mkdir -p "$CACHE_DIR" /capstor/scratch/cscs/fffoivos/runs/eval

echo "=== cache_native_greek_benchmarks.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "REPO_ROOT: $REPO_ROOT"
echo "CACHE_DIR: $CACHE_DIR"
echo

export HF_HOME="${HF_HOME:-$CACHE_DIR/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_DIR/hf_datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export TMPDIR="${TMPDIR:-$CACHE_DIR/tmp}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

uv run --isolated \
  --with 'numpy<2' \
  --with scipy \
  --with datasets \
  --with pandas \
  --with pyarrow \
  --with huggingface_hub \
  --with fsspec \
  --python "$PYTHON_VERSION" \
  "$SCRIPT_DIR/cache_native_greek_benchmarks.py" \
  --registry "$SCRIPT_DIR/native_greek_benchmark_registry.json" \
  --repo-root "$REPO_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --include-diagnostic \
  --allow-blocked || status=$?

status="${status:-0}"
echo
echo "cache status: $status"
echo "manifest: $CACHE_DIR/native_greek_benchmark_cache_manifest.json"
date -u
exit "$status"

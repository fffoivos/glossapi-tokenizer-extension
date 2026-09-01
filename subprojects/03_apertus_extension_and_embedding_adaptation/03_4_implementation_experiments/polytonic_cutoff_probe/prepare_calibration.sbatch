#!/usr/bin/env bash
#SBATCH --job-name=poly_cut_cal_data
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/%x-%j.err

set -euo pipefail
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizer_finalization/20260729T094000Z-poly512-1024}"
PYTHON_BIN="${PYTHON_BIN:-$RUN_ROOT/python_envs/coverage_py311/bin/python}"
SCRIPT="$RUN_ROOT/code/polytonic_cutoff_probe/prepare_calibration_jsonl.py"

test -x "$PYTHON_BIN"
test -f "$SCRIPT"
"$PYTHON_BIN" "$SCRIPT" \
  --ancient-train-parquet "$RUN_ROOT/assets/fineweb2/grc_train_000_00000.parquet" \
  --modern-test-parquet "$RUN_ROOT/assets/fineweb2/ell_test_000_00000.parquet" \
  --ancient-eval-jsonl "$RUN_ROOT/probe_data/ancient_test.jsonl" \
  --modern-eval-jsonl "$RUN_ROOT/probe_data/modern_test.jsonl" \
  --output-jsonl "$RUN_ROOT/probe_data/output_calibration.jsonl" \
  --manifest "$RUN_ROOT/probe_data/output_calibration_manifest.json" \
  --docs-per-register 1000 \
  --modern-skip-valid-docs 1000 \
  --max-chars-per-doc 3000

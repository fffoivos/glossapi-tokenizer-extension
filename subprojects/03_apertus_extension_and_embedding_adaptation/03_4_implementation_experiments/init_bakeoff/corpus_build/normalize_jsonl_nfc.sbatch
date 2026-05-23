#!/usr/bin/env bash
#SBATCH --job-name=normalize_jsonl_nfc
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/normalize_jsonl_nfc_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/normalize_jsonl_nfc_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/fffoivos/repo}"
INIT_BAKEOFF_DIR="$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff"
SCRIPT_DIR="$INIT_BAKEOFF_DIR/corpus_build"
source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "normalize_jsonl_nfc"

INPUT_JSONL="${INPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl}"
MANIFEST="${MANIFEST:-${OUTPUT_JSONL%.jsonl}.manifest.json}"
TEXT_KEY="${TEXT_KEY:-text}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess "$(dirname "$OUTPUT_JSONL")"

echo "=== normalize_jsonl_nfc ==="
date -u
echo "host:        $(hostname)"
echo "INPUT_JSONL: $INPUT_JSONL"
echo "OUTPUT_JSONL: $OUTPUT_JSONL"
echo "MANIFEST:    $MANIFEST"
echo "TEXT_KEY:    $TEXT_KEY"
echo "OVERWRITE:   $OVERWRITE"
echo

args=(
  --input-jsonl "$INPUT_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --manifest "$MANIFEST"
  --text-key "$TEXT_KEY"
)
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi

python3 -u "$SCRIPT_DIR/normalize_jsonl_nfc.py" "${args[@]}"

echo
echo "=== normalized JSONL ==="
ls -lh "$OUTPUT_JSONL" "$MANIFEST"
date -u

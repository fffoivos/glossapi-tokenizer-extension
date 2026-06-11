#!/usr/bin/env bash
#SBATCH --job-name=v2_tok_forget
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-5 : tokenize each forgetting val set x2 (base 131072 + ext 148480) into the v2 megatron dir.
# These become extra-valid binaries val_forget_<name>_<base|ext>_text_document. Mirrors tokenize_vals.sbatch.
# A set whose jsonl is absent (e.g. de/ru/zh not yet built) is skipped (idempotent).
set -euo pipefail
source "$(dirname "$0")/../paths.env"
cd "$MEG"; mkdir -p "$MEGOUT"; test -x "$PY"

VALS=(english de ru zh code old_greek)
V=${VALS[${SLURM_ARRAY_TASK_ID:-0}]}
if [ ! -s "$STAGE/val_forget_$V.jsonl" ]; then
  echo "[SKIP] val_forget_$V.jsonl absent — build it first (de/ru/zh need extra shards; code needs the snapshot)"; exit 0
fi
for pair in "base $BASE_TOK" "ext $EXT_TOK"; do
  set -- $pair; TAG=$1; TOK=$2
  rm -f "$MEGOUT/val_forget_${V}_${TAG}_text_document.bin" "$MEGOUT/val_forget_${V}_${TAG}_text_document.idx"
  "$PY" tools/preprocess_data.py --input "$STAGE/val_forget_$V.jsonl" \
    --output-prefix "$MEGOUT/val_forget_${V}_${TAG}" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOK" \
    --append-eod --workers 64 --json-keys text
  test -s "$MEGOUT/val_forget_${V}_${TAG}_text_document.bin"
  test -s "$MEGOUT/val_forget_${V}_${TAG}_text_document.idx"
done
echo "DONE val_forget=$V $(date -u)"

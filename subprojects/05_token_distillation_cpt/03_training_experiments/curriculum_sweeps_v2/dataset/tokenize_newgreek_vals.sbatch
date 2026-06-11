#!/usr/bin/env bash
#SBATCH --job-name=v2_tok_newgreek
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%A_%a.err
# array 0-2 : tokenize each NEW-GREEK adaptation val set x2 (base 131072 + ext 148480) into the v2
# megatron dir. These become extra-valid binaries val_<name>_<base|ext>_text_document (NO _forget_),
# matching UPSTREAM EDIT-1's new-Greek naming. The deployed tokenize_vals.sbatch CANNOT be reused: it
# hardcodes STAGE=cpt_2arm_13b + partition=xfer (in maintenance), so its --export STAGE is clobbered and
# it would tokenize the PILOT's val jsonls, not build_holdout_vals.py's curriculum_v2 output. This is the
# v2 equivalent, reading $STAGE/val_$V.jsonl (build_holdout_vals.py --output-dir $STAGE) on partition=normal.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"
cd "$MEG"; mkdir -p "$MEGOUT"; check_build_py

VALS=(hplt openarchives greek_phd)
V=${VALS[${SLURM_ARRAY_TASK_ID:-0}]}
if [ ! -s "$STAGE/val_$V.jsonl" ]; then
  echo "[SKIP] val_$V.jsonl absent — run build_holdout_vals.py --output-dir \$STAGE first"; exit 0
fi
for pair in "base $BASE_TOK" "ext $EXT_TOK"; do
  set -- $pair; TAG=$1; TOK=$2
  rm -f "$MEGOUT/val_${V}_${TAG}_text_document.bin" "$MEGOUT/val_${V}_${TAG}_text_document.idx"
  run_build_py tools/preprocess_data.py --input "$STAGE/val_$V.jsonl" \
    --output-prefix "$MEGOUT/val_${V}_${TAG}" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOK" \
    --append-eod --workers 64 --json-keys text
  test -s "$MEGOUT/val_${V}_${TAG}_text_document.bin"
  test -s "$MEGOUT/val_${V}_${TAG}_text_document.idx"
done
echo "DONE val_newgreek=$V $(date -u)"

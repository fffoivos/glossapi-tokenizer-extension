#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_tok_vals
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%A_%a.err
# array 0-2 over the 3 val sets; tokenize each with BOTH tokenizers (base + ext).
# Megatron preprocess loads the tokenizer via AutoTokenizer -> needs the
# Apertus-aware transformers in the xfer-native venv (see HANDOFF §3).
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
MEG=$SC/code/training/Megatron-LM-Swiss-AI
STAGE=$SC/cpt_corpus/cpt_2arm_13b
BASE_TOK=$SC/models/apertus-8b-2509
EXT_TOK=$SC/tokenizers/apertus_greek_modern_only_148480
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python
TOKENIZE_BASE="${TOKENIZE_BASE:-1}"
TOKENIZE_EXT="${TOKENIZE_EXT:-1}"
VALS=(hplt openarchives greek_phd)
V="${VALS[${SLURM_ARRAY_TASK_ID:-0}]}"
mkdir -p "$STAGE/megatron"
test -s "$STAGE/val_$V.jsonl"
test -x "$PY"
cd "$MEG"
if [ "$TOKENIZE_BASE" = "1" ]; then
rm -f "$STAGE/megatron/val_${V}_base_text_document.bin" "$STAGE/megatron/val_${V}_base_text_document.idx"
"$PY" tools/preprocess_data.py --input "$STAGE/val_$V.jsonl" \
  --output-prefix "$STAGE/megatron/val_${V}_base" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$BASE_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "$STAGE/megatron/val_${V}_base_text_document.bin"
test -s "$STAGE/megatron/val_${V}_base_text_document.idx"
fi
if [ "$TOKENIZE_EXT" = "1" ]; then
rm -f "$STAGE/megatron/val_${V}_ext_text_document.bin" "$STAGE/megatron/val_${V}_ext_text_document.idx"
"$PY" tools/preprocess_data.py --input "$STAGE/val_$V.jsonl" \
  --output-prefix "$STAGE/megatron/val_${V}_ext" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$EXT_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "$STAGE/megatron/val_${V}_ext_text_document.bin"
test -s "$STAGE/megatron/val_${V}_ext_text_document.idx"
fi
echo "DONE val=$V $(date -u)"
ls -lah "$STAGE/megatron/val_${V}_"*text_document* 2>&1 | head

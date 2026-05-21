#!/usr/bin/env bash
#SBATCH --job-name=preprocess_data
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/%x-%j.err
#
# Convert mix_builder JSONL → Megatron-LM-Swiss-AI binary indexed dataset.
# Wraps Megatron-LM's tools/preprocess_data.py — the canonical, well-established
# binary-packing step shared by every Megatron-LM training run.
#
# Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py
# Apertus fork uses the same tool with the same CLI surface.
#
# NOTE on fidelity: Apertus production actually used a DataTrove-based
# preprocessor at `swiss-ai/pretrain-data/examples/tokenize_megatron/preprocess_megatron.py`
# (not Megatron's stock `tools/preprocess_data.py`). Both produce byte-identical
# Megatron binary indexed datasets given the same tokenizer + input — the
# DataTrove wrapper just adds streaming + sharding ergonomics. We use the
# stock Megatron tool here for simplicity; switch to swiss-ai/pretrain-data
# if Apertus-exact preprocessing semantics matter (they don't for our
# bakeoff — see TRAINING_RECIPE.md §1).
#
# Inputs (env vars or defaults):
#   INPUT_JSONL    — output of corpus_build/mix_builder.py
#   OUTPUT_PREFIX  — Megatron output prefix (will produce <prefix>.bin + <prefix>.idx)
#   TOKENIZER_DIR  — HF-format tokenizer dir
#                      For Vanilla bakeoff data: $BASE_TOKENIZER_DIR (131,072 base)
#                      For ReTok/Centroid data: $EXT_TOKENIZER_DIR  (148,480 ship bundle)
#   WORKERS        — preprocess workers (default 288; normal partition full CPU node)
#
# Per reviewer round-2 Blocker 2: must be invoked TWICE — once per tokenizer —
# producing two separately-tokenized Megatron binaries. The Vanilla arm's
# 131,072-vocab embedding table can't index IDs from a 148,480-vocab
# tokenization (out-of-range), so the two arm families need separate inputs.
# Helper at the bottom of this file: see "Per-tokenizer pair invocation".

set -euo pipefail

# Source common config (paths to Megatron-LM-Swiss-AI, BASE_/EXT_ paths, etc.)
source "$(dirname "$0")/_train_config_common.env"

INPUT_JSONL="${INPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
# Default to extended (the "main" arm), but caller should override per family.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$EXT_DATA_PREFIX}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$EXT_TOKENIZER_DIR}"
WORKERS="${WORKERS:-288}"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

echo "=== preprocess_data.sbatch ==="
date -u
echo "host: $(hostname)"
echo "INPUT_JSONL:    $INPUT_JSONL"
echo "OUTPUT_PREFIX:  $OUTPUT_PREFIX"
echo "TOKENIZER_DIR:  $TOKENIZER_DIR"
echo "MEGATRON_DIR:   $MEGATRON_LM_SWISSAI_DIR"
echo "WORKERS:        $WORKERS"
echo

# Apertus tokenizer is HuggingFace-format; Megatron's preprocess_data
# supports HF tokenizers via --tokenizer-type HuggingFaceTokenizer.
# Confirm exact flag names from Agent A (Q D1).

uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
cd $MEGATRON_LM_SWISSAI_DIR
git rev-parse HEAD
python3 tools/preprocess_data.py \
    --input '$INPUT_JSONL' \
    --output-prefix '$OUTPUT_PREFIX' \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model '$TOKENIZER_DIR' \
    --append-eod \
    --workers $WORKERS \
    --log-interval 1000
"

echo
echo "=== done ==="
date -u
ls -la "$(dirname "$OUTPUT_PREFIX")" | head -10
echo
echo "Megatron prefix ready: $OUTPUT_PREFIX"
echo "Next: sbatch bakeoff_train.sbatch (or bash submit_all_arms.sh)"

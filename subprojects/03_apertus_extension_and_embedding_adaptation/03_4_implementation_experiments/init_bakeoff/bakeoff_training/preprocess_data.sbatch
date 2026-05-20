#!/usr/bin/env bash
#SBATCH --job-name=preprocess_data
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
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
#   TOKENIZER_DIR  — HF-format tokenizer dir (the 148,480 modern-only ship bundle)
#   WORKERS        — preprocess workers (default 64; xfer partition has 256 vCPU)

set -euo pipefail

# Source common config (paths to Megatron-LM-Swiss-AI, etc.)
source "$(dirname "$0")/_train_config_common.env"

INPUT_JSONL="${INPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_megatron/bulk_mix}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
WORKERS="${WORKERS:-64}"

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

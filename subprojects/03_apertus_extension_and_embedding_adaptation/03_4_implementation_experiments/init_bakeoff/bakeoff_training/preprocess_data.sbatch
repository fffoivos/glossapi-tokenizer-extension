#!/usr/bin/env bash
#SBATCH --job-name=preprocess_data
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
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
#   WORKERS        — preprocess workers (default: Slurm cpus-per-task)
#
# Per reviewer round-2 Blocker 2: must be invoked TWICE — once per tokenizer —
# producing two separately-tokenized Megatron binaries. The Vanilla arm's
# 131,072-vocab embedding table can't index IDs from a 148,480-vocab
# tokenization (out-of-range), so the two arm families need separate inputs.
# Helper at the bottom of this file: see "Per-tokenizer pair invocation".

set -euo pipefail

# Source common config. Slurm runs a copied script from /var/spool/slurmd, so
# direct dirname "$0" is not reliable inside sbatch. submit/resubmit commands
# should pass SCRIPT_DIR_OVERRIDE; SLURM_SUBMIT_DIR is the fallback.
SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"
GUARD="$SCRIPT_DIR/../slurm_cpu_only_guard.sh"
if [ ! -f "$GUARD" ]; then
    GUARD=/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/slurm_cpu_only_guard.sh
fi
source "$GUARD"
require_cpu_only_slurm "preprocess_data"

INPUT_JSONL="${INPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
# Default to extended (the "main" arm), but caller should override per family.
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$EXT_DATA_PREFIX}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$EXT_TOKENIZER_DIR}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
UENV_BIN="${UENV_BIN:-$(command -v uenv || true)}"
if [ -z "$UENV_BIN" ] && [ -x /usr/bin/uenv ]; then
    UENV_BIN=/usr/bin/uenv
fi
if [ -z "$UENV_BIN" ]; then
    echo "Could not find uenv; set UENV_BIN explicitly." >&2
    exit 127
fi

# Megatron's tools/preprocess_data.py appends "_text_document" for the default
# JSON key (`text`). Treat OUTPUT_PREFIX as the desired final data prefix used
# by training, and strip that suffix for the Megatron call when present.
if [[ "$OUTPUT_PREFIX" == *_text_document ]]; then
    MEGATRON_OUTPUT_PREFIX="${OUTPUT_PREFIX%_text_document}"
    DATA_PREFIX="$OUTPUT_PREFIX"
else
    MEGATRON_OUTPUT_PREFIX="$OUTPUT_PREFIX"
    DATA_PREFIX="${OUTPUT_PREFIX}_text_document"
fi

mkdir -p "$(dirname "$MEGATRON_OUTPUT_PREFIX")"

echo "=== preprocess_data.sbatch ==="
date -u
echo "host: $(hostname)"
echo "INPUT_JSONL:    $INPUT_JSONL"
echo "OUTPUT_PREFIX:  $OUTPUT_PREFIX"
echo "MEGATRON_PREFIX: $MEGATRON_OUTPUT_PREFIX"
echo "DATA_PREFIX:    $DATA_PREFIX"
echo "TOKENIZER_DIR:  $TOKENIZER_DIR"
echo "MEGATRON_DIR:   $MEGATRON_LM_SWISSAI_DIR"
echo "WORKERS:        $WORKERS"
echo "UENV_BIN:       $UENV_BIN"
echo

# Apertus tokenizer is HuggingFace-format; Megatron's preprocess_data
# supports HF tokenizers via --tokenizer-type HuggingFaceTokenizer.
# Confirm exact flag names from Agent A (Q D1).

\"$UENV_BIN\" run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
cd $MEGATRON_LM_SWISSAI_DIR
git rev-parse HEAD
python3 tools/preprocess_data.py \
    --input '$INPUT_JSONL' \
    --output-prefix '$MEGATRON_OUTPUT_PREFIX' \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model '$TOKENIZER_DIR' \
    --append-eod \
    --workers $WORKERS \
    --log-interval 1000
"

echo
echo "=== done ==="
date -u
test -f "${DATA_PREFIX}.bin"
test -f "${DATA_PREFIX}.idx"
ls -la "$(dirname "$DATA_PREFIX")" | head -10
echo
echo "Megatron prefix ready: $DATA_PREFIX"
echo "Next: sbatch bakeoff_train.sbatch (or bash submit_all_arms.sh)"

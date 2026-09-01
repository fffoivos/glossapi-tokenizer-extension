#!/usr/bin/env bash
#SBATCH --job-name=mini_mix_prepare
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

: "${SCHEDULE_CODE_BUNDLE:?set immutable schedule code bundle}"
: "${SOURCE_REPO_ROOT:?set frozen corpus builder checkout}"
: "${SOURCE_STAGE_ROOT:?set frozen vetted corpus stage}"
: "${SCHEDULE_STAGE_ROOT:?set new immutable Mini corpus stage}"
: "${MINI_SOURCE_DIR:?set pinned Mini tokenizer source directory}"
: "${MINI_OVERLAY_DIR:?set Mini-compatible tokenizer output directory}"

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
PRODUCTION_TOKENIZER_DIR=${PRODUCTION_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
BRIDGE_DIR="$SOURCE_REPO_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge"
SOURCE_INPUT="$SOURCE_STAGE_ROOT/input_receipt.json"
SOURCE_HELDOUT="$SOURCE_STAGE_ROOT/heldouts/heldout_manifest.json"
mkdir -p "$SCHEDULE_STAGE_ROOT/receipts"

uenv run "$UENV_IMAGE" --view=default -- python \
  "$SCHEDULE_CODE_BUNDLE/dataset/prepare_mini_overlay.py" \
  --mini-tokenizer-dir "$MINI_SOURCE_DIR" \
  --production-tokenizer-dir "$PRODUCTION_TOKENIZER_DIR" \
  --overlay-builder "$SCHEDULE_CODE_BUNDLE/initialization/build_mini_tokenizer_overlay.py" \
  --output-dir "$MINI_OVERLAY_DIR" \
  --cache-dir /iopsstor/scratch/cscs/fffoivos/huggingface_cache

uenv run "$UENV_IMAGE" --view=default -- python \
  "$SCHEDULE_CODE_BUNDLE/dataset/derive_mini_corpus_inputs.py" \
  --source-input-receipt "$SOURCE_INPUT" \
  --source-heldout-manifest "$SOURCE_HELDOUT" \
  --overlay-dir "$MINI_OVERLAY_DIR" \
  --output-stage "$SCHEDULE_STAGE_ROOT" \
  --bridge-common "$BRIDGE_DIR/scripts/bridge_common.py" \
  --derivation-id aperture_v1_1_0p5b_five_schedule_corpus_v1 \
  | tee "$SCHEDULE_STAGE_ROOT/receipts/derive_inputs.stdout.json"

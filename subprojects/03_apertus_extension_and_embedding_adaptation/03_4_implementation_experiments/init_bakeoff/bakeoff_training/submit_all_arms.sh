#!/usr/bin/env bash
# Submit all three bakeoff arms (vanilla / retok / centroid) in parallel
# with a shared data seed. Each arm gets one node × 4 × GH200 × 12 h.
#
# Usage:
#   bash submit_all_arms.sh
#
# Pre-conditions:
#   1. preprocess_data.sbatch has produced TWO Megatron .bin/.idx datasets —
#      one tokenized with the BASE 131,072 Apertus tokenizer (for Vanilla) at
#      $BASE_DATA_PREFIX, and one tokenized with the EXTENDED 148,480 ship
#      bundle (for ReTok / Centroid) at $EXT_DATA_PREFIX. Both come from the
#      same shuffled bulk_mix.jsonl with the same seed — they differ only in
#      tokenization (reviewer round-2 Blocker 2).
#   2. arms/build_init_checkpoints.sbatch + arms/convert_init_checkpoints.sbatch
#      have produced Megatron-format init checkpoints under
#      $INIT_CKPT_ROOT/{vanilla,retok,centroid}/megatron/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-bakeoff_$(date -u +%Y%m%d_%H%M%S)}"

echo "=== submit_all_arms.sh ==="
echo "INIT_CKPT_ROOT:   $INIT_CKPT_ROOT"
echo "OUT_ROOT:         $OUT_ROOT"
echo "RUN_TAG:          $RUN_TAG"
echo "BASE_DATA_PREFIX: $BASE_DATA_PREFIX  (Vanilla)"
echo "EXT_DATA_PREFIX:  $EXT_DATA_PREFIX   (ReTok / Centroid)"
echo

# Sanity: do all three init checkpoints exist?
for arm in vanilla retok centroid; do
    ckpt="$INIT_CKPT_ROOT/$arm/megatron"
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: init checkpoint missing for arm '$arm': $ckpt" >&2
        echo "  Run ../arms/submit_init_pipeline.sh first." >&2
        exit 2
    fi
done

# Sanity: do BOTH Megatron data prefixes exist?
for prefix in "$BASE_DATA_PREFIX" "$EXT_DATA_PREFIX"; do
    if [ ! -f "${prefix}.bin" ] || [ ! -f "${prefix}.idx" ]; then
        echo "ERROR: Megatron data missing at ${prefix}{.bin,.idx}" >&2
        echo "  Run preprocess_data.sbatch twice — once for the base tokenizer," >&2
        echo "  once for the extended tokenizer." >&2
        exit 2
    fi
done

# Submit all three
for arm in vanilla retok centroid; do
    output_dir="$OUT_ROOT/${RUN_TAG}_${arm}"
    init_ckpt="$INIT_CKPT_ROOT/$arm/megatron"
    mkdir -p "$output_dir"

    echo "Submitting $arm  →  $output_dir"
    sbatch \
        --job-name="bakeoff_${arm}" \
        --export=ALL,ARM=$arm,INIT_CKPT=$init_ckpt,OUTPUT_DIR=$output_dir,SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        "$SCRIPT_DIR/bakeoff_train.sbatch"
done

echo
echo "All three arms submitted. Monitor with: squeue -u $USER"
echo "When done, run per-arm eval:"
echo "  for arm in vanilla retok centroid; do"
echo "    bash ../eval/run_bakeoff_arm_eval.sh $OUT_ROOT/${RUN_TAG}_\$arm/checkpoints/<latest>"
echo "  done"

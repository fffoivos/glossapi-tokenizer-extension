#!/usr/bin/env bash
# Submit all three bakeoff arms (vanilla / retok / centroid) in parallel
# with a shared data seed. Each arm gets one node × 4 × GH200 × 12 h.
#
# Usage:
#   bash submit_all_arms.sh
#
# Pre-conditions:
#   1. preprocess_data.sbatch has produced the Megatron .bin/.idx at
#      $TRAIN_DATA_PREFIX (see _train_config_common.env)
#   2. arms/build_init_checkpoints.py + hfconverter have produced
#      Megatron-format init checkpoints under
#      $INIT_CKPT_ROOT/{vanilla,retok,centroid}/megatron/

set -euo pipefail

source "$(dirname "$0")/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-bakeoff_$(date -u +%Y%m%d_%H%M%S)}"

echo "=== submit_all_arms.sh ==="
echo "INIT_CKPT_ROOT: $INIT_CKPT_ROOT"
echo "OUT_ROOT:       $OUT_ROOT"
echo "RUN_TAG:        $RUN_TAG"
echo

# Sanity: do all three init checkpoints exist?
for arm in vanilla retok centroid; do
    ckpt="$INIT_CKPT_ROOT/$arm/megatron"
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: init checkpoint missing for arm '$arm': $ckpt" >&2
        echo "  Run ../arms/build_init_checkpoints.py first." >&2
        exit 2
    fi
done

# Sanity: does the Megatron data prefix exist?
if [ ! -f "${TRAIN_DATA_PREFIX}.bin" ] || [ ! -f "${TRAIN_DATA_PREFIX}.idx" ]; then
    echo "ERROR: Megatron data missing at $TRAIN_DATA_PREFIX{.bin,.idx}" >&2
    echo "  Run preprocess_data.sbatch first." >&2
    exit 2
fi

# Submit all three
for arm in vanilla retok centroid; do
    output_dir="$OUT_ROOT/${RUN_TAG}_${arm}"
    init_ckpt="$INIT_CKPT_ROOT/$arm/megatron"
    mkdir -p "$output_dir"

    echo "Submitting $arm  →  $output_dir"
    sbatch \
        --export=ALL,ARM=$arm,INIT_CKPT=$init_ckpt,OUTPUT_DIR=$output_dir \
        --job-name="bakeoff_${arm}" \
        "$(dirname "$0")/bakeoff_train.sbatch"
done

echo
echo "All three arms submitted. Monitor with: squeue -u $USER"
echo "When done, run per-arm eval:"
echo "  for arm in vanilla retok centroid; do"
echo "    bash ../eval/run_bakeoff_arm_eval.sh $OUT_ROOT/${RUN_TAG}_\$arm/checkpoints/<latest>"
echo "  done"

#!/usr/bin/env bash
# Submit all three bakeoff arms (vanilla / retok / centroid) in parallel
# with a shared data seed. Resource shape is controlled by _train_config_common.env
# and can be overridden with NODES=... TIME_LIMIT=... at submit time.
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
#      + megatron_patches/r17_patch_roundtrip.sbatch have produced patched
#      Megatron-format init checkpoints under
#      $INIT_CKPT_ROOT/{vanilla,retok,centroid}/megatron_tp2_r17patched/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
INIT_CKPT_SUBDIR="${INIT_CKPT_SUBDIR:-megatron_tp2_r17patched}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-bakeoff_$(date -u +%Y%m%d_%H%M%S)}"
CHAIN_RESUME="${CHAIN_RESUME:-0}"

echo "=== submit_all_arms.sh ==="
echo "INIT_CKPT_ROOT:   $INIT_CKPT_ROOT"
echo "INIT_CKPT_SUBDIR: $INIT_CKPT_SUBDIR"
echo "OUT_ROOT:         $OUT_ROOT"
echo "RUN_TAG:          $RUN_TAG"
echo "ACCOUNT:          $ACCOUNT"
echo "PARTITION:        $PARTITION"
echo "NODES:            $NODES"
echo "GPUS_PER_NODE:    $GPUS_PER_NODE"
echo "LAUNCH_MODE:      $LAUNCH_MODE"
echo "TIME_LIMIT:       $TIME_LIMIT"
echo "CHAIN_RESUME:     $CHAIN_RESUME"
echo "BASE_DATA_PREFIX: $BASE_DATA_PREFIX  (Vanilla)"
echo "EXT_DATA_PREFIX:  $EXT_DATA_PREFIX   (ReTok / Centroid)"
echo

if [ "$CHAIN_RESUME" != "0" ] && [ "$CHAIN_RESUME" != "1" ]; then
    echo "ERROR: CHAIN_RESUME=$CHAIN_RESUME not recognized (expected 0|1)" >&2
    exit 2
fi

SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

# Sanity: do all three init checkpoints exist?
for arm in vanilla retok centroid; do
    ckpt="$INIT_CKPT_ROOT/$arm/$INIT_CKPT_SUBDIR"
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
    init_ckpt="$INIT_CKPT_ROOT/$arm/$INIT_CKPT_SUBDIR"
    mkdir -p "$output_dir"

    echo "Submitting $arm  →  $output_dir"
    job_id=$(sbatch --parsable \
        --job-name="bakeoff_${arm}" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --nodes="$NODES" \
        --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --gres="gpu:$GPUS_PER_NODE" \
        --time="$TIME_LIMIT" \
        --export=ALL,ARM=$arm,INIT_CKPT=$init_ckpt,OUTPUT_DIR=$output_dir,SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$TIME_LIMIT",RESUME_TRAINING=0 \
        "$SCRIPT_DIR/bakeoff_train.sbatch")
    echo "  initial job: $job_id"

    if [ "$CHAIN_RESUME" = "1" ]; then
        resume_job_id=$(sbatch --parsable \
            --dependency="afterany:$job_id" \
            --job-name="bakeoff_resume_${arm}" \
            --account="$ACCOUNT" \
            --partition="$PARTITION" \
            --nodes="$NODES" \
            --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
            --gpus-per-node="$GPUS_PER_NODE" \
            --gres="gpu:$GPUS_PER_NODE" \
            --time="$TIME_LIMIT" \
            --export=ALL,ARM=$arm,INIT_CKPT=$output_dir/checkpoints,OUTPUT_DIR=$output_dir,SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$TIME_LIMIT",RESUME_TRAINING=1 \
            "$SCRIPT_DIR/bakeoff_train.sbatch")
        echo "  resume job:  $resume_job_id (afterany:$job_id)"
    fi
done

echo
echo "All three arms submitted. Monitor with: squeue -u $USER"
echo "When done, run per-arm eval:"
echo "  for arm in vanilla retok centroid; do"
echo "    bash ../eval/run_bakeoff_arm_eval.sh $OUT_ROOT/${RUN_TAG}_\$arm/checkpoints/<latest>"
echo "  done"

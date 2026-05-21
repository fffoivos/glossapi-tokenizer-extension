#!/usr/bin/env bash
# Submit the init-checkpoint build -> HF-to-Megatron conversion chain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
VOCAB_SIZE="${VOCAB_SIZE:-148480}"
ARMS="${ARMS:-vanilla retok centroid}"
INIT_UENV_IMAGE="${INIT_UENV_IMAGE:-pytorch/v2.9.1:v2}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/init

echo "=== submit_init_pipeline.sh ==="
echo "INIT_CKPT_ROOT: $INIT_CKPT_ROOT"
echo "VOCAB_SIZE:     $VOCAB_SIZE"
echo "ARMS:           $ARMS"
echo "INIT_UENV:      $INIT_UENV_IMAGE"
echo

build_job="$(
    sbatch --parsable \
        --export=ALL,INIT_CKPT_ROOT="$INIT_CKPT_ROOT",VOCAB_SIZE="$VOCAB_SIZE",ARMS="$ARMS",INIT_UENV_IMAGE="$INIT_UENV_IMAGE" \
        "$SCRIPT_DIR/build_init_checkpoints.sbatch"
)"

convert_job="$(
    sbatch --parsable \
        --dependency="afterok:$build_job" \
        --kill-on-invalid-dep=yes \
        --export=ALL,INIT_CKPT_ROOT="$INIT_CKPT_ROOT",ARMS="$ARMS",INIT_UENV_IMAGE="$INIT_UENV_IMAGE" \
        "$SCRIPT_DIR/convert_init_checkpoints.sbatch"
)"

echo "build_init_ckpts:   $build_job"
echo "convert_init_ckpts: $convert_job (afterok:$build_job)"
echo
echo "Monitor:"
echo "  squeue -j $build_job,$convert_job -o '%.18i %.32j %.10T %.10M %.12P %.40R'"

#!/usr/bin/env bash
# Source this from CPU-only Slurm jobs before doing real work.

require_cpu_only_slurm() {
    local label="${1:-CPU-only job}"
    local allowed_partition="${CPU_ONLY_PARTITION:-xfer}"
    local partition="${SLURM_JOB_PARTITION:-}"
    local gres_blob="${SLURM_JOB_GRES:-} ${SLURM_STEP_GRES:-} ${SLURM_GPUS:-} ${SLURM_STEP_GPUS:-}"

    if [ "${ALLOW_GPU_NODE_FOR_CPU:-0}" = "1" ]; then
        echo "WARNING: ALLOW_GPU_NODE_FOR_CPU=1; allowing $label on partition ${partition:-unknown}." >&2
        return 0
    fi

    if [ -n "$partition" ] && [ "$partition" != "$allowed_partition" ]; then
        echo "ERROR: $label is CPU-only but is running on Slurm partition '$partition'." >&2
        echo "Clariden partitions normal/debug/low allocate GPU nodes; use partition '$allowed_partition' or set ALLOW_GPU_NODE_FOR_CPU=1 intentionally." >&2
        exit 88
    fi

    if [[ "$gres_blob" == *gpu* ]]; then
        echo "ERROR: $label is CPU-only but Slurm assigned GPU GRES: $gres_blob" >&2
        echo "Use partition '$allowed_partition' or set ALLOW_GPU_NODE_FOR_CPU=1 intentionally." >&2
        exit 88
    fi
}

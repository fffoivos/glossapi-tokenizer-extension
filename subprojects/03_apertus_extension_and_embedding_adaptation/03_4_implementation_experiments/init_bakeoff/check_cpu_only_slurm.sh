#!/usr/bin/env bash
# Verify that CPU-only dataset/build/conversion sbatches cannot silently land on GPU partitions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cpu_only_jobs=(
    "corpus_build/prepare_greek_pool.sbatch"
    "corpus_build/normalize_nfc.sbatch"
    "corpus_build/mix_builder_smoke.sbatch"
    "corpus_build/mix_builder_full.sbatch"
    "corpus_build/concat_bulk_mix.sbatch"
    "bakeoff_training/preprocess_data.sbatch"
    "eval/build_cpt_heldout_jsonl.sbatch"
    "eval/convert_bakeoff_checkpoint_to_hf.sbatch"
    "arms/build_init_checkpoints.sbatch"
    "arms/convert_init_checkpoints.sbatch"
)

fail=0

for job in "${cpu_only_jobs[@]}"; do
    if [ ! -f "$job" ]; then
        echo "ERROR: missing CPU-only sbatch: $job" >&2
        fail=1
        continue
    fi

    if ! grep -Eq '^#SBATCH[[:space:]]+--partition=xfer($|[[:space:]])' "$job"; then
        echo "ERROR: $job must use #SBATCH --partition=xfer" >&2
        fail=1
    fi

    if grep -Eq '^#SBATCH[[:space:]]+(--gpus|--gpus-per-node|--gres=.*gpu|--gres[[:space:]]+gpu)' "$job"; then
        echo "ERROR: $job is CPU-only but requests GPU resources" >&2
        fail=1
    fi

    if ! grep -q 'require_cpu_only_slurm' "$job"; then
        echo "ERROR: $job must call require_cpu_only_slurm before work starts" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "CPU-only Slurm audit passed for ${#cpu_only_jobs[@]} jobs."

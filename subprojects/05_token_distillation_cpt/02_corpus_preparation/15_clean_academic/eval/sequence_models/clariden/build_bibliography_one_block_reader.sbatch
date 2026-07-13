#!/usr/bin/env bash
# Build the one-BIB-block document reader on a Clariden CPU node.
#SBATCH --job-name=cpt_bib_one_reader
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set the immutable code-bundle root}
EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?set INVENTORY.sha256 hash}
INPUT=${INPUT:?set the physically test-stripped STRUCT-2K LLM-silver JSONL}
OUTPUT_DIR=${OUTPUT_DIR:?set a new immutable output directory}
CODE_COMMIT=${CODE_COMMIT:?set the local source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT/sequence_models" && ! -L "$CODE_ROOT" ]] || {
    echo "ERROR: invalid CODE_ROOT: $CODE_ROOT" >&2
    exit 90
}
[[ "$(sha256sum "$CODE_ROOT/INVENTORY.sha256" | cut -d' ' -f1)" == "$EXPECTED_INVENTORY_SHA256" ]] || {
    echo "ERROR: code inventory hash mismatch" >&2
    exit 91
}
(cd "$CODE_ROOT" && sha256sum -c INVENTORY.sha256)
[[ -f "$INPUT" && ! -L "$INPUT" ]] || {
    echo "ERROR: INPUT must be a regular file: $INPUT" >&2
    exit 92
}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
    echo "ERROR: immutable output already exists: $OUTPUT_DIR" >&2
    exit 93
}

trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH="$CODE_ROOT"

uenv run "$UENV" --view=default -- \
    python3 -m sequence_models.bibliography_one_block_reader \
        --input "$INPUT" \
        --output-dir "$OUTPUT_DIR" \
        --workers "$SLURM_CPUS_PER_TASK" \
        --max-physical-gap 64 \
        --code-commit "$CODE_COMMIT" \
        --slurm-job-id "$SLURM_JOB_ID"

trap - EXIT
echo "BIBLIOGRAPHY_ONE_BLOCK_READER=$OUTPUT_DIR"


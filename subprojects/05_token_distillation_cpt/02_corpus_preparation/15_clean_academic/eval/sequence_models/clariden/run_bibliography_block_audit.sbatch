#!/usr/bin/env bash
# Count continuous silver BIB blocks on a Clariden CPU node.
#SBATCH --job-name=cpt_bib_block_audit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

SCRIPT=${SCRIPT:?set the immutable audit script path}
EXPECTED_SCRIPT_SHA256=${EXPECTED_SCRIPT_SHA256:?set the audit script SHA-256}
INPUT=${INPUT:?set the physically test-stripped STRUCT-2K LLM-silver JSONL}
OUTPUT_DIR=${OUTPUT_DIR:?set a new immutable output directory}
CODE_COMMIT=${CODE_COMMIT:?set the local source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$SCRIPT" && ! -L "$SCRIPT" ]] || {
    echo "ERROR: SCRIPT must be a regular file: $SCRIPT" >&2
    exit 90
}
[[ "$(sha256sum "$SCRIPT" | cut -d' ' -f1)" == "$EXPECTED_SCRIPT_SHA256" ]] || {
    echo "ERROR: script hash mismatch" >&2
    exit 91
}
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
    python3 "$SCRIPT" \
        --input "$INPUT" \
        --output-dir "$OUTPUT_DIR" \
        --top-n 25 \
        --max-physical-gap 64 \
        --code-commit "$CODE_COMMIT" \
        --slurm-job-id "$SLURM_JOB_ID"

trap - EXIT
echo "BIBLIOGRAPHY_BLOCK_AUDIT=$OUTPUT_DIR"

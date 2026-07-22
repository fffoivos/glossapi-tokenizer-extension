#!/usr/bin/env bash
# Audit what frozen bibliography features demonstrate on CONTINUATION lines.
#SBATCH --job-name=bib_cont_features
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
TABLE_DIR=${TABLE_DIR:?set connector table root}
MODEL_DIR=${MODEL_DIR:?set connector OOF model root}
ENTRY_OOF=${ENTRY_OOF:?set frozen P0D OOF probability path}
SOURCE_JSONL=${SOURCE_JSONL:?set provenance-bound source JSONL}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable audit root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_continuation_feature_audit \
    --table-dir "$TABLE_DIR" --model-dir "$MODEL_DIR" --entry-oof "$ENTRY_OOF" \
    --source-jsonl "$SOURCE_JSONL" --output-dir "$OUTPUT_DIR" \
    --seed 20260717 --repetitions 12 --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_CONTINUATION_FEATURE_AUDIT=$OUTPUT_DIR"

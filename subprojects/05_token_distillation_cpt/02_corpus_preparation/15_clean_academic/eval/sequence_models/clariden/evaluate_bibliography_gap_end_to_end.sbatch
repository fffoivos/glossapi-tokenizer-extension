#!/usr/bin/env bash
# Decode gap-connector OOF predictions and evaluate complete bibliography masks.
#SBATCH --job-name=bib_gap_e2e
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned sklearn dependency root}
SOURCE=${SOURCE:?set immutable Struct2K source JSONL}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set immutable base feature table root}
ENTRY_OOF_DIR=${ENTRY_OOF_DIR:?set immutable entry OOF root}
HEADING_OOF_DIR=${HEADING_OOF_DIR:?set immutable heading OOF root}
CANDIDATE_TABLE_DIR=${CANDIDATE_TABLE_DIR:?set immutable candidate table root}
PROBABILITY_NPZ=${PROBABILITY_NPZ:?set candidate OOF probability archive}
THRESHOLD_NPZ=${THRESHOLD_NPZ:?set candidate OOF threshold archive}
CONFIGURATION_KEY=${CONFIGURATION_KEY:?set exact candidate configuration key}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable evaluation root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_gap_end_to_end \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" \
    --entry-oof-dir "$ENTRY_OOF_DIR" --heading-oof-dir "$HEADING_OOF_DIR" \
    --candidate-table-dir "$CANDIDATE_TABLE_DIR" \
    --probability-npz "$PROBABILITY_NPZ" --threshold-npz "$THRESHOLD_NPZ" \
    --configuration-key "$CONFIGURATION_KEY" --output-dir "$OUTPUT_DIR" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_GAP_END_TO_END=$OUTPUT_DIR"

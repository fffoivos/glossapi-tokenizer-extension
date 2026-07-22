#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-unseen-feat
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
DOCUMENTS=${DOCUMENTS:?set sealed test documents}
CANDIDATE_FREEZE=${CANDIDATE_FREEZE:?set frozen candidate receipt}
TEST_SEAL=${TEST_SEAL:?set test seal receipt}
ENTRY_MODEL_DIR=${ENTRY_MODEL_DIR:?set P0D model root}
SIGNAL_MODEL_DIR=${SIGNAL_MODEL_DIR:?set signal-TCN model root}
HEADING_MODEL_DIR=${HEADING_MODEL_DIR:?set heading model root}
CONNECTOR_MODEL_DIR=${CONNECTOR_MODEL_DIR:?set connector model root}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable unseen feature root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_nextgen_unseen_features \
    --documents "$DOCUMENTS" --candidate-freeze "$CANDIDATE_FREEZE" \
    --test-seal "$TEST_SEAL" --entry-model-dir "$ENTRY_MODEL_DIR" \
    --signal-model-dir "$SIGNAL_MODEL_DIR" --heading-model-dir "$HEADING_MODEL_DIR" \
    --connector-model-dir "$CONNECTOR_MODEL_DIR" --output-dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_UNSEEN_FEATURES=$OUTPUT_DIR"

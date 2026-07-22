#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-post-test
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
FEATURE_TABLE_DIR=${FEATURE_TABLE_DIR:?set opened-test feature table}
DOCUMENTS=${DOCUMENTS:?set opened-test documents}
LINE_KEY=${LINE_KEY:?set opened-test line key}
LABELS=${LABELS:?set opened-test labels}
LINE_PROBABILITY=${LINE_PROBABILITY:?set frozen line probability}
BASELINE_PREDICTION=${BASELINE_PREDICTION:?set frozen baseline prediction}
DECODER_REPORT=${DECODER_REPORT:?set post-review decoder report}
SCOPE_DIR=${SCOPE_DIR:?set post-review scope model}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable evaluation root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set code identity}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_nextgen_postreview_test \
    --feature-table-dir "$FEATURE_TABLE_DIR" --documents "$DOCUMENTS" \
    --line-key "$LINE_KEY" --labels "$LABELS" \
    --line-probability "$LINE_PROBABILITY" \
    --baseline-prediction "$BASELINE_PREDICTION" \
    --decoder-report "$DECODER_REPORT" --scope-dir "$SCOPE_DIR" \
    --output-dir "$OUTPUT_DIR" --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_POSTREVIEW_TEST=$OUTPUT_DIR"

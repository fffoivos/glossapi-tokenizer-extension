#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-unseen-eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
CANDIDATE_FREEZE=${CANDIDATE_FREEZE:?set frozen candidate receipt}
PREDICTION_DIR=${PREDICTION_DIR:?set label-blind prediction root}
TEST_SEAL=${TEST_SEAL:?set test seal receipt}
DOCUMENTS=${DOCUMENTS:?set sealed documents}
LINE_KEY=${LINE_KEY:?set sealed line key}
LABELS=${LABELS:?set sealed consensus labels}
FEATURE_TABLE_DIR=${FEATURE_TABLE_DIR:?set unseen feature table}
OUTPUT=${OUTPUT:?set new immutable evaluation report}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 90
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_unseen_evaluate \
    --candidate-freeze "$CANDIDATE_FREEZE" --prediction-dir "$PREDICTION_DIR" \
    --test-seal "$TEST_SEAL" --documents "$DOCUMENTS" --line-key "$LINE_KEY" \
    --labels "$LABELS" --feature-table-dir "$FEATURE_TABLE_DIR" --output "$OUTPUT" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
echo "BIBLIOGRAPHY_NEXTGEN_UNSEEN_EVALUATION=$OUTPUT"

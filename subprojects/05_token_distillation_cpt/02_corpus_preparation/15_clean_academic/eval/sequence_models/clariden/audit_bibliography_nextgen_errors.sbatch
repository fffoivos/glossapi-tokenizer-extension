#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-audit
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
SOURCE=${SOURCE:?set train source JSONL}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set entry feature table}
PREDICTION=${PREDICTION:?set candidate OOF prediction}
OUTPUT=${OUTPUT:?set create-only audit JSON}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
BASELINE_PREDICTION=${BASELINE_PREDICTION:-}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 90
arguments=()
if [[ -n "$BASELINE_PREDICTION" ]]; then
  arguments+=(--baseline-prediction "$BASELINE_PREDICTION")
fi
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_error_audit \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" \
    --prediction "$PREDICTION" "${arguments[@]}" --output "$OUTPUT" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
echo "BIBLIOGRAPHY_NEXTGEN_AUDIT=$OUTPUT"

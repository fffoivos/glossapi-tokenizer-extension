#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-scope
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
SOURCE=${SOURCE:?set development source}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set base table}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable scope root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_auxiliary_scope \
    --source "$SOURCE" --base-table-dir "$BASE_TABLE_DIR" --output-dir "$OUTPUT_DIR" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_AUXILIARY_SCOPE=$OUTPUT_DIR"

#!/usr/bin/env bash
# Re-index the frozen train table so each source is one held-out fold.
#SBATCH --job-name=bib_source_table
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
SOURCE_TABLE_DIR=${SOURCE_TABLE_DIR:?set original train table}
OUTPUT_DIR=${OUTPUT_DIR:?set new source-holdout table directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.6.0:v1}

[[ -f "$SOURCE_TABLE_DIR/receipt.json" && ! -L "$SOURCE_TABLE_DIR/receipt.json" ]] || exit 90
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_source_holdout_table \
  --source-table-dir "$SOURCE_TABLE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SOURCE_HOLDOUT_TABLE=$OUTPUT_DIR"

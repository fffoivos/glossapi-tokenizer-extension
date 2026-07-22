#!/usr/bin/env bash
# Train grouped OOF role-aware bibliography sequence models on CPU.
#SBATCH --job-name=bib_role_seq
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line model}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set frozen B0 configuration}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:?set train-only role arrays}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable role-sequence directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in \
  "$TABLE_DIR/receipt.json" \
  "$LINE_OOF_DIR/receipt.json" \
  "$BLOCK_OOF_DIR/receipt.json" \
  "$DETERMINISTIC_ROLES_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ -f "$QUALITY_DECISIONS" && ! -L "$QUALITY_DECISIONS" ]] || exit 91
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 92

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_entry_role_sequence \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ROLE_SEQUENCE=$OUTPUT_DIR"

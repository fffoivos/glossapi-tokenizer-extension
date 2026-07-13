#!/usr/bin/env bash
# Build the full-document high-impact bibliography recall-failure reader.
#SBATCH --job-name=bib_entry_fn_site
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
VALIDATION_DIR=${VALIDATION_DIR:?set completed retrospective-validation directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable failure-review directory}
DOCUMENT_IDS_COLON=${DOCUMENT_IDS_COLON:?set colon-separated validation document IDs}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$VALIDATION_DIR/receipt.json" && ! -L "$VALIDATION_DIR" ]] || exit 91
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 92

IFS=: read -r -a DOCUMENT_IDS <<< "$DOCUMENT_IDS_COLON"
DOCUMENT_ARGS=()
for document_id in "${DOCUMENT_IDS[@]}"; do
  [[ "$document_id" =~ ^[0-9a-f]{64}$ ]] || exit 93
  DOCUMENT_ARGS+=(--document-id "$document_id")
done

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_failure_review \
  --input "$INPUT" \
  --validation-dir "$VALIDATION_DIR" \
  "${DOCUMENT_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_FAILURE_REVIEW=$OUTPUT_DIR"

#!/usr/bin/env bash
# Read-only structural audit of train-OOF bibliography components.
#SBATCH --job-name=bib_component_diag
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
COMPONENT_DIR=${COMPONENT_DIR:?set train-OOF component-gate directory}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line-model directory}
OUTPUT_PATH=${OUTPUT_PATH:?set new diagnostic JSON path}
VARIANT=${VARIANT:-no_length}
MODEL_ARM=${MODEL_ARM:-logistic_l2}
THRESHOLD=${THRESHOLD:-0.5}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:-}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR/receipt.json" ]] || exit 90
[[ -f "$COMPONENT_DIR/receipt.json" && ! -L "$COMPONENT_DIR/receipt.json" ]] || exit 91
[[ ! -e "$OUTPUT_PATH" && ! -L "$OUTPUT_PATH" ]] || exit 92

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

ROLE_ARGS=()
if [[ -n "$DETERMINISTIC_ROLES_DIR" ]]; then
  [[ -f "$DETERMINISTIC_ROLES_DIR/receipt.json" && ! -L "$DETERMINISTIC_ROLES_DIR/receipt.json" ]] || exit 93
  ROLE_ARGS+=(--deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR")
fi

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_entry_component_diagnostics \
  --table-dir "$TABLE_DIR" \
  --component-dir "$COMPONENT_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --variant "$VARIANT" \
  --model-arm "$MODEL_ARM" \
  --threshold "$THRESHOLD" \
  "${ROLE_ARGS[@]}" \
  --output-path "$OUTPUT_PATH"

echo "BIBLIOGRAPHY_COMPONENT_DIAGNOSTICS=$OUTPUT_PATH"

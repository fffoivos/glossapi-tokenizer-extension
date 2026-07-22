#!/usr/bin/env bash
# Cross-fitted learned gate over high-recall bibliography components.
#SBATCH --job-name=bib_component_gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set line OOF directory}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set block OOF directory}
SEQUENCE_OOF_DIR=${SEQUENCE_OOF_DIR:?set sequence-ablation OOF directory}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:?set train-only deterministic roles}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable component-gate directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in "$TABLE_DIR/receipt.json" "$LINE_OOF_DIR/receipt.json" "$BLOCK_OOF_DIR/receipt.json" "$SEQUENCE_OOF_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ -f "$QUALITY_DECISIONS" && ! -L "$QUALITY_DECISIONS" ]] || exit 91
[[ -f "$DETERMINISTIC_ROLES_DIR/receipt.json" && ! -L "$DETERMINISTIC_ROLES_DIR/receipt.json" ]] || exit 94
[[ -d "$PY_DEPS" && ! -L "$PY_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_component_gate \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --sequence-oof-dir "$SEQUENCE_OOF_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_COMPONENT_GATE=$OUTPUT_DIR"

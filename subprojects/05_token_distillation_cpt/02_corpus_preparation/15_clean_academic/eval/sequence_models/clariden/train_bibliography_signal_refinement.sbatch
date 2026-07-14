#!/usr/bin/env bash
# Train-only edge and component refinements over the frozen signal decoder.
#SBATCH --job-name=bib_signal_refine
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line model}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set frozen train-OOF signal TCN}
SIGNAL_BLOCK_DIR=${SIGNAL_BLOCK_DIR:?set frozen recall-first decoder artifact}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:?set train-only deterministic roles}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable refinement directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in \
  "$TABLE_DIR/receipt.json" \
  "$LINE_OOF_DIR/receipt.json" \
  "$SIGNAL_TCN_DIR/receipt.json" \
  "$SIGNAL_BLOCK_DIR/receipt.json" \
  "$DETERMINISTIC_ROLES_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ -f "$QUALITY_DECISIONS" && ! -L "$QUALITY_DECISIONS" ]] || exit 91
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93
for module in line_lr.py span_signals.py span_seq_data.py; do
  [[ -f "$CODE_ROOT/$module" && ! -L "$CODE_ROOT/$module" ]] || exit 94
done

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" \
  python3 -m sequence_models.bibliography_signal_refinement \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --signal-block-dir "$SIGNAL_BLOCK_DIR" \
  --deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SIGNAL_REFINEMENT=$OUTPUT_DIR"

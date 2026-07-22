#!/usr/bin/env bash
# Build a prediction-blind extraction-quality audit for bibliography validation.
#SBATCH --job-name=bib_val_quality
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

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned Python dependency directory}
NOISE_DEPS=${NOISE_DEPS:?set directory containing glossapi_rs_noise wheel install}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
VALIDATION_DIR=${VALIDATION_DIR:?set completed retrospective-validation directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable quality-audit directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$VALIDATION_DIR/receipt.json" && ! -L "$VALIDATION_DIR" ]] || exit 91
[[ -d "$NOISE_DEPS" && ! -L "$NOISE_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

DECISION_ARGS=()
if [[ -n "${DECISIONS:-}" ]]; then
  [[ -f "$DECISIONS" && ! -L "$DECISIONS" ]] || exit 94
  DECISION_ARGS=(--decisions "$DECISIONS")
fi

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$NOISE_DEPS:$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_validation_quality \
  --input "$INPUT" \
  --validation-dir "$VALIDATION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "${DECISION_ARGS[@]}" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_VALIDATION_QUALITY=$OUTPUT_DIR"

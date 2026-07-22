#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-scope
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
TABLE_DIR=${TABLE_DIR:?set nextgen feature table}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set entry feature table}
MODEL_DIR=${MODEL_DIR:?set line-model root}
DECODER_DIR=${DECODER_DIR:?set decoded development root}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable scope-model root}
KIND=${KIND:?set hist or linear}
WEIGHTING=${WEIGHTING:-balanced}
TARGET_MIN_GOLD_FRACTION=${TARGET_MIN_GOLD_FRACTION:-0.5}
HEADING_RESCUE_FLOOR=${HEADING_RESCUE_FLOOR:-0.6}
IMAGE_FRACTION_VETO=${IMAGE_FRACTION_VETO:-0.15}
RULE_FRACTION_VETO=${RULE_FRACTION_VETO:-0.05}
PROPOSAL_POOL=${PROPOSAL_POOL:-1}
MAX_SPURIOUS_BLOCKS=${MAX_SPURIOUS_BLOCKS:-0.02}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 92
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
ARGS=(
  --base-table-dir "$BASE_TABLE_DIR" --table-dir "$TABLE_DIR"
  --model-dir "$MODEL_DIR" --decoder-dir "$DECODER_DIR"
  --output-dir "$OUTPUT_DIR" --kind "$KIND" --weighting "$WEIGHTING"
  --target-min-gold-fraction "$TARGET_MIN_GOLD_FRACTION"
  --heading-rescue-floor "$HEADING_RESCUE_FLOOR"
  --image-fraction-veto "$IMAGE_FRACTION_VETO"
  --rule-fraction-veto "$RULE_FRACTION_VETO"
  --max-spurious-blocks "$MAX_SPURIOUS_BLOCKS"
  --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
)
[[ "$PROPOSAL_POOL" == 1 ]] && ARGS+=(--proposal-pool)
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_nextgen_scope "${ARGS[@]}"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_SCOPE=$OUTPUT_DIR"

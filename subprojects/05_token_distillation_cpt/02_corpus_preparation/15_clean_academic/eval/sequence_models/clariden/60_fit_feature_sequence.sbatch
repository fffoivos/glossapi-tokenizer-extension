#!/usr/bin/env bash
#SBATCH --job-name=tocbib_feature_crf
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

SEQUENCE_CLARIDEN_DIR=${SEQUENCE_CLARIDEN_DIR:?submit through submit_sequence.sh}
SEQUENCE_DIR=$(cd "$SEQUENCE_CLARIDEN_DIR/.." && pwd)
ACADEMIC_DIR=$(cd "$SEQUENCE_DIR/../.." && pwd)
PHASE04_DIR=$(cd "$ACADEMIC_DIR/../../04_full_corpus_preparation" && pwd)
source "$PHASE04_DIR/clariden/paths.env"
source "$PHASE04_DIR/clariden/common.sh"
phase04_require_compute_partition "feature sequence-model fitting"
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
[[ "${SLURM_GPUS:-0}" == "0" ]] || { echo "ERROR: sequence fitting is CPU-only" >&2; exit 93; }

GOLD=${GOLD:?set GOLD to imported academic-structure-gold-v1 JSONL}
SPLIT_MANIFEST=${SPLIT_MANIFEST:?set SPLIT_MANIFEST from the locked packet set}
ARCHITECTURE=${ARCHITECTURE:-c2-char-ngram-feature-bioes-crf}
case "$ARCHITECTURE" in
  c1-feature-bioes-crf|c2-char-ngram-feature-bioes-crf) ;;
  *) echo "ERROR: unsupported fit architecture $ARCHITECTURE" >&2; exit 94 ;;
esac
RUN_TAG=${RUN_TAG:-${ARCHITECTURE}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$RUN_ROOT/$RUN_TAG}
[[ ! -e "$OUT" ]] || { echo "ERROR: refusing to overwrite $OUT" >&2; exit 95; }
PARTIAL="$OUT.partial-${SLURM_JOB_ID}"
mkdir -p "$PARTIAL"
phase04_require_file "$GOLD"
phase04_require_file "$SPLIT_MANIFEST"
export PYTHONPATH="$ACADEMIC_DIR/eval${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0

uenv run "$PHASE04_UENV" --view=default -- \
  "$RUNTIME_VENV/bin/python" -m sequence_models.contract validate \
  --gold "$GOLD" --config "$SEQUENCE_DIR/config.json" \
  --split-manifest "$SPLIT_MANIFEST" --promotion \
  --output "$PARTIAL/contract_receipt.json"

uenv run "$PHASE04_UENV" --view=default -- \
  "$RUNTIME_VENV/bin/python" -m sequence_models.feature_crf \
  --gold "$GOLD" --config "$SEQUENCE_DIR/config.json" \
  --split-manifest "$SPLIT_MANIFEST" --architecture "$ARCHITECTURE" \
  --model-out "$PARTIAL/model.npz" \
  --validation-predictions "$PARTIAL/validation_predictions.jsonl" \
  > "$PARTIAL/training_receipt.json"

sha256sum "$PARTIAL"/* > "$PARTIAL/outputs.sha256"
touch "$PARTIAL/COMPLETED"
mv "$PARTIAL" "$OUT"
echo "FIT_COMPLETE=$OUT"

#!/usr/bin/env bash
# Build code/math heldout JSONLs on CPU-only resources after the 04 dataset exists.

#SBATCH --job-name=04_cm_heldout
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "04 code/math heldout build"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
DATASET_RUN_DIR="${DATASET_RUN_DIR:-$RUN_ROOT/latest_dataset_5b}"
OUT_ROOT="${OUT_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout}"
CODE_OUTPUT_JSONL="${CODE_OUTPUT_JSONL:-$OUT_ROOT/cpt_code_heldout_200_20260528.jsonl}"
MATH_OUTPUT_JSONL="${MATH_OUTPUT_JSONL:-$OUT_ROOT/cpt_math_heldout_200_20260528.jsonl}"
MANIFEST_JSON="${MANIFEST_JSON:-$OUT_ROOT/cpt_code_math_heldout_20260528.manifest.json}"
CODE_QUOTA="${CODE_QUOTA:-200}"
MATH_QUOTA="${MATH_QUOTA:-200}"
MIN_CHARS="${MIN_CHARS:-400}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /users/fffoivos/.local/bin/uv ]; then
  UV_BIN=/users/fffoivos/.local/bin/uv
fi

test -f "$DATASET_RUN_DIR/dataset_paths.env"
# shellcheck disable=SC1090
source "$DATASET_RUN_DIR/dataset_paths.env"
test -f "$JSONL"
test -f "$RECIPE"
test -x "$UV_BIN"
mkdir -p "$OUT_ROOT"

echo "=== build_code_math_heldouts.sbatch ==="
date -u
echo "host:              $HOSTNAME"
echo "DATASET_RUN_DIR:   $DATASET_RUN_DIR"
echo "TRAINING_JSONL:    $JSONL"
echo "RECIPE:            $RECIPE"
echo "CODE_OUTPUT_JSONL: $CODE_OUTPUT_JSONL"
echo "MATH_OUTPUT_JSONL: $MATH_OUTPUT_JSONL"
echo "MANIFEST_JSON:     $MANIFEST_JSON"
echo "CODE/MATH_QUOTA:   $CODE_QUOTA / $MATH_QUOTA"
echo "MIN_CHARS:         $MIN_CHARS"
echo

"$UV_BIN" run \
  --python 3.12 \
  --with datasets \
  --with pyarrow \
  --with huggingface-hub \
  --with numpy \
  -- python3 "$SUBPROJECT_DIR/scripts/build_code_math_heldouts.py" \
    --recipe-json "$RECIPE" \
    --training-jsonl "$JSONL" \
    --mix-builder-dir "$LEGACY_INIT_DIR/corpus_build" \
    --output-dir "$OUT_ROOT" \
    --code-output-jsonl "$CODE_OUTPUT_JSONL" \
    --math-output-jsonl "$MATH_OUTPUT_JSONL" \
    --manifest-json "$MANIFEST_JSON" \
    --code-quota "$CODE_QUOTA" \
    --math-quota "$MATH_QUOTA" \
    --min-chars "$MIN_CHARS"

echo
echo "=== code/math heldout build complete ==="
date -u
wc -l "$CODE_OUTPUT_JSONL" "$MATH_OUTPUT_JSONL"
ls -lh "$CODE_OUTPUT_JSONL" "$MATH_OUTPUT_JSONL" "$MANIFEST_JSON"

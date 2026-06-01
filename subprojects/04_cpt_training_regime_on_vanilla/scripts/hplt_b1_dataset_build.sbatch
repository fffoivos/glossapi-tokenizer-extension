#!/usr/bin/env bash
# Build the final Task-1 HPLT+B1 base-tokenized dataset on CPU-only resources.

#SBATCH --job-name=04_hplt_b1_build
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "04 HPLT+B1 final dataset build"

STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
SELECTED="${SELECTED:-$STAGE_ROOT/cpt/selected_after_apertus_and_internal_dedup.parquet}"
BASE_TOKENIZER="${BASE_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
OLD_RECIPE="${OLD_RECIPE:-$LEGACY_INIT_DIR/corpus_build/recipes/bulk.json}"
TARGET_TOKENS="${TARGET_TOKENS:-5000000000}"
RUN_TAG="${RUN_TAG:-04_hplt_b1_dataset_5b_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
UENV_BIN="${UENV_BIN:-$(command -v uenv || true)}"
if [ -z "$UENV_BIN" ] && [ -x /usr/bin/uenv ]; then
  UENV_BIN=/usr/bin/uenv
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /users/fffoivos/.local/bin/uv ]; then
  UV_BIN=/users/fffoivos/.local/bin/uv
fi

RECIPE="$RUN_DIR/recipe/hplt_b1_vanilla_regime.json"
JSONL="$RUN_DIR/jsonl/hplt_b1_5b.jsonl"
DATA_PREFIX="$RUN_DIR/megatron/hplt_b1_base_text_document"
VALIDATION_JSON="$RUN_DIR/validation/dataset_validation.json"

mkdir -p "$RUN_ROOT" "$RUN_DIR"/{recipe,jsonl,megatron,validation}

echo "=== 04 HPLT+B1 final dataset build ==="
date -u
echo "host: $HOSTNAME"
echo "REPO_ROOT: $REPO_ROOT"
echo "LEGACY_INIT_DIR: $LEGACY_INIT_DIR"
echo "SELECTED: $SELECTED"
echo "BASE_TOKENIZER: $BASE_TOKENIZER"
echo "OLD_RECIPE: $OLD_RECIPE"
echo "TARGET_TOKENS: $TARGET_TOKENS"
echo "RUN_DIR: $RUN_DIR"
echo "UENV_BIN: $UENV_BIN"
echo "UV_BIN: $UV_BIN"
echo

test -f "$SELECTED"
test -d "$BASE_TOKENIZER"
test -f "$OLD_RECIPE"
if [ -z "$UENV_BIN" ] || [ ! -x "$UENV_BIN" ]; then
  test -x "$UV_BIN"
fi

export SELECTED REPO_ROOT LEGACY_INIT_DIR SUBPROJECT_DIR BASE_TOKENIZER OLD_RECIPE
export TARGET_TOKENS RUN_DIR RECIPE JSONL DATA_PREFIX VALIDATION_JSON

run_dataset_steps() {
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:$LEGACY_INIT_DIR/corpus_build"

python3 "$SUBPROJECT_DIR/scripts/make_hplt_b1_recipe.py" \
  --input "$OLD_RECIPE" \
  --output "$RECIPE"

python3 -u "$LEGACY_INIT_DIR/corpus_build/mix_builder.py" \
  --recipe "$RECIPE" \
  --target-tokens "$TARGET_TOKENS" \
  --tokenizer "$BASE_TOKENIZER" \
  --output "$JSONL" \
  --seed 20260528 \
  --progress-every-tokens 50000000

python3 -u "$LEGACY_INIT_DIR/corpus_build/preprocess_hf_jsonl_to_megatron.py" \
  --input "$JSONL" \
  --output-prefix "$DATA_PREFIX" \
  --tokenizer-model "$BASE_TOKENIZER" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --chunk-size 32 \
  --log-interval 5000 \
  --manifest "$DATA_PREFIX.manifest.json"

python3 "$SUBPROJECT_DIR/scripts/validate_dataset_smoke.py" \
  --recipe "$RECIPE" \
  --mix-manifest "${JSONL%.jsonl}.manifest.json" \
  --preprocess-manifest "$DATA_PREFIX.manifest.json" \
  --data-prefix "$DATA_PREFIX" \
  --vocab-size 131072 \
  --output-json "$VALIDATION_JSON"
}
export -f run_dataset_steps

if [ -n "$UENV_BIN" ] && [ -x "$UENV_BIN" ]; then
  "$UENV_BIN" run "$UENV_IMAGE" --view=default -- bash -lc run_dataset_steps
else
  "$UV_BIN" run \
    --python 3.12 \
    --with transformers \
    --with datasets \
    --with pyarrow \
    --with numpy \
    --with huggingface-hub \
    -- bash -lc run_dataset_steps
fi

ln -sfn "$RUN_DIR" "$RUN_ROOT/latest_dataset_5b"
ln -sfn "$RUN_DIR" "$RUN_ROOT/latest_dataset"
cat > "$RUN_DIR/dataset_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
DATA_PREFIX="$DATA_PREFIX"
JSONL="$JSONL"
RECIPE="$RECIPE"
VALIDATION_JSON="$VALIDATION_JSON"
TARGET_TOKENS="$TARGET_TOKENS"
PATHS

echo
echo "=== final dataset build complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "DATA_PREFIX=$DATA_PREFIX"
echo "VALIDATION_JSON=$VALIDATION_JSON"

#!/usr/bin/env bash
#SBATCH --job-name=build_cpt_heldout
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SCRIPT_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd)"

SELECTED_PARQUET="${SELECTED_PARQUET:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
BULK_MIX_JSONL="${BULK_MIX_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
RECIPE_JSON="${RECIPE_JSON:-$SCRIPT_DIR/../corpus_build/recipes/bulk.json}"
OUT_ROOT="${OUT_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUT_ROOT/cpt_greek_heldout_500_20260522.jsonl}"
MANIFEST_JSON="${MANIFEST_JSON:-$OUT_ROOT/cpt_greek_heldout_500_20260522.manifest.json}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/eval "$OUT_ROOT"

echo "=== build_cpt_heldout_jsonl.sbatch ==="
date -u
echo "host:             $(hostname)"
echo "SCRIPT_DIR:       $SCRIPT_DIR"
echo "SELECTED_PARQUET: $SELECTED_PARQUET"
echo "BULK_MIX_JSONL:   $BULK_MIX_JSONL"
echo "RECIPE_JSON:      $RECIPE_JSON"
echo "OUTPUT_JSONL:     $OUTPUT_JSONL"
echo "MANIFEST_JSON:    $MANIFEST_JSON"
echo "UENV_IMAGE:       $UENV_IMAGE"
echo "PYTHONPATH_EXTRA: $PYTHONPATH_EXTRA"
echo

uenv run "$UENV_IMAGE" --view=default -- bash <<INNER
set -euo pipefail
export PYTHONPATH="$PYTHONPATH_EXTRA"
python3 "$SCRIPT_DIR/build_cpt_heldout_jsonl.py" \
    --selected-parquet "$SELECTED_PARQUET" \
    --bulk-mix-jsonl "$BULK_MIX_JSONL" \
    --recipe-json "$RECIPE_JSON" \
    --output-jsonl "$OUTPUT_JSONL" \
    --manifest-json "$MANIFEST_JSON"
INNER

echo
echo "=== held-out build complete ==="
date -u
wc -l "$OUTPUT_JSONL"
ls -lh "$OUTPUT_JSONL" "$MANIFEST_JSON"

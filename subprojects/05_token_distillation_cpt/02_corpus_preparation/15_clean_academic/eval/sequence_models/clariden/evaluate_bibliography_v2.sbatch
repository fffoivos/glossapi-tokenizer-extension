#!/usr/bin/env bash
# CPU-only, comparison-only evaluation of explicit bibliography features.
# Run train while developing rules; run validation only after the rules freeze.
# The historically named test partition is never an accepted CLI choice.
#SBATCH --job-name=cpt_bib_v2_eval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

REPO_ROOT=${REPO_ROOT:?set the exact clean checkout}
SILVER=${SILVER:?set the physically test-stripped STRUCT-2K LLM-silver JSONL}
SPLIT=${SPLIT:?set train while developing or validation after rules freeze}
OUTPUT=${OUTPUT:?set a new immutable report path}
UENV=${UENV:-pytorch/v2.9.1:v2}

case "$SPLIT" in
    train|validation) ;;
    *) echo "ERROR: only train or validation is permitted" >&2; exit 90 ;;
esac
[[ -f "$SILVER" && ! -L "$SILVER" ]] || {
    echo "ERROR: SILVER must be a real file: $SILVER" >&2
    exit 91
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
    echo "ERROR: immutable output already exists: $OUTPUT" >&2
    exit 92
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "ERROR: REPO_ROOT is not clean" >&2
    exit 93
}

EVAL_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
test -f "$EVAL_DIR/sequence_models/bibliography_v2_eval.py"
mkdir -p "$(dirname "$OUTPUT")"
cd "$EVAL_DIR"

echo "BIB_V2_COMPARISON split=$SPLIT commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "BIB_V2_COMPARISON accelerator=hidden corpus_mutation=none test_access=forbidden"
uenv run "$UENV" --view=default -- \
    env CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 \
    python3 -m sequence_models.bibliography_v2_eval \
        --silver "$SILVER" \
        --split "$SPLIT" \
        --output "$OUTPUT"

python3 - "$OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
assert report["status"] == "comparison_only"
assert report["split"] in {"train", "validation"}
assert report["claims"]["test_or_sealed_data_accessed"] is False
assert report["claims"]["corpus_mutated"] is False
assert report["code_revision"]["clean"] is True
print(f"BIB_V2_REPORT={path}")
print(f"BIB_V2_REPORT_SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY

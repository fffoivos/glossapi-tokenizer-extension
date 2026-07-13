#!/usr/bin/env bash
# Build the unweighted, source-balanced bibliography line-feature explorer.
# Corpus scanning happens on the worker; the MacBook only receives the result.
#SBATCH --job-name=cpt_bib_feature_site
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
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set the exact clean commit}
INPUT=${INPUT:?set the physically test-stripped STRUCT-2K LLM-silver JSONL}
OUTPUT_DIR=${OUTPUT_DIR:?set a new immutable output directory}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
    echo "ERROR: commit mismatch" >&2
    exit 90
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
    echo "ERROR: checkout is dirty" >&2
    exit 91
}
[[ -f "$INPUT" && ! -L "$INPUT" ]] || {
    echo "ERROR: INPUT must be a real file: $INPUT" >&2
    exit 92
}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
    echo "ERROR: immutable output already exists: $OUTPUT_DIR" >&2
    exit 93
}

EVAL_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
test -f "$EVAL_DIR/sequence_models/bibliography_feature_explorer.py"
mkdir -p "$OUTPUT_DIR"
trap 'rm -rf "$OUTPUT_DIR"' EXIT

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0

cd "$EVAL_DIR"
uenv run "$UENV" --view=default -- \
    python3 -m sequence_models.bibliography_feature_explorer \
        --input "$INPUT" \
        --output "$OUTPUT_DIR/index.html" \
        --receipt "$OUTPUT_DIR/build.receipt.json" \
        --document-count 20 \
        --sources greek_phd,kallipos,openarchives \
        --split train \
        --coverage full_document \
        --seed bibliography-feature-explorer-20260713

python3 - "$OUTPUT_DIR/build.receipt.json" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
receipt = json.loads(path.read_text(encoding="utf-8"))
assert receipt["status"] == "passed"
assert receipt["feature_count"] == 37
assert receipt["selection"]["document_count"] == 20
assert receipt["selection"]["source_counts"] == {
    "greek_phd": 7,
    "kallipos": 7,
    "openarchives": 6,
}
receipt["code_commit"] = sys.argv[2]
receipt["slurm_job_id"] = sys.argv[3]
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

trap - EXIT
echo "BIBLIOGRAPHY_FEATURE_EXPLORER=$OUTPUT_DIR/index.html"

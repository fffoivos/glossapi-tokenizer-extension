#!/usr/bin/env bash
# Materialize lossless bibliography match/nonmatch geometry on Clariden CPU.
#SBATCH --job-name=bib_pos_table
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
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact source commit}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?pin STRUCT-2K input}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen bibliography entry table}
OUTPUT_DIR=${OUTPUT_DIR:?set new positional table directory}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT/sequence_models" && ! -L "$CODE_ROOT" ]] || exit 90
[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 91
[[ -f "$BASE_TABLE_DIR/manifest.json" && ! -L "$BASE_TABLE_DIR" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93
test -f "$CODE_ROOT/sequence_models/bibliography_positional_table.py"

trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_positional_table \
    --input "$INPUT" \
    --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
    --base-table-dir "$BASE_TABLE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --split train \
    --workers "$SLURM_CPUS_PER_TASK" \
    --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"

python3 - "$OUTPUT_DIR/receipt.json" "$EXPECTED_COMMIT" <<'PY'
import json, pathlib, sys
receipt=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["status"] == "passed_geometry_only_targets_external"
assert receipt["code_commit"] == sys.argv[2]
assert receipt["line_count"] == 939014
assert receipt["document_count"] == 1118
assert receipt["count_span_parity_with_base"] is True
PY
trap - EXIT
echo "BIBLIOGRAPHY_POSITIONAL_TABLE=$OUTPUT_DIR"

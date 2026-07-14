#!/usr/bin/env bash
# Prediction-blind all-line role profile and source-balanced 60-block pilot.
#SBATCH --job-name=bib_role_profile
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set the immutable code-bundle root containing sequence_models}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set the exact clean commit}
INPUT=${INPUT:?set the physically test-stripped STRUCT-2K LLM-silver JSONL}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?pin the exact STRUCT-2K input SHA-256}
OUTPUT_DIR=${OUTPUT_DIR:?set a new immutable output directory}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
    echo "ERROR: CODE_ROOT must be a real directory: $CODE_ROOT" >&2
    exit 90
}
[[ -f "$CODE_ROOT/bundle.receipt.json" && ! -L "$CODE_ROOT/bundle.receipt.json" ]] || {
    echo "ERROR: code bundle receipt is missing" >&2
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

ROLE_CONTRACT="$CODE_ROOT/sequence_models/bibliography_role_contract_v1.json"
test -f "$CODE_ROOT/sequence_models/bibliography_role_profile.py"
test -f "$CODE_ROOT/sequence_models/bibliography_positional_features.py"
test -f "$ROLE_CONTRACT"

python3 - "$CODE_ROOT" "$EXPECTED_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
receipt = json.loads((root / "bundle.receipt.json").read_text(encoding="utf-8"))
assert receipt["schema_version"] == "bibliography-role-code-bundle-v1"
assert receipt["code_commit"] == sys.argv[2]
for relative, expected in receipt["files"].items():
    path = root / relative
    assert path.is_file() and not path.is_symlink(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
PY

trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
    env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_role_profile \
        --input "$INPUT" \
        --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
        --output-dir "$OUTPUT_DIR" \
        --role-contract "$ROLE_CONTRACT" \
        --split train \
        --coverage full_document \
        --sources greek_phd,kallipos,openarchives \
        --blocks-per-source 20 \
        --context-lines 5 \
        --workers 16 \
        --seed bibliography-role-pilot-v1 \
        --code-commit "$EXPECTED_COMMIT" \
        --slurm-job-id "$SLURM_JOB_ID"

python3 - "$OUTPUT_DIR/receipt.json" "$EXPECTED_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
receipt = json.loads(path.read_text(encoding="utf-8"))
assert receipt["status"] == "passed_prediction_blind_profile"
assert receipt["code_commit"] == sys.argv[2]
assert receipt["slurm_job_id"] == sys.argv[3]
assert receipt["selected_block_count"] == 60
assert receipt["prediction_inputs_loaded"] is False
assert receipt["model_outcomes_used_for_selection"] is False
packet = json.loads((path.parent / "role_review_packet.blind.json").read_text(encoding="utf-8"))
assert packet["selection"]["source_counts"] == {
    "greek_phd": 20,
    "kallipos": 20,
    "openarchives": 20,
}
assert all(packet["blinding"].values())
PY

trap - EXIT
echo "BIBLIOGRAPHY_ROLE_PROFILE=$OUTPUT_DIR"

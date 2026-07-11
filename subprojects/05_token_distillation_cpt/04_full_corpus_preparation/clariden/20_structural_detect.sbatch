#!/usr/bin/env bash
#SBATCH --job-name=cpt4_struct
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

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh to export the original script directory}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "academic structural detection"

INPUT=${INPUT:?set INPUT to canonical Parquet path/dir/glob}
INPUT_RECEIPT=${INPUT_RECEIPT:?set INPUT_RECEIPT to the passed acquisition receipt for INPUT}
SOURCE=${SOURCE:?set SOURCE}
INPUT_SCOPE=${INPUT_SCOPE:?set INPUT_SCOPE to single_source or canonical_mixed}
TEXT_COLUMN=${TEXT_COLUMN:-text}
ID_COLUMN=${ID_COLUMN:-source_doc_id}
SOURCE_COLUMN=${SOURCE_COLUMN:-source_dataset}
SOURCE_REGEX=${SOURCE_REGEX:-}
PARITY_RECEIPT=${PARITY_RECEIPT:?set PARITY_RECEIPT to a passed full-heldout receipt for this binary}
DETECTOR_BUILD_RECEIPT=${DETECTOR_BUILD_RECEIPT:?set DETECTOR_BUILD_RECEIPT to the receipt for REFERENCE_BIN}
REFERENCE_BIN=${REFERENCE_BIN:?set REFERENCE_BIN to the immutable published detector binary}
RUN_TAG=${RUN_TAG:-struct_${SOURCE}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$RUN_ROOT/$RUN_TAG}

phase04_require_uenv_python
phase04_require_file "$REFERENCE_BIN"
phase04_require_file "$PARITY_RECEIPT"
phase04_require_file "$INPUT_RECEIPT"
phase04_require_file "$DETECTOR_BUILD_RECEIPT"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
[[ ! -e "$OUT" ]] || { echo "ERROR: refusing to overwrite detector run: $OUT" >&2; exit 88; }
PARTIAL="${OUT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$PARTIAL" ]] || { echo "ERROR: stale partial run exists: $PARTIAL" >&2; exit 89; }
mkdir -p "$(dirname "$OUT")" "$PARTIAL"
CODE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
SOURCES_SHA256=$(sha256sum "$SOURCE_CONFIG" | awk '{print $1}')
POLICY_SHA256=$(sha256sum "$CLEANING_POLICY" | awk '{print $1}')
DETECTOR_BINARY_SHA256=$(sha256sum "$REFERENCE_BIN" | awk '{print $1}')
INPUT_RECEIPT_SHA256=$(sha256sum "$INPUT_RECEIPT" | awk '{print $1}')
DETECTOR_BUILD_RECEIPT_SHA256=$(sha256sum "$DETECTOR_BUILD_RECEIPT" | awk '{print $1}')
export CODE_COMMIT SOURCES_SHA256 POLICY_SHA256 TOKENIZER_SHA256 DETECTOR_BINARY_SHA256
export INPUT_RECEIPT_SHA256 DETECTOR_BUILD_RECEIPT_SHA256
phase04_print_runtime | tee "$PARTIAL/runtime.env"

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/validate_input_receipt.py" \
    --receipt "$INPUT_RECEIPT" \
    --sources "$SOURCE_CONFIG" \
    --source "$SOURCE" \
    --input "$INPUT" \
    --text-column "$TEXT_COLUMN" \
    --id-column "$ID_COLUMN" \
    --source-column "$SOURCE_COLUMN" \
    | tee "$PARTIAL/input_receipt_check.json"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/validate_detector_build_receipt.py" \
    --receipt "$DETECTOR_BUILD_RECEIPT" \
    --repo "$REPO_ROOT" \
    --binary "$REFERENCE_BIN" \
    --cargo-lock "$ACADEMIC_DIR/reference_detector/Cargo.lock" \
    --cargo-toml "$ACADEMIC_DIR/reference_detector/Cargo.toml" \
    | tee "$PARTIAL/detector_build_check.json"

route_args=(
    --sources "$SOURCE_CONFIG"
    --source "$SOURCE"
    --input-scope "$INPUT_SCOPE"
    --source-regex "$SOURCE_REGEX"
    --text-column "$TEXT_COLUMN"
    --id-column "$ID_COLUMN"
    --source-column "$SOURCE_COLUMN"
)
[[ "${ALLOW_SHADOW:-0}" == "1" ]] && route_args+=(--allow-shadow)
[[ "${SEGMENTATION_COMPLETE:-0}" == "1" ]] && route_args+=(--segmentation-complete)
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/check_source_route.py" "${route_args[@]}" \
    | tee "$PARTIAL/source_route.json"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/validate_parity_receipt.py" \
    --receipt "$PARITY_RECEIPT" --binary "$REFERENCE_BIN" --policy "$CLEANING_POLICY" \
    | tee "$PARTIAL/parity_check.json"

stream_args=(
    --input "$INPUT"
    --text-column "$TEXT_COLUMN"
    --id-column "$ID_COLUMN"
    --source-column "$SOURCE_COLUMN"
    --source-name "$SOURCE"
    --manifest-out "$PARTIAL/input_stream_manifest.json"
)
if [[ -n "$SOURCE_REGEX" ]]; then
    stream_args+=(--source-regex "$SOURCE_REGEX")
fi
if [[ -n "${MAX_ROWS:-}" ]]; then
    stream_args+=(--max-rows "$MAX_ROWS")
fi

export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-240}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export REFERENCE_BIN SOURCE PARTIAL
uenv run "$PHASE04_UENV" --view=default -- bash -c '
set -euo pipefail
"$1" "$2" "${@:3}" |
  RAYON_NUM_THREADS="$RAYON_NUM_THREADS" "$REFERENCE_BIN" \
    --mode structure-spans --source "$SOURCE" --input - --text-field text --id-field id \
    --out-spans "$PARTIAL/structure_spans.jsonl" --out-counters "$PARTIAL/structure_counters.jsonl"
' bash "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/stream_parquet_jsonl.py" "${stream_args[@]}"

phase04_require_file "$PARTIAL/input_stream_manifest.json"
phase04_require_file "$PARTIAL/structure_counters.jsonl"
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/validate_detector_run.py" \
    --source "$SOURCE" \
    --stream-manifest "$PARTIAL/input_stream_manifest.json" \
    --counters "$PARTIAL/structure_counters.jsonl" \
    --spans "$PARTIAL/structure_spans.jsonl" \
    --binary "$REFERENCE_BIN" \
    --parity-receipt "$PARITY_RECEIPT" \
    --input-receipt "$INPUT_RECEIPT" \
    --input-receipt-check "$PARTIAL/input_receipt_check.json" \
    --detector-build-receipt "$DETECTOR_BUILD_RECEIPT" \
    --sources-config "$SOURCE_CONFIG" \
    --cleaning-policy "$CLEANING_POLICY" \
    --code-commit "$CODE_COMMIT" \
    --output "$PARTIAL/detector_run_manifest.json"
sha256sum "$PARTIAL"/*.json "$PARTIAL"/*.jsonl > "$PARTIAL/outputs.sha256"
touch "$PARTIAL/COMPLETED"
mv "$PARTIAL" "$OUT"
echo "AUDIT_COMPLETE=$OUT"

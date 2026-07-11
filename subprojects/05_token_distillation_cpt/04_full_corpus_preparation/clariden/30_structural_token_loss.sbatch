#!/usr/bin/env bash
#SBATCH --job-name=cpt4_tokloss
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
phase04_require_compute_partition "exact structural token-loss audit"

INPUT=${INPUT:?set INPUT to canonical Parquet path/dir/glob}
INPUT_RECEIPT=${INPUT_RECEIPT:?set INPUT_RECEIPT to the passed acquisition receipt for INPUT}
INPUT_SCOPE=${INPUT_SCOPE:?set INPUT_SCOPE to single_source or canonical_mixed}
SPANS=${SPANS:?set SPANS to structure_spans.jsonl}
DETECTOR_RUN_MANIFEST=${DETECTOR_RUN_MANIFEST:?set DETECTOR_RUN_MANIFEST to detector_run_manifest.json}
SOURCE_NAME=${SOURCE_NAME:?set SOURCE_NAME to the tracked structural route id}
SOURCE_REGEX=${SOURCE_REGEX:-}
TEXT_COLUMN=${TEXT_COLUMN:-text}
ID_COLUMN=${ID_COLUMN:-source_doc_id}
SOURCE_COLUMN=${SOURCE_COLUMN:-source_dataset}
RUN_TAG=${RUN_TAG:-struct_token_loss_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$RUN_ROOT/$RUN_TAG}

phase04_require_uenv_python
phase04_require_file "$TOKENIZER_JSON"
phase04_require_file "$INPUT_RECEIPT"
phase04_require_file "$SPANS"
phase04_require_file "$DETECTOR_RUN_MANIFEST"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
actual_tokenizer_sha=$(sha256sum "$TOKENIZER_JSON" | awk '{print $1}')
[[ "$actual_tokenizer_sha" == "$TOKENIZER_SHA256" ]] || {
    echo "ERROR: tokenizer hash mismatch: $actual_tokenizer_sha" >&2
    exit 90
}
[[ ! -e "$OUT" ]] || { echo "ERROR: refusing to overwrite token audit: $OUT" >&2; exit 91; }
PARTIAL="${OUT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$PARTIAL" ]] || { echo "ERROR: stale partial token audit exists: $PARTIAL" >&2; exit 92; }
mkdir -p "$(dirname "$OUT")" "$PARTIAL"
CODE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
SOURCES_SHA256=$(sha256sum "$SOURCE_CONFIG" | awk '{print $1}')
POLICY_SHA256=$(sha256sum "$CLEANING_POLICY" | awk '{print $1}')
INPUT_RECEIPT_SHA256=$(sha256sum "$INPUT_RECEIPT" | awk '{print $1}')
export CODE_COMMIT SOURCES_SHA256 POLICY_SHA256 TOKENIZER_SHA256 INPUT_RECEIPT_SHA256
phase04_print_runtime | tee "$PARTIAL/runtime.env"

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/validate_input_receipt.py" \
    --receipt "$INPUT_RECEIPT" \
    --sources "$SOURCE_CONFIG" \
    --source "$SOURCE_NAME" \
    --input "$INPUT" \
    --text-column "$TEXT_COLUMN" \
    --id-column "$ID_COLUMN" \
    --source-column "$SOURCE_COLUMN" \
    | tee "$PARTIAL/input_receipt_check.json"

route_args=(
    --sources "$SOURCE_CONFIG"
    --source "$SOURCE_NAME"
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

args=(
    --input "$INPUT"
    --spans "$SPANS"
    --tokenizer-json "$TOKENIZER_JSON"
    --expected-tokenizer-sha256 "$TOKENIZER_SHA256"
    --text-column "$TEXT_COLUMN"
    --id-column "$ID_COLUMN"
    --source-column "$SOURCE_COLUMN"
    --source-name "$SOURCE_NAME"
    --detector-run-manifest "$DETECTOR_RUN_MANIFEST"
    --input-receipt "$INPUT_RECEIPT"
    --output-dir "$PARTIAL"
)
if [[ -n "$SOURCE_REGEX" ]]; then
    args+=(--source-regex "$SOURCE_REGEX")
fi
if [[ -n "${MAX_ROWS:-}" ]]; then
    args+=(--max-rows "$MAX_ROWS")
fi

export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-240}
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/structural_token_loss.py" "${args[@]}"

sha256sum "$PARTIAL"/*.json "$PARTIAL"/*.jsonl "$PARTIAL"/*.csv "$PARTIAL"/*.parquet \
    > "$PARTIAL/outputs.sha256"
touch "$PARTIAL/COMPLETED"
mv "$PARTIAL" "$OUT"
echo "AUDIT_COMPLETE=$OUT"

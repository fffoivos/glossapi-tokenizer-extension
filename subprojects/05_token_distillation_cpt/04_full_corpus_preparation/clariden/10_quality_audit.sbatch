#!/usr/bin/env bash
#SBATCH --job-name=cpt4_quality
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
phase04_require_compute_partition "source-quality audit"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

INPUT=${INPUT:?set INPUT to a normalized Parquet path/dir/glob}
INPUT_RECEIPT=${INPUT_RECEIPT:?set INPUT_RECEIPT to the passed acquisition receipt for INPUT}
SOURCE=${SOURCE:?set SOURCE}
TEXT_COLUMN=${TEXT_COLUMN:?set TEXT_COLUMN}
ID_COLUMN=${ID_COLUMN:?set ID_COLUMN}
SOURCE_COLUMN=${SOURCE_COLUMN:-source_dataset}
METADATA_COLUMN=${METADATA_COLUMN:-}
WORKERS=${WORKERS:-128}
RUN_TAG=${RUN_TAG:-quality_${SOURCE}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-$RUN_ROOT/$RUN_TAG}

phase04_require_uenv_python
phase04_require_file "$INPUT_RECEIPT"
[[ ! -e "$OUT" ]] || { echo "ERROR: refusing to overwrite quality audit: $OUT" >&2; exit 88; }
PARTIAL="${OUT}.partial-${SLURM_JOB_ID}"
[[ ! -e "$PARTIAL" ]] || { echo "ERROR: stale partial quality audit exists: $PARTIAL" >&2; exit 89; }
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
    --source "$SOURCE" \
    --input "$INPUT" \
    --text-column "$TEXT_COLUMN" \
    --id-column "$ID_COLUMN" \
    --source-column "$SOURCE_COLUMN" \
    | tee "$PARTIAL/input_receipt_check.json"

args=(
    --input "$INPUT"
    --source "$SOURCE"
    --text-column "$TEXT_COLUMN"
    --id-column "$ID_COLUMN"
    --workers "$WORKERS"
    --output-dir "$PARTIAL"
)
if [[ -n "$METADATA_COLUMN" ]]; then
    args+=(--metadata-column "$METADATA_COLUMN")
fi
if [[ -n "${MAX_ROW_GROUPS:-}" ]]; then
    args+=(--max-row-groups "$MAX_ROW_GROUPS")
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=4
uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/profile_source_quality.py" "${args[@]}"

sha256sum "$PARTIAL"/*.json "$PARTIAL"/*.jsonl "$PARTIAL"/*.csv > "$PARTIAL/outputs.sha256"
touch "$PARTIAL/COMPLETED"
mv "$PARTIAL" "$OUT"
echo "AUDIT_COMPLETE=$OUT"

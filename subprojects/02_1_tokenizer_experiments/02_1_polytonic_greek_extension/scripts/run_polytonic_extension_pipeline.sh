#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/foivos/Projects/glossapi-tokenizer-extension}"
PROJECT="$REPO/subprojects/02_1_tokenizer_experiments/02_1_polytonic_greek_extension"
TRAINING="$REPO/subprojects/02_1_tokenizer_experiments/02_1_1_tokenizer_training"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID="${RUN_ID:-c3p_polytonic_$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_ROOT="${WORK_ROOT:-/home/foivos/data/glossapi_work/polytonic_extension/c3p_runs/$RUN_ID}"

INPUT_PARQUET="${INPUT_PARQUET:-/home/foivos/data/glossapi_work/polytonic_extension/strict_w050_c010/training_data/polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet}"
C3_BASE="${C3_BASE:-$REPO/subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded}"

mkdir -p "$WORK_ROOT"/{logs,splits,training,variants,eval,report}

echo "RUN_ID=$RUN_ID"
echo "WORK_ROOT=$WORK_ROOT"
echo "INPUT_PARQUET=$INPUT_PARQUET"
echo "C3_BASE=$C3_BASE"

"$PYTHON_BIN" "$PROJECT/scripts/build_polytonic_splits.py" \
  --input-parquet "$INPUT_PARQUET" \
  --output-dir "$WORK_ROOT/splits" \
  2>&1 | tee "$WORK_ROOT/logs/01_splits.log"

"$PYTHON_BIN" "$TRAINING/scripts/train_continuous_bpe_tokenizer.py" \
  --base-tokenizer-dir "$C3_BASE" \
  --reference-tokenizer "$C3_BASE" \
  --input-glob "$WORK_ROOT/splits/poly_train.parquet" \
  --output-dir "$WORK_ROOT/training" \
  --target-vocab-size 153600 \
  --text-column text \
  --checkpoint-every 256 \
  --name "$RUN_ID" \
  --skip-identity-check \
  2>&1 | tee "$WORK_ROOT/logs/02_training.log"

"$PYTHON_BIN" "$PROJECT/scripts/build_polytonic_cutoff_variants.py" \
  --base-tokenizer-dir "$C3_BASE" \
  --full-tokenizer-dir "$WORK_ROOT/training/tokenizer" \
  --output-dir "$WORK_ROOT/variants" \
  --step 512 \
  --max-added 5120 \
  2>&1 | tee "$WORK_ROOT/logs/03_variants.log"

SLICE_ARGS=(
  --slice "poly_val_balanced=$WORK_ROOT/splits/poly_val_balanced.parquet"
  --slice "poly_test_balanced=$WORK_ROOT/splits/poly_test_balanced.parquet"
  --slice "poly_high_diacritic_test=$WORK_ROOT/splits/poly_high_diacritic_test.parquet"
  --slice "poly_underaccented_test=$WORK_ROOT/splits/poly_underaccented_test.parquet"
)
while IFS= read -r -d '' f; do
  name="$(basename "$f" .parquet)"
  SLICE_ARGS+=(--slice "poly_${name}=$f")
done < <(find "$WORK_ROOT/splits/by_source" -name '*_test.parquet' -print0 | sort -z)

if [[ -n "${MODERN_C3_VAL_CLEAN:-}" && -f "${MODERN_C3_VAL_CLEAN:-}" ]]; then
  SLICE_ARGS+=(--slice "modern_c3_val_clean=$MODERN_C3_VAL_CLEAN")
fi
if [[ -n "${MODERN_C3_TEST_CLEAN:-}" && -f "${MODERN_C3_TEST_CLEAN:-}" ]]; then
  SLICE_ARGS+=(--slice "modern_c3_test_clean=$MODERN_C3_TEST_CLEAN")
fi
if [[ -n "${FINEWEB2_GRC_REFERENCE:-}" && -f "${FINEWEB2_GRC_REFERENCE:-}" ]]; then
  SLICE_ARGS+=(--slice "fineweb2_grc_reference=$FINEWEB2_GRC_REFERENCE")
fi

"$PYTHON_BIN" "$PROJECT/scripts/evaluate_polytonic_variants.py" \
  --variants-manifest "$WORK_ROOT/variants/variants_manifest.json" \
  "${SLICE_ARGS[@]}" \
  --output-dir "$WORK_ROOT/eval" \
  2>&1 | tee "$WORK_ROOT/logs/04_eval.log"

"$PYTHON_BIN" "$PROJECT/scripts/render_polytonic_eval_report.py" \
  --metrics-csv "$WORK_ROOT/eval/metrics_by_slice.csv" \
  --output-dir "$WORK_ROOT/report" \
  --report-path "$WORK_ROOT/report/REPORT.md" \
  2>&1 | tee "$WORK_ROOT/logs/05_report.log"

cat > "$WORK_ROOT/RUN_COMPLETE.json" <<EOF
{
  "run_id": "$RUN_ID",
  "work_root": "$WORK_ROOT",
  "completed_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "report": "$WORK_ROOT/report/REPORT.md",
  "variants_manifest": "$WORK_ROOT/variants/variants_manifest.json",
  "metrics": "$WORK_ROOT/eval/metrics_by_slice.csv"
}
EOF

cat "$WORK_ROOT/RUN_COMPLETE.json"

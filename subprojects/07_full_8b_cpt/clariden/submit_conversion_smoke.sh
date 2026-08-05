#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_BENCHMARK_ROOT:?set completed benchmark root}"
: "${FULL8_PRELAUNCH_ROOT:?set prelaunch root}"
: "${FULL8_SELECTED_PROFILE:?set selected profile receipt}"
DRY_RUN=${DRY_RUN:-1}
EVALUATION_BUNDLE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments"
NATIVE_ROOT="$FULL8_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
COMPAT=${FULL8_PYTHON_COMPAT_DIR:-/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260802T230000Z-mini-b2-v8/compat}
FULL8_EVALUATION_ROOT="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation"
CLEAN_SUBSET=${FULL8_GREEKMMLU_CLEAN_SUBSET:-/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_subset_manifest.json}
TOKENIZER_SHA=bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b

profile=$(python3 - "$FULL8_SELECTED_PROFILE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selection"]["profile_id"])
PY
)
case "$profile" in
  dp64_32node) source="$FULL8_BENCHMARK_ROOT/candidate_dp64/checkpoints" ;;
  dp32_16node) source="$FULL8_BENCHMARK_ROOT/control_dp32/checkpoints" ;;
  *) echo "unknown selected profile" >&2; exit 2 ;;
esac
root="$FULL8_PRELAUNCH_ROOT/conversion_smoke"
submit() { if [[ "$DRY_RUN" == 1 ]]; then { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY_JOB; else sbatch --parsable "$@"; fi; }
if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$root" ]] || { echo "conversion smoke root exists" >&2; exit 2; }
  mkdir -p "$root/export" "$root/greekmmlu" "$FULL8_PRELAUNCH_ROOT/logs"
fi
convert=$(submit --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,EVALUATION_CODE_BUNDLE_ROOT=$FULL8_CODE_ROOT,EVALUATION_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,MEGATRON_DIR=$MEGATRON,SOURCE_CHECKPOINT_ROOT=$source,SOURCE_ITERATION=160,TOKENIZER_DIR=$TOKENIZER,EXPORT_ROOT=$root/export,PYTHON_COMPAT_DIR=$COMPAT,TOKENIZER_SHA=$TOKENIZER_SHA,EXPORT_MODEL_SCALE=8B,FULL8_EVALUATION_ROOT=$FULL8_EVALUATION_ROOT" \
  "$EVALUATION_BUNDLE/clariden/convert_checkpoint_for_native_greekmmlu.sbatch")
evaluate=$(submit --time=01:15:00 --dependency="afterok:$convert" --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,NATIVE_GREEK_EVAL_ROOT=$NATIVE_ROOT,EXPORT_ROOT=$root/export,GREEKMMLU_ROOT=$root/greekmmlu,MODEL_LABEL=full8b_conversion_smoke,EVALUATION_NAMESPACE=full8b_conversion_smoke,EVAL_DTYPE=float32" \
  "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu.sbatch")
receipt=$(submit --dependency="afterok:$evaluate" --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,EXPORT_RECEIPT=$root/export/checkpoint_eval_export_receipt.json,GREEKMMLU_ROOT=$root/greekmmlu,MODEL_LABEL=full8b_conversion_smoke,OUTPUT_RECEIPT=$root/exact_checkpoint_native_greekmmlu_receipt.json,GREEKMMLU_CLEAN_SUBSET=$CLEAN_SUBSET,EVALUATION_NAMESPACE=full8b_conversion_smoke" \
  "$EVALUATION_BUNDLE/clariden/finalize_checkpoint_greekmmlu.sbatch")
gate=$(submit --dependency="afterok:$receipt" --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_CONVERSION_SMOKE_ROOT=$root" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_conversion_smoke.sbatch")
printf '{"convert":"%s","evaluate":"%s","receipt":"%s","gate":"%s"}\n' "$convert" "$evaluate" "$receipt" "$gate"

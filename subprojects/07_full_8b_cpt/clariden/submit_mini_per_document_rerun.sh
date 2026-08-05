#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${MINI_DOCVAL_ROOT:?set new rerun output root}"
DRY_RUN=${DRY_RUN:-1}
MINI_VALIDATION_MANIFEST=${MINI_VALIDATION_MANIFEST:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/dataset_scheduling_0p5b/20260802T221000Z-neutral-gpp-v4/validation-frozen-v1/validation_manifest.json}
MINI_ENDPOINT_ROOT=${MINI_ENDPOINT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z/evaluations_fp32_v1/iteration_0038496/attempt_3/tasks}
MINI_TOKENIZER=${MINI_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_v1_1_0p5b_greek_overlay_fcd33ec}
manifest="$MINI_DOCVAL_ROOT/validation_manifest.json"

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}
if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$MINI_DOCVAL_ROOT" ]] || { echo "rerun root exists" >&2; exit 2; }
  mkdir -p "$MINI_DOCVAL_ROOT/models" "$MINI_DOCVAL_ROOT/logs"
fi
manifest_job=$(submit \
  --output="$MINI_DOCVAL_ROOT/logs/%x-%j.out" --error="$MINI_DOCVAL_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,MINI_VALIDATION_MANIFEST=$MINI_VALIDATION_MANIFEST,MINI_DOCVAL_MANIFEST=$manifest" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/build_mini_per_document_manifest.sbatch")
arms=(D0_mixed D1_hard_h_to_g D2_hard_g_to_h D3_gradual_h_to_g D4_gradual_g_to_h)
smoke=$(submit --dependency="afterok:$manifest_job" \
  --output="$MINI_DOCVAL_ROOT/logs/%x-%j.out" --error="$MINI_DOCVAL_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,MINI_DOCVAL_MANIFEST=$manifest,MINI_SMOKE_MODEL=$MINI_ENDPOINT_ROOT/D0_mixed/export/hf,MINI_TOKENIZER=$MINI_TOKENIZER,MINI_DOCVAL_ROOT=$MINI_DOCVAL_ROOT" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_mini_per_document_smoke.sbatch")
jobs=()
for arm in "${arms[@]}"; do
  model="$MINI_ENDPOINT_ROOT/$arm/export/hf"
  job=$(submit --dependency="afterok:$smoke" \
    --output="$MINI_DOCVAL_ROOT/logs/%x-%A_%a.out" --error="$MINI_DOCVAL_ROOT/logs/%x-%A_%a.err" \
    --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_VALIDATION_MANIFEST=$manifest,FULL8_HF_MODEL=$model,FULL8_HF_TOKENIZER=$MINI_TOKENIZER,FULL8_DOCVAL_OUTPUT=$MINI_DOCVAL_ROOT/models/$arm" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")
  jobs+=("$job")
done
dependency="afterok:$(IFS=:; echo "${jobs[*]}")"
finalize=$(submit --dependency="$dependency" \
  --output="$MINI_DOCVAL_ROOT/logs/%x-%j.out" --error="$MINI_DOCVAL_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,MINI_DOCVAL_ROOT=$MINI_DOCVAL_ROOT,MINI_DOCVAL_MANIFEST=$manifest" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_mini_per_document_comparison.sbatch")
python3 - "$manifest_job" "$smoke" "$finalize" "${jobs[@]}" <<'PY'
import json,sys
print(json.dumps({"manifest_job":sys.argv[1],"smoke_job":sys.argv[2],"model_arrays":sys.argv[4:],"finalizer":sys.argv[3]},separators=(",",":")))
PY

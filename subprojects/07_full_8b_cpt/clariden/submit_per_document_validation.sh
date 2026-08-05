#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${FULL8_HF_MODEL:?set one HF model/checkpoint path}"
: "${FULL8_HF_TOKENIZER:?set tokenizer path}"
: "${FULL8_DOCVAL_OUTPUT:?set checkpoint-specific output directory}"
DRY_RUN=${DRY_RUN:-1}
DEPENDENCY=${DEPENDENCY:-}
args=(--export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_VALIDATION_MANIFEST=$FULL8_VALIDATION_MANIFEST,FULL8_HF_MODEL=$FULL8_HF_MODEL,FULL8_HF_TOKENIZER=$FULL8_HF_TOKENIZER,FULL8_DOCVAL_OUTPUT=$FULL8_DOCVAL_OUTPUT")
[[ -z "$DEPENDENCY" ]] || args+=(--dependency="$DEPENDENCY")
args+=("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'sbatch'; printf ' %q' "${args[@]}"; printf '\n'
else
  sbatch --parsable "${args[@]}"
fi

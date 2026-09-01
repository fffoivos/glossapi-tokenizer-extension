#!/usr/bin/env bash
set -euo pipefail
: "${HFV2_CODE_ROOT:?}"
: "${HFV2_RUN_ROOT:?}"
: "${HFV2_INPUT_ROOT:?}"
: "${HFV2_DATA_PYTHON:?}"
: "${HFV2_HF_PYTHON:?}"
: "${HFV2_HF_TOKEN_FILE:?}"
: "${HFV2_TOKENIZER_JSON:?}"
: "${HFV2_DEDUP_RECEIPT:?}"
: "${HFV2_SOURCE_BREAKDOWN:?}"
: "${HFV2_STAGE:?set to prepare, canary-a, canary-b, transform, finalize, or publish}"
root="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/clariden"
export_args="ALL,HFV2_CODE_ROOT=$HFV2_CODE_ROOT,HFV2_RUN_ROOT=$HFV2_RUN_ROOT,HFV2_INPUT_ROOT=$HFV2_INPUT_ROOT,HFV2_DATA_PYTHON=$HFV2_DATA_PYTHON,HFV2_HF_PYTHON=$HFV2_HF_PYTHON,HFV2_HF_TOKEN_FILE=$HFV2_HF_TOKEN_FILE,HFV2_TOKENIZER_JSON=$HFV2_TOKENIZER_JSON,HFV2_DEDUP_RECEIPT=$HFV2_DEDUP_RECEIPT,HFV2_SOURCE_BREAKDOWN=$HFV2_SOURCE_BREAKDOWN"
dependency_args=()
if [[ -n ${HFV2_DEPENDENCY:-} ]]; then dependency_args=(--dependency="afterok:$HFV2_DEPENDENCY"); fi
case "$HFV2_STAGE" in
  prepare) script=prepare.sbatch; extra=() ;;
  canary-a) script=transform.sbatch; extra=(--export="$export_args,HFV2_TASK_INDEX=396") ;;
  canary-b) script=transform.sbatch; extra=(--export="$export_args,HFV2_TASK_INDEX=418") ;;
  transform) script=transform_batch.sbatch; extra=() ;;
  finalize) script=finalize.sbatch; extra=() ;;
  publish) script=publish.sbatch; extra=() ;;
  *) echo "unknown HFV2_STAGE: $HFV2_STAGE" >&2; exit 2 ;;
esac
if ((${#extra[@]} == 0)); then extra=(--export="$export_args"); fi
command=(sbatch --parsable "${dependency_args[@]}" "${extra[@]}" "$root/$script")
if [[ ${HFV2_DRY_RUN:-0} == 1 ]]; then
  command=(sbatch --test-only "${dependency_args[@]}" "${extra[@]}" "$root/$script")
fi
"${command[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=h2g-phase3-tokenize
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_PHASE3_STREAM:?set phase3_openarchives, phase3_foreign, or phase3_old_greek}"
: "${H2G_MEGATRON_RECEIPT:?set exact frozen compiled Megatron runtime receipt}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
case "$H2G_PHASE3_STREAM" in
  phase3_openarchives) pool=openarchives ;;
  phase3_foreign) pool=foreign_replay ;;
  phase3_old_greek) pool=old_greek_replay ;;
  *) echo "invalid Phase-3 tokenization stream: $H2G_PHASE3_STREAM" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "Phase-3 tokenization requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
requirements="$subproject/configs/data_runtime_requirements_v1.txt"
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$requirements" --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null

input="$H2G_STAGE_ROOT/data/phase3_unseen/phase3_${pool}.jsonl"
input_receipt="$H2G_STAGE_ROOT/receipts/phase3_unseen_catalog.json"
tokenizer_root="$H2G_STAGE_ROOT/assets/tokenizer_148480"
tokenizer_receipt="$H2G_STAGE_ROOT/receipts/tokenizer_148480.json"
megatron_root="$H2G_STAGE_ROOT/tools/megatron_training_c92402e_extra_valid"
megatron_receipt="$H2G_MEGATRON_RECEIPT"
case "$megatron_receipt" in
  "$H2G_STAGE_ROOT"/receipts/*) ;;
  *) echo "Megatron receipt must stay inside the stage receipts root" >&2; exit 2 ;;
esac
output_dir="$H2G_STAGE_ROOT/megatron"
output_tag="${H2G_PHASE3_STREAM}_ext"
dataset_prefix="$output_dir/${output_tag}_text_document"
receipt="$H2G_STAGE_ROOT/receipts/tokenized_${H2G_PHASE3_STREAM}.json"
document_catalog="$H2G_STAGE_ROOT/catalogs/${H2G_PHASE3_STREAM}_documents.jsonl"
for path in "$input" "$input_receipt" "$tokenizer_receipt" "$megatron_receipt"; do
  [[ -s "$path" ]] || { echo "Phase-3 tokenization prerequisite missing: $path" >&2; exit 2; }
done
for path in "${dataset_prefix}.bin" "${dataset_prefix}.idx" "$document_catalog" "$receipt"; do
  [[ ! -e "$path" ]] || { echo "immutable Phase-3 tokenization output exists: $path" >&2; exit 2; }
done

mkdir -p "$output_dir" "$H2G_STAGE_ROOT/catalogs" "$H2G_STAGE_ROOT/receipts" "$H2G_STAGE_ROOT/_failed"
temporary_dir="$output_dir/.${output_tag}.partial-${SLURM_JOB_ID}"
failed_dir="$H2G_STAGE_ROOT/_failed/tokenize-${H2G_PHASE3_STREAM}-${SLURM_JOB_ID}"
[[ ! -e "$temporary_dir" && ! -e "$failed_dir" ]] || { echo "Phase-3 tokenization work path exists" >&2; exit 2; }
mkdir "$temporary_dir"
published=0
quarantine_failure() {
  mkdir -p "$failed_dir"
  [[ ! -d "$temporary_dir" ]] || mv -- "$temporary_dir" "$failed_dir/work"
  if [[ "$published" == 1 ]]; then
    for suffix in bin idx; do [[ ! -e "${dataset_prefix}.${suffix}" ]] || mv -- "${dataset_prefix}.${suffix}" "$failed_dir/"; done
  fi
  [[ ! -e "$document_catalog" ]] || mv -- "$document_catalog" "$failed_dir/"
  [[ ! -e "$receipt" ]] || mv -- "$receipt" "$failed_dir/"
}
trap quarantine_failure ERR

temporary_prefix="$temporary_dir/$output_tag"
cd "$megatron_root"
PYTHONDONTWRITEBYTECODE=1 uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" tools/preprocess_data.py \
    --input "$input" --output-prefix "$temporary_prefix" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$tokenizer_root" \
    --append-eod --workers 64 --json-keys text
temporary_dataset_prefix="${temporary_prefix}_text_document"
[[ -s "${temporary_dataset_prefix}.bin" && -s "${temporary_dataset_prefix}.idx" ]] || {
  echo "Phase-3 preprocessing did not produce both indexed files" >&2; exit 2;
}
mv -- "${temporary_dataset_prefix}.bin" "${dataset_prefix}.bin"
mv -- "${temporary_dataset_prefix}.idx" "${dataset_prefix}.idx"
published=1
rmdir "$temporary_dir"

export H2G_CODE_ROOT H2G_CODE_RECEIPT
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/freeze_tokenized_stream.py" \
    --stream "$H2G_PHASE3_STREAM" --input-jsonl "$input" --input-receipt "$input_receipt" \
    --tokenizer-root "$tokenizer_root" --tokenizer-receipt "$tokenizer_receipt" \
    --megatron-root "$megatron_root" --megatron-receipt "$megatron_receipt" \
    --dataset-prefix "$dataset_prefix" --workers 64 \
    --document-catalog "$document_catalog" --output "$receipt"
trap - ERR
printf '%s\n%s\n' "$dataset_prefix" "$receipt"

#!/usr/bin/env bash
#SBATCH --job-name=h2g-tokenize
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
: "${H2G_STREAM:?set hplt, openarchives, foreign, or old_greek}"
: "${H2G_MEGATRON_RECEIPT:?set exact frozen compiled Megatron runtime receipt}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

case "$H2G_STREAM" in
  hplt)
    input="$H2G_STAGE_ROOT/data/hplt_stage_b.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/hplt_stage_b.json"
    output_tag=hplt_only_ext
    ;;
  openarchives)
    input="$H2G_STAGE_ROOT/data/openarchives_stage_b.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/openarchives_stage_b.json"
    output_tag=glossapi_only_ext
    ;;
  foreign)
    input="$H2G_STAGE_ROOT/data/foreign_replay_only_final.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/replay_split.json"
    output_tag=foreign_replay_only_ext
    ;;
  old_greek)
    input="$H2G_STAGE_ROOT/data/old_greek_replay_only_final.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/replay_split.json"
    output_tag=old_greek_replay_only_ext
    ;;
  *) echo "invalid tokenization stream: $H2G_STREAM" >&2; exit 2 ;;
esac

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "stream tokenization requires one debug node" >&2; exit 2;
}
verify_bundle="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_bundle" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
requirements="$subproject/configs/data_runtime_requirements_v1.txt"
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" \
    --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$requirements" \
    --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null

tokenizer_root="$H2G_STAGE_ROOT/assets/tokenizer_148480"
tokenizer_receipt="$H2G_STAGE_ROOT/receipts/tokenizer_148480.json"
megatron_root="$H2G_STAGE_ROOT/tools/megatron_training_c92402e_extra_valid"
megatron_receipt="$H2G_MEGATRON_RECEIPT"
case "$megatron_receipt" in
  "$H2G_STAGE_ROOT"/receipts/*) ;;
  *) echo "Megatron receipt must stay inside the stage receipts root" >&2; exit 2 ;;
esac
output_dir="$H2G_STAGE_ROOT/megatron"
dataset_prefix="$output_dir/${output_tag}_text_document"
receipt="$H2G_STAGE_ROOT/receipts/tokenized_${H2G_STREAM}.json"
document_catalog="$H2G_STAGE_ROOT/catalogs/${H2G_STREAM}_documents.jsonl"
for path in "${dataset_prefix}.bin" "${dataset_prefix}.idx" "$document_catalog" "$receipt"; do
  [[ ! -e "$path" ]] || { echo "immutable tokenization output exists: $path" >&2; exit 2; }
done
[[ -s "$input" && -s "$input_receipt" && -s "$tokenizer_receipt" && -s "$megatron_receipt" ]] || {
  echo "tokenization prerequisite missing" >&2; exit 2;
}

mkdir -p "$output_dir" "$H2G_STAGE_ROOT/catalogs" "$H2G_STAGE_ROOT/receipts" "$H2G_STAGE_ROOT/_failed"
temporary_dir="$output_dir/.${output_tag}.partial-${SLURM_JOB_ID}"
failed_dir="$H2G_STAGE_ROOT/_failed/tokenize-${H2G_STREAM}-${SLURM_JOB_ID}"
[[ ! -e "$temporary_dir" && ! -e "$failed_dir" ]] || { echo "tokenization work path exists" >&2; exit 2; }
mkdir "$temporary_dir"
published=0
quarantine_failure() {
  mkdir -p "$failed_dir"
  if [[ -d "$temporary_dir" ]]; then mv -- "$temporary_dir" "$failed_dir/work"; fi
  if [[ "$published" == 1 ]]; then
    for suffix in bin idx; do
      if [[ -e "${dataset_prefix}.${suffix}" ]]; then
        mv -- "${dataset_prefix}.${suffix}" "$failed_dir/"
      fi
    done
  fi
  [[ ! -e "$receipt" ]] || mv -- "$receipt" "$failed_dir/"
  [[ ! -e "$document_catalog" ]] || mv -- "$document_catalog" "$failed_dir/"
}
trap quarantine_failure ERR

temporary_prefix="$temporary_dir/$output_tag"
cd "$megatron_root"
PYTHONDONTWRITEBYTECODE=1 uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" tools/preprocess_data.py \
    --input "$input" \
    --output-prefix "$temporary_prefix" \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$tokenizer_root" \
    --append-eod --workers 64 --json-keys text
temporary_dataset_prefix="${temporary_prefix}_text_document"
[[ -s "${temporary_dataset_prefix}.bin" && -s "${temporary_dataset_prefix}.idx" ]] || {
  echo "Megatron preprocessing did not produce both indexed files" >&2; exit 2;
}
mv -- "${temporary_dataset_prefix}.bin" "${dataset_prefix}.bin"
mv -- "${temporary_dataset_prefix}.idx" "${dataset_prefix}.idx"
published=1
rmdir "$temporary_dir"

export H2G_CODE_ROOT H2G_CODE_RECEIPT
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/freeze_tokenized_stream.py" \
    --stream "$H2G_STREAM" --input-jsonl "$input" --input-receipt "$input_receipt" \
    --tokenizer-root "$tokenizer_root" --tokenizer-receipt "$tokenizer_receipt" \
    --megatron-root "$megatron_root" --megatron-receipt "$megatron_receipt" \
    --dataset-prefix "$dataset_prefix" --workers 64 \
    --document-catalog "$document_catalog" --output "$receipt"
trap - ERR
printf '%s\n%s\n' "$dataset_prefix" "$receipt"

#!/usr/bin/env bash
# Convert one exact training checkpoint into a verified HF evaluation artifact.
#SBATCH --job-name=h2g-export
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable scientific code root}"
: "${H2G_CODE_RECEIPT:?set immutable scientific code receipt}"
: "${H2G_SCALE:?set 8b or 1p5b}"
: "${H2G_SOURCE_CHECKPOINT_ROOT:?set exact Megatron checkpoint root}"
: "${H2G_SOURCE_ITERATION:?set exact iteration}"
: "${H2G_MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${H2G_TOKENIZER_DIR:?set frozen 148480 tokenizer}"
: "${H2G_MODEL_CONTRACT:?set frozen matched-study contract}"
: "${H2G_EXPORT_ROOT:?set unique checkpoint export root}"
: "${H2G_EVAL_PYTHON:?set frozen evaluation Python}"

case "${SLURM_JOB_PARTITION:-}" in
  debug|normal) ;;
  *) echo "checkpoint export requires a debug or held normal allocation" >&2; exit 2 ;;
esac
[[ "${SLURM_NNODES:-0}" == 1 && "${SLURM_JOB_NUM_NODES:-0}" == 1 ]] || {
  echo "checkpoint export requires exactly one allocated node" >&2
  exit 2
}
case "$H2G_SCALE" in 8b|1p5b) ;; *) echo "invalid model scale" >&2; exit 2;; esac
[[ "$H2G_SOURCE_ITERATION" =~ ^[0-9]+$ ]] || { echo "invalid source iteration" >&2; exit 2; }

UENV_IMAGE=${H2G_UENV_IMAGE:-pytorch/v2.9.1:v2}
TRUE_VOCAB_SIZE=${H2G_TRUE_VOCAB_SIZE:-148480}
TOKENIZER_SHA256=${H2G_TOKENIZER_SHA256:-358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394}
PYTHON_COMPAT_DIR=${H2G_PYTHON_COMPAT_DIR:-$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/runtime_compat}
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"

/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
[[ -f "$PYTHON_COMPAT_DIR/sitecustomize.py" ]] || { echo "compatibility shim missing" >&2; exit 2; }
[[ ! -e "$H2G_EXPORT_ROOT" ]] || { echo "immutable export root exists" >&2; exit 2; }
export PYTHONPATH="$PYTHON_COMPAT_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1

uenv run "$UENV_IMAGE" --view=default -- "$H2G_EVAL_PYTHON" \
  "$subproject/evaluation/checkpoint_export_receipt.py" prepare \
  --scale "$H2G_SCALE" \
  --source-checkpoint-root "$H2G_SOURCE_CHECKPOINT_ROOT" \
  --iteration "$H2G_SOURCE_ITERATION" \
  --output-root "$H2G_EXPORT_ROOT"

start_dist=$(date +%s)
uenv run "$UENV_IMAGE" --view=default -- bash -lc "
  set -euo pipefail
  export PYTHONPATH='$PYTHON_COMPAT_DIR:$H2G_MEGATRON_DIR'
  torchrun --standalone --nproc_per_node=1 \
    '$H2G_MEGATRON_DIR/scripts/conversion/torchdist_2_torch.py' \
    --bf16 \
    --load '$H2G_EXPORT_ROOT/source_view' \
    --ckpt-convert-save '$H2G_EXPORT_ROOT/intermediate' \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model '$H2G_TOKENIZER_DIR'
"
end_dist=$(date +%s)

if [[ "$H2G_SOURCE_ITERATION" == 0 ]]; then
  intermediate_root="$H2G_EXPORT_ROOT/intermediate/torch"
  [[ -d "$intermediate_root/iter_0000000" && ! -e "$intermediate_root/release" ]] || {
    echo "iteration-zero intermediate checkpoint drift" >&2; exit 2;
  }
  ln -s iter_0000000 "$intermediate_root/release"
  printf 'release\n' > "$intermediate_root/latest_checkpointed_iteration.txt"
fi

start_hf=$(date +%s)
uenv run "$UENV_IMAGE" --view=default -- bash -lc "
  set -o pipefail
  export PYTHONPATH='$PYTHON_COMPAT_DIR:$H2G_MEGATRON_DIR:$H2G_MEGATRON_DIR/tools/checkpoint'
  '$H2G_EVAL_PYTHON' '$H2G_MEGATRON_DIR/tools/checkpoint/convert.py' \
    --model-type GPT \
    --loader core \
    --saver swissai_hf \
    --load-dir '$H2G_EXPORT_ROOT/intermediate/torch' \
    --save-dir '$H2G_EXPORT_ROOT/hf' \
    --megatron-path '$H2G_MEGATRON_DIR' \
    --loader-transformer-impl transformer_engine \
    --true-vocab-size '$TRUE_VOCAB_SIZE' \
    --position-embedding-type rope \
    --hf-tokenizer '$H2G_TOKENIZER_DIR' \
    --test-logits 2>&1 | tee '$H2G_EXPORT_ROOT/hf_conversion.log'
"
end_hf=$(date +%s)

printf -v intermediate_iteration 'iter_%07d' "$H2G_SOURCE_ITERATION"
start_mapping=$(date +%s)
uenv run "$UENV_IMAGE" --view=default -- env \
  PYTHONPATH="$subproject/scripts:$H2G_MEGATRON_DIR" \
  "$H2G_EVAL_PYTHON" \
  "$subproject/evaluation/verify_exact_checkpoint_weight_mapping.py" \
  --intermediate-checkpoint "$H2G_EXPORT_ROOT/intermediate/torch/$intermediate_iteration/mp_rank_00/model_optim_rng.pt" \
  --hf-root "$H2G_EXPORT_ROOT/hf" \
  --model-contract "$H2G_MODEL_CONTRACT" \
  --scale "$H2G_SCALE" \
  --true-vocab-size "$TRUE_VOCAB_SIZE" \
  --output "$H2G_EXPORT_ROOT/exact_weight_mapping_receipt.json"
end_mapping=$(date +%s)

uenv run "$UENV_IMAGE" --view=default -- "$H2G_EVAL_PYTHON" - \
  "$H2G_EXPORT_ROOT/conversion_timing.json" \
  "$start_dist" "$end_dist" "$start_hf" "$end_hf" "$start_mapping" "$end_mapping" <<'PY'
import json, sys
out, a, b, c, d, e, f = sys.argv[1:]
payload = {
    "torch_dist_to_torch_seconds": int(b) - int(a),
    "torch_to_hf_and_logit_test_seconds": int(d) - int(c),
    "exact_weight_mapping_verification_seconds": int(f) - int(e),
    "total_conversion_seconds": int(b) - int(a) + int(d) - int(c) + int(f) - int(e),
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

uenv run "$UENV_IMAGE" --view=default -- "$H2G_EVAL_PYTHON" \
  "$subproject/evaluation/checkpoint_export_receipt.py" finalize \
  --scale "$H2G_SCALE" \
  --output-root "$H2G_EXPORT_ROOT" \
  --model-contract "$H2G_MODEL_CONTRACT" \
  --true-vocab-size "$TRUE_VOCAB_SIZE" \
  --tokenizer-json "$H2G_TOKENIZER_DIR/tokenizer.json" \
  --tokenizer-sha256 "$TOKENIZER_SHA256"

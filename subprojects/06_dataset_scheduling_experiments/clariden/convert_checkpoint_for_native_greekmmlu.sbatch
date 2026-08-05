#!/usr/bin/env bash
#SBATCH --job-name=mini_ckpt_to_hf
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=01:00:00

set -euo pipefail
: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${SOURCE_CHECKPOINT_ROOT:?set the immutable training checkpoint root}"
: "${SOURCE_ITERATION:?set exact iteration to export}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${EXPORT_ROOT:?set unique export root}"
: "${PYTHON_COMPAT_DIR:?set frozen NumPy compatibility shim directory}"

UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
LOADER_TRANSFORMER_IMPL=${LOADER_TRANSFORMER_IMPL:-transformer_engine}
TOKENIZER_SHA=${TOKENIZER_SHA:-cc3f544817da0e8d1623e3f7484df7f67464aeb00867aece956880e9b407ef8f}
EXPORT_MODEL_SCALE=${EXPORT_MODEL_SCALE:-0p5B}
case "$EXPORT_MODEL_SCALE" in
  0p5B) HF_LOGIT_TEST_FLAG=--test-logits ;;
  8B)
    : "${FULL8_EVALUATION_ROOT:?8B export requires its frozen evaluation root}"
    HF_LOGIT_TEST_FLAG=
    ;;
  *) echo "unsupported EXPORT_MODEL_SCALE: $EXPORT_MODEL_SCALE" >&2; exit 2 ;;
esac
case "$LOADER_TRANSFORMER_IMPL" in
  transformer_engine|local) ;;
  *)
    echo "unsupported LOADER_TRANSFORMER_IMPL: $LOADER_TRANSFORMER_IMPL" >&2
    exit 2
    ;;
esac
mkdir -p "$EXPORT_ROOT"
[[ -f "$PYTHON_COMPAT_DIR/sitecustomize.py" ]] || {
  echo "missing frozen compatibility shim: $PYTHON_COMPAT_DIR/sitecustomize.py" >&2
  exit 2
}
export PYTHONPATH="$PYTHON_COMPAT_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
  "$EVALUATION_BUNDLE/evaluation/checkpoint_eval_export_receipt.py" prepare \
  --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
  --iteration "$SOURCE_ITERATION" \
  --output-root "$EXPORT_ROOT"

start_dist=$(date +%s)
uenv run "$UENV_IMAGE" --view=default -- bash -lc "
  export PYTHONPATH='$PYTHON_COMPAT_DIR:$MEGATRON_DIR'
  torchrun --standalone --nproc_per_node=1 \
    '$MEGATRON_DIR/scripts/conversion/torchdist_2_torch.py' \
    --bf16 \
    --load '$EXPORT_ROOT/source_view' \
    --ckpt-convert-save '$EXPORT_ROOT/intermediate' \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model '$TOKENIZER_DIR'
"
end_dist=$(date +%s)

if [[ "$SOURCE_ITERATION" == 0 ]]; then
  intermediate_root="$EXPORT_ROOT/intermediate/torch"
  [[ -d "$intermediate_root/iter_0000000" ]] || {
    echo "missing converted iteration-zero checkpoint" >&2
    exit 2
  }
  [[ ! -e "$intermediate_root/release" ]] || {
    echo "refusing to replace existing intermediate release checkpoint" >&2
    exit 2
  }
  ln -s iter_0000000 "$intermediate_root/release"
  printf 'release\n' > "$intermediate_root/latest_checkpointed_iteration.txt"
fi

start_hf=$(date +%s)
uenv run "$UENV_IMAGE" --view=default -- bash -lc "
  set -o pipefail
  export PYTHONPATH='$PYTHON_COMPAT_DIR:$MEGATRON_DIR:$MEGATRON_DIR/tools/checkpoint'
  '$EVAL_VENV/bin/python' '$MEGATRON_DIR/tools/checkpoint/convert.py' \
    --model-type GPT \
    --loader core \
    --saver swissai_hf \
    --load-dir '$EXPORT_ROOT/intermediate/torch' \
    --save-dir '$EXPORT_ROOT/hf' \
    --megatron-path '$MEGATRON_DIR' \
    --loader-transformer-impl '$LOADER_TRANSFORMER_IMPL' \
    --true-vocab-size 148992 \
    --position-embedding-type rope \
    --hf-tokenizer '$TOKENIZER_DIR' \
    $HF_LOGIT_TEST_FLAG 2>&1 | tee '$EXPORT_ROOT/hf_conversion.log'
"
end_hf=$(date +%s)

start_mapping=$(date +%s)
printf -v intermediate_iteration 'iter_%07d' "$SOURCE_ITERATION"
if [[ "$EXPORT_MODEL_SCALE" == 8B ]]; then
  uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$MEGATRON_DIR" \
    "$EVAL_VENV/bin/python" \
    "$FULL8_EVALUATION_ROOT/verify_exact_checkpoint_weight_mapping_8b.py" \
    --intermediate-checkpoint \
    "$EXPORT_ROOT/intermediate/torch/$intermediate_iteration/mp_rank_00/model_optim_rng.pt" \
    --hf-root "$EXPORT_ROOT/hf" \
    --hf-config "$EXPORT_ROOT/hf/config.json" \
    --output "$EXPORT_ROOT/exact_weight_mapping_receipt.json"
else
  uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$MEGATRON_DIR" \
    "$EVAL_VENV/bin/python" \
    "$EVALUATION_BUNDLE/evaluation/verify_exact_checkpoint_weight_mapping.py" \
    --intermediate-checkpoint \
    "$EXPORT_ROOT/intermediate/torch/$intermediate_iteration/mp_rank_00/model_optim_rng.pt" \
    --hf-model "$EXPORT_ROOT/hf/model.safetensors" \
    --hf-config "$EXPORT_ROOT/hf/config.json" \
    --output "$EXPORT_ROOT/exact_weight_mapping_receipt.json"
fi
end_mapping=$(date +%s)

uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" - \
  "$EXPORT_ROOT/conversion_timing.json" \
  "$start_dist" "$end_dist" "$start_hf" "$end_hf" \
  "$start_mapping" "$end_mapping" <<'PY'
import json,sys
out,a,b,c,d,e,f=sys.argv[1:]
payload={
 "torch_dist_to_torch_seconds":int(b)-int(a),
 "torch_to_hf_and_logit_test_seconds":int(d)-int(c),
 "exact_weight_mapping_verification_seconds":int(f)-int(e),
 "total_conversion_seconds":int(b)-int(a)+int(d)-int(c)+int(f)-int(e),
}
open(out,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

if [[ "$EXPORT_MODEL_SCALE" == 8B ]]; then
  uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
    "$FULL8_EVALUATION_ROOT/finalize_checkpoint_export_8b.py" \
    --output-root "$EXPORT_ROOT" \
    --tokenizer-json "$TOKENIZER_DIR/tokenizer.json" \
    --tokenizer-json-sha256 "$TOKENIZER_SHA"
else
  uenv run "$UENV_IMAGE" --view=default -- "$EVAL_VENV/bin/python" \
    "$EVALUATION_BUNDLE/evaluation/checkpoint_eval_export_receipt.py" finalize \
    --output-root "$EXPORT_ROOT" \
    --tokenizer-json "$TOKENIZER_DIR/tokenizer.json" \
    --tokenizer-json-sha256 "$TOKENIZER_SHA"
fi

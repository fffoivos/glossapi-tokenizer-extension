#!/usr/bin/env bash
#SBATCH --job-name=mini_td_convert
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=300G
#SBATCH --time=03:00:00

set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_FULL_TD_DIR:?set verified full tied-TD HF directory}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron checkout}"
: "${MEGATRON_PATCH_DIR:?set pinned Apertus conversion patch directory}"
: "${TD_CONVERSION_ROOT:?set new conversion output root}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
EVAL_VENV=${EVAL_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026}
PYTHON_COMPAT_DIR=${PYTHON_COMPAT_DIR:-/iopsstor/scratch/cscs/fffoivos/python_envs/numpy_compat_shim}

[[ ! -e "$TD_CONVERSION_ROOT" ]] || { echo "refusing to replace $TD_CONVERSION_ROOT" >&2; exit 2; }
test -s "$MINI_FULL_TD_DIR/tied_td_manifest.json"
test -s "$MEGATRON_PATCH_DIR/patch_apertus_extras.py"
test -s "$MEGATRON_DIR/tools/checkpoint/loader_apertus_hf.py"
test -s "$MEGATRON_DIR/tools/checkpoint/saver_swissai_hf.py"
[[ "$(sha256sum "$MEGATRON_DIR/tools/checkpoint/loader_apertus_hf.py" | awk '{print $1}')" == 5b0215969ef05257be22e50fd60f47498542a99637179fbcade73a61384b1b77 ]] || {
  echo "Apertus HF loader drift" >&2; exit 3;
}
[[ "$(sha256sum "$MEGATRON_DIR/tools/checkpoint/saver_swissai_hf.py" | awk '{print $1}')" == 4a113dbf49bb53afaaf138314c9270edc05aa5798bf905d314cd310a5291c57d ]] || {
  echo "SwissAI HF saver drift" >&2; exit 3;
}
[[ "$(sha256sum "$MEGATRON_PATCH_DIR/patch_apertus_extras.py" | awk '{print $1}')" == 4263bc2e2604dd61fc1a492b8e140aa8f8c4c80da52569f347e9e5284046c9ac ]] || {
  echo "Apertus extras patch drift" >&2; exit 3;
}
[[ "$(sha256sum "$MEGATRON_PATCH_DIR/verify_hf_roundtrip.py" | awk '{print $1}')" == 3b45279fdd09f4e37032ae959a3619dc315f849f94470da12904ce92803951af ]] || {
  echo "HF roundtrip verifier drift" >&2; exit 3;
}
mkdir -p "$TD_CONVERSION_ROOT"
commit=$(git -C "$MEGATRON_DIR" rev-parse HEAD)
[[ "$commit" == c92402e39ef3c8e69ea378a59e79059dc14541f4 ]] || {
  echo "Megatron commit drift: $commit" >&2; exit 3;
}

export PYTHONPATH="$PYTHON_COMPAT_DIR:$MEGATRON_DIR:$MEGATRON_DIR/tools/checkpoint"
export CUDA_DEVICE_MAX_CONNECTIONS=1
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  "$EVAL_VENV/bin/python" \
  "$MEGATRON_DIR/tools/checkpoint/convert.py" --model-type GPT \
  --loader apertus_hf --saver core \
  --load-dir "$MINI_FULL_TD_DIR" --save-dir "$TD_CONVERSION_ROOT/legacy_raw" \
  --tokenizer-model "$MINI_FULL_TD_DIR" --bf16 \
  --loader-transformer-impl transformer_engine \
  --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 \
  2>&1 | tee "$TD_CONVERSION_ROOT/hf_to_megatron.log"

if [[ -d "$TD_CONVERSION_ROOT/legacy_raw/iter_0000000" ]]; then
  mv "$TD_CONVERSION_ROOT/legacy_raw/iter_0000000" "$TD_CONVERSION_ROOT/legacy_raw/release"
fi
echo release > "$TD_CONVERSION_ROOT/legacy_raw/latest_checkpointed_iteration.txt"
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  "$EVAL_VENV/bin/python" \
  "$MEGATRON_PATCH_DIR/patch_apertus_extras.py" \
  --hf-dir "$MINI_FULL_TD_DIR" \
  --megatron-dir "$TD_CONVERSION_ROOT/legacy_raw" \
  --out-dir "$TD_CONVERSION_ROOT/legacy_patched" \
  --max-current-diff 1e-3 --overwrite \
  2>&1 | tee "$TD_CONVERSION_ROOT/patch_apertus_extras.log"

uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  torchrun --standalone --nproc_per_node=1 \
  "$MINI_CODE_BUNDLE/initialization/torch_to_torchdist.py" \
  --bf16 --load "$TD_CONVERSION_ROOT/legacy_patched" \
  --ckpt-format torch --ckpt-convert-format torch_dist \
  --ckpt-convert-save "$TD_CONVERSION_ROOT/distributed" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$MINI_FULL_TD_DIR"
INITIAL_ROOT="$TD_CONVERSION_ROOT/distributed/torch_dist"
if [[ -d "$INITIAL_ROOT/iter_0000000" ]]; then
  mv "$INITIAL_ROOT/iter_0000000" "$INITIAL_ROOT/release"
fi
test -d "$INITIAL_ROOT/release"
echo release > "$INITIAL_ROOT/latest_checkpointed_iteration.txt"

uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  torchrun --standalone --nproc_per_node=1 \
  "$MEGATRON_DIR/scripts/conversion/torchdist_2_torch.py" \
  --bf16 --load "$INITIAL_ROOT" \
  --ckpt-convert-save "$TD_CONVERSION_ROOT/roundtrip_intermediate" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$MINI_FULL_TD_DIR"
ROUNDTRIP_TORCH_ROOT="$TD_CONVERSION_ROOT/roundtrip_intermediate/torch"
if [[ -d "$ROUNDTRIP_TORCH_ROOT/iter_0000000" ]]; then
  mv "$ROUNDTRIP_TORCH_ROOT/iter_0000000" "$ROUNDTRIP_TORCH_ROOT/release"
fi
test -d "$ROUNDTRIP_TORCH_ROOT/release"
echo release > "$ROUNDTRIP_TORCH_ROOT/latest_checkpointed_iteration.txt"
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  "$EVAL_VENV/bin/python" \
  "$MEGATRON_DIR/tools/checkpoint/convert.py" --model-type GPT \
  --loader core --saver swissai_hf \
  --load-dir "$ROUNDTRIP_TORCH_ROOT" \
  --save-dir "$TD_CONVERSION_ROOT/hf_roundtrip" \
  --megatron-path "$MEGATRON_DIR" --true-vocab-size 148992 \
  --position-embedding-type rope --hf-tokenizer "$MINI_FULL_TD_DIR" --test-logits \
  2>&1 | tee "$TD_CONVERSION_ROOT/megatron_to_hf.log"
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  "$EVAL_VENV/bin/python" \
  "$MEGATRON_PATCH_DIR/verify_hf_roundtrip.py" \
  --reference-hf-dir "$MINI_FULL_TD_DIR" \
  --roundtrip-hf-dir "$TD_CONVERSION_ROOT/hf_roundtrip" \
  --tokenizer-dir "$MINI_FULL_TD_DIR" --require-r17-match --logits \
  --output-json "$TD_CONVERSION_ROOT/roundtrip_verification.json"
uenv run "$UENV_IMAGE" --view=default -- env PYTHONPATH="$PYTHONPATH" \
  "$EVAL_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/finalize_td_megatron_conversion.py" \
  --hf-reference "$MINI_FULL_TD_DIR" --torch-dist-root "$INITIAL_ROOT" \
  --hf-roundtrip "$TD_CONVERSION_ROOT/hf_roundtrip" \
  --verification "$TD_CONVERSION_ROOT/roundtrip_verification.json" \
  --megatron-commit "$commit" --output "$TD_CONVERSION_ROOT/conversion_receipt.json"

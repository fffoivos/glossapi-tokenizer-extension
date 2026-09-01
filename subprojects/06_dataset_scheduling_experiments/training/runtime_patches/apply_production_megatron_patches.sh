#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_MEGATRON:?set clean pinned SwissAI Megatron root}"
: "${OUTPUT_MEGATRON:?set new immutable patched Megatron root}"
: "${SCIENTIFIC_ROOT:?set dataset-scheduling subproject root}"

expected_commit=c92402e39ef3c8e69ea378a59e79059dc14541f4
[[ "$(git -C "$SOURCE_MEGATRON" rev-parse HEAD)" == "$expected_commit" ]] || {
  echo "pinned Megatron commit drift" >&2; exit 2;
}
git -C "$SOURCE_MEGATRON" diff --quiet
git -C "$SOURCE_MEGATRON" diff --cached --quiet
[[ ! -e "$OUTPUT_MEGATRON" ]] || { echo "refusing to replace $OUTPUT_MEGATRON" >&2; exit 2; }
extra_patch="$SCIENTIFIC_ROOT/training/runtime_patches/megatron_extra_valid_c92402e.patch"
exact_eval_patch="$SCIENTIFIC_ROOT/training/runtime_patches/megatron_exact_eval_iterations_c92402e.patch"
[[ "$(sha256sum "$extra_patch" | awk '{print $1}')" == 2e6810fa8b6c25597ccb3bcb9dc1ff5bf843ead2337e3edde0344605a23ec4c6 ]] || {
  echo "named extra-validation patch drift" >&2; exit 2;
}
[[ "$(sha256sum "$exact_eval_patch" | awk '{print $1}')" == 6d9392cfb0dd08e62089d0a98e2817b222bb9a25ee5cefa8f3cdf29a8ce16bea ]] || {
  echo "exact-evaluation patch drift" >&2; exit 2;
}
git clone --quiet --no-hardlinks "$SOURCE_MEGATRON" "$OUTPUT_MEGATRON"
git -C "$OUTPUT_MEGATRON" checkout --quiet --detach "$expected_commit"
git -C "$OUTPUT_MEGATRON" apply \
  "$extra_patch"
git -C "$OUTPUT_MEGATRON" apply \
  "$exact_eval_patch"
grep -q 'MINI_SCHEDULE_EVAL_ITERATIONS' "$OUTPUT_MEGATRON/megatron/training/training.py"
grep -q 'extra_valid_datasets_provider' "$OUTPUT_MEGATRON/megatron/training/training.py"
[[ "$(git -C "$OUTPUT_MEGATRON" diff --name-only | sort | tr '\n' ' ')" == "megatron/training/arguments.py megatron/training/training.py pretrain_gpt.py " ]] || {
  echo "patched Megatron file set drift" >&2; exit 2;
}
git -C "$OUTPUT_MEGATRON" diff --check

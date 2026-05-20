#!/usr/bin/env bash
# Idempotent: clone (or fetch + check out the pinned commit for) every repo
# referenced by the training recipe. Run from anywhere; CDs into references/repos/.
#
# Updates the pinned commits in MANIFEST.md if you ever bump them.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p repos
cd repos

# (name, url, pinned-commit-or-"HEAD")
REFS=(
    "swiss-ai_Megatron-LM|https://github.com/swiss-ai/Megatron-LM.git|c92402e39ef3c8e69ea378a59e79059dc14541f4"
    "swiss-ai_pretrain-code|https://github.com/swiss-ai/pretrain-code.git|531cc8be2f76064127cad99a61019f985a7c7ee2"
    "swiss-ai_pretrain-data|https://github.com/swiss-ai/pretrain-data.git|HEAD"
    "swiss-ai_lm-evaluation-harness|https://github.com/swiss-ai/lm-evaluation-harness.git|HEAD"
    "swiss-ai_apertus-finetuning-recipes|https://github.com/swiss-ai/apertus-finetuning-recipes.git|HEAD"
    "swiss-ai_apertus-tech-report|https://github.com/swiss-ai/apertus-tech-report.git|HEAD"
    "apple_ml-ademamix|https://github.com/apple/ml-ademamix.git|HEAD"
    "EleutherAI_lm-evaluation-harness|https://github.com/EleutherAI/lm-evaluation-harness.git|HEAD"
)

for ref in "${REFS[@]}"; do
    IFS='|' read -r name url pin <<< "$ref"
    if [ -d "$name/.git" ]; then
        echo "=== $name: updating ==="
        (cd "$name" && git fetch --quiet origin)
    else
        echo "=== $name: cloning ==="
        git clone --quiet "$url" "$name"
    fi

    if [ "$pin" != "HEAD" ]; then
        (cd "$name" && git checkout --quiet "$pin")
        echo "  checked out $pin"
    else
        echo "  at HEAD: $(cd "$name" && git rev-parse --short HEAD)"
    fi
done

echo
echo "=== summary ==="
for ref in "${REFS[@]}"; do
    IFS='|' read -r name url pin <<< "$ref"
    if [ -d "$name/.git" ]; then
        commit=$(cd "$name" && git rev-parse HEAD)
        size=$(du -sh "$name" 2>/dev/null | cut -f1)
        echo "  $name  $commit  $size"
    fi
done

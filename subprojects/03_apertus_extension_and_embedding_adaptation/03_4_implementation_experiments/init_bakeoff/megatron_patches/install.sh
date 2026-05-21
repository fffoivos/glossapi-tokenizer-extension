#!/usr/bin/env bash
# Symlink loader_apertus_hf.py into swiss-ai/Megatron-LM/tools/checkpoint/
# so `convert.py --loader apertus_hf …` can find it.
#
# Run this once after cloning swiss-ai/Megatron-LM on Clariden (or any host).
# Idempotent — uses ln -sf.
#
# Usage:
#   bash install.sh /path/to/swiss-ai/Megatron-LM
#
# If $MEGATRON_LM_DIR is set in the environment, it's used as default.

set -euo pipefail

MEGATRON_DIR="${1:-${MEGATRON_LM_DIR:-}}"
if [ -z "$MEGATRON_DIR" ]; then
    echo "ERROR: pass Megatron-LM dir as arg or set MEGATRON_LM_DIR" >&2
    echo "Usage: bash $0 /path/to/swiss-ai/Megatron-LM" >&2
    exit 2
fi

TARGET_DIR="$MEGATRON_DIR/tools/checkpoint"
if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: $TARGET_DIR does not exist (is $MEGATRON_DIR the right Megatron repo?)" >&2
    exit 2
fi

SRC="$(cd "$(dirname "$0")" && pwd)/loader_apertus_hf.py"
DEST="$TARGET_DIR/loader_apertus_hf.py"

ln -sf "$SRC" "$DEST"
echo "✓ symlinked $DEST → $SRC"

# Sanity: ast-parse the loader (doesn't require torch/Megatron — runs anywhere).
# A real import-check happens implicitly when convert.py loads the loader,
# inside the Megatron pytorch env at Clariden setup time.
python3 -c "
import ast
with open('$DEST') as f:
    code = f.read()
tree = ast.parse(code)
funcs = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
assert 'add_arguments' in funcs and 'load_checkpoint' in funcs, \
    f'loader contract missing required functions: {funcs}'
print('✓ loader_apertus_hf parses + has add_arguments + load_checkpoint')
print('  functions:', funcs)
"

echo
echo "Use with:"
echo "  cd $MEGATRON_DIR"
echo "  python3 tools/checkpoint/convert.py \\"
echo "      --loader apertus_hf \\"
echo "      --saver core \\"
echo "      --load-dir <HF apertus dir> \\"
echo "      --save-dir <output megatron dir> \\"
echo "      --tokenizer-model <HF apertus dir> \\"
echo "      --bf16"

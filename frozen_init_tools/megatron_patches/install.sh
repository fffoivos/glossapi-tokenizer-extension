#!/usr/bin/env bash
# Symlink the Apertus loader plus the Swiss-AI saver plugins into
# swiss-ai/Megatron-LM/tools/checkpoint/ so checkpoint conversion can find them.
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for plugin in loader_apertus_hf.py saver_core.py saver_swissai_hf.py schema_core.py schema_base.py utils.py; do
    src="$SCRIPT_DIR/$plugin"
    dest="$TARGET_DIR/$plugin"
    if [ ! -f "$src" ]; then
        echo "ERROR: missing plugin source: $src" >&2
        exit 3
    fi
    ln -sf "$src" "$dest"
    echo "✓ symlinked $dest → $src"
done

# Sanity: ast-parse the plugins (doesn't require torch/Megatron — runs anywhere).
# A real import-check happens implicitly when convert.py loads them, inside the
# Megatron pytorch env at Clariden setup time.
python3 -c "
import ast
from pathlib import Path
checks = {
    'loader_apertus_hf.py': {'add_arguments', 'load_checkpoint'},
    'saver_core.py': {'add_arguments', 'save_checkpoint'},
    'saver_swissai_hf.py': {'add_arguments', 'save_checkpoint'},
    'schema_core.py': set(),
    'schema_base.py': set(),
    'utils.py': set(),
}
for name, required in checks.items():
    path = Path('$TARGET_DIR') / name
    tree = ast.parse(path.read_text())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = required - funcs
    assert not missing, f'{name} missing required functions: {sorted(missing)}'
    print(f'✓ {name} parses + has {sorted(required)}')
"

echo
echo "Use with:"
echo "  cd $MEGATRON_DIR"
echo "  python3 tools/checkpoint/convert.py \\"
echo "      --model-type GPT \\"
echo "      --loader apertus_hf \\"
echo "      --saver core \\"
echo "      --load-dir <HF apertus dir> \\"
echo "      --save-dir <output megatron dir> \\"
echo "      --tokenizer-model <HF apertus dir> \\"
echo "      --bf16"
echo
echo "Note (R17): raw saver_core output resets xIELU + QK-Norm extras because"
echo "the core protocol has no slots for them. Accepted bakeoff/production"
echo "checkpoints must be post-patched with patch_apertus_extras.py and verified"
echo "with verify_hf_roundtrip.py before use."

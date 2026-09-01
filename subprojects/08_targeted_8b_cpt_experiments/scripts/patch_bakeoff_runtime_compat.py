#!/usr/bin/env python3
"""Patch the frozen bakeoff launcher to load an explicit runtime compatibility shim."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    text = args.trainer.read_text(encoding="utf-8")
    marker = 'runtime_pythonpath="$MEGATRON_LM_SWISSAI_DIR"'
    if marker in text:
        return 0
    parent = 'export PYTHONPATH="$MEGATRON_LM_SWISSAI_DIR"\n'
    if text.count(parent) != 1:
        raise RuntimeError("unexpected bakeoff PYTHONPATH initialization shape")
    replacement = '''runtime_pythonpath="$MEGATRON_LM_SWISSAI_DIR"
if [[ -n "${RUNTIME_COMPAT_DIR:-}" ]]; then
    [[ -f "$RUNTIME_COMPAT_DIR/sitecustomize.py" ]] || {
        echo "ERROR: runtime compatibility shim is missing" >&2
        exit 2
    }
    runtime_pythonpath="$RUNTIME_COMPAT_DIR:$runtime_pythonpath"
fi
export PYTHONPATH="$runtime_pythonpath"
'''
    text = text.replace(parent, replacement, 1)
    child = "export PYTHONPATH='$MEGATRON_LM_SWISSAI_DIR'"
    if text.count(child) != 2:
        raise RuntimeError("unexpected bakeoff child PYTHONPATH shape")
    text = text.replace(child, "export PYTHONPATH='$runtime_pythonpath'")
    args.trainer.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

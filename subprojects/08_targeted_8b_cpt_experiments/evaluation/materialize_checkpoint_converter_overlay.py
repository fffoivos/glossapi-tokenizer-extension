#!/usr/bin/env python3
"""Build a tiny immutable evaluation-only overlay for the Megatron converter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def atomic_json(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable receipt already exists: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    megatron_root = args.megatron_root.resolve()
    patch = args.patch.resolve()
    output_root = args.output_root.resolve()
    source_convert = megatron_root / "tools/checkpoint/convert.py"
    source_saver = megatron_root / "tools/checkpoint/saver_swissai_hf.py"
    if not source_convert.is_file() or not source_saver.is_file() or not patch.is_file():
        raise FileNotFoundError("pinned converter source or overlay patch is missing")
    if output_root.exists():
        raise FileExistsError(f"immutable converter overlay exists: {output_root}")

    output_root.mkdir(parents=True)
    try:
        shutil.copy2(source_convert, output_root / "convert.py")
        shutil.copy2(source_saver, output_root / "saver_swissai_hf.py")
        result = subprocess.run(
            ["patch", "--batch", "--forward", "--strip=1", "--input", str(patch)],
            cwd=output_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "evaluation-only converter patch did not apply: "
                f"{result.stdout}{result.stderr}"
            )
        if (output_root / "convert.py").read_bytes() != source_convert.read_bytes():
            raise ValueError("converter overlay altered convert.py")
        receipt = output_root / "converter_overlay_receipt.json"
        atomic_json(
            receipt,
            {
                "schema_version": "apertus_evaluation_converter_overlay_v1",
                "status": "completed",
                "source_megatron_root": str(megatron_root),
                "overlay_root": str(output_root),
                "patch": binding(patch),
                "source_convert": binding(source_convert),
                "source_saver": binding(source_saver),
                "overlay_convert": binding(output_root / "convert.py"),
                "overlay_saver": binding(output_root / "saver_swissai_hf.py"),
            },
        )
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

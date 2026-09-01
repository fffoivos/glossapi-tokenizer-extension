#!/usr/bin/env python3
"""Verify the immutable x86 runtime used by the long TD coverage scan."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


SCHEMA = "apertus_hard_h_to_g_td_xfer_runtime_v1"
PACKAGES = {"numpy": "2.4.6", "tokenizers": "0.22.1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.runtime_root.resolve()
    receipt = read_object(args.receipt.resolve())
    code_receipt = read_object(args.code_receipt.resolve())
    live_packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    checks = {
        "schema": receipt.get("schema_version") == SCHEMA,
        "status": receipt.get("status") == "completed",
        "machine": receipt.get("machine") == platform.machine() == "x86_64",
        "runtime_root": receipt.get("runtime_root") == str(root),
        "python_executable": Path(sys.executable).resolve() == (root / "bin/python").resolve(),
        "python_version": receipt.get("python_version") == platform.python_version(),
        "requirements_path": receipt.get("requirements_path") == str(args.requirements.resolve()),
        "requirements_sha256": receipt.get("requirements_sha256") == sha256_file(args.requirements),
        "packages": receipt.get("packages") == live_packages == PACKAGES,
        "code_root": receipt.get("code_bundle", {}).get("root") == str(args.code_root.resolve()),
        "code_tree": receipt.get("code_bundle", {}).get("tree_sha256")
        == code_receipt.get("tree_sha256"),
        "code_kind": code_receipt.get("kind") == "scientific",
        "code_status": code_receipt.get("status") == "frozen",
    }
    for name in PACKAGES:
        module = importlib.import_module(name)
        checks[f"import:{name}"] = bool(getattr(module, "__file__", None))
    recorded = receipt.get("required_files", {})
    checks["required_files_shape"] = isinstance(recorded, dict) and bool(recorded)
    if checks["required_files_shape"]:
        for relative, expected in recorded.items():
            path = root / relative
            checks[f"file:{relative}"] = path.is_file() and sha256_file(path) == expected

    failed = sorted(name for name, value in checks.items() if value is not True)
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"TD xfer runtime verification failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

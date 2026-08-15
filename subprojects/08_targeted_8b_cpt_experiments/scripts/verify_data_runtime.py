#!/usr/bin/env python3
"""Fail-closed verifier for the targeted CPT data-preparation runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


SCHEMA = "apertus_hard_h_to_g_data_runtime_v1"
REQUIRED_DISTRIBUTIONS = {
    "blake3": "1.0.9",
    "datasets": "4.0.0",
    "duckdb": "1.5.4",
    "huggingface-hub": "0.36.0",
    "numpy": "2.5.1",
    "pyarrow": "21.0.0",
    "pytest": "8.4.1",
    "regex": "2026.7.10",
    "tokenizers": "0.22.1",
    "transformers": "4.57.0",
    "zstandard": "0.25.0",
}
REQUIRED_IMPORTS = (
    "blake3",
    "datasets",
    "duckdb",
    "huggingface_hub",
    "numpy",
    "pyarrow",
    "pyarrow.compute",
    "pyarrow.dataset",
    "pyarrow.parquet",
    "pytest",
    "regex",
    "tokenizers",
    "torch",
    "transformers",
    "zstandard",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
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
    receipt_path = args.receipt.resolve()
    requirements = args.requirements.resolve()
    code_root = args.code_root.resolve()
    code_receipt_path = args.code_receipt.resolve()
    receipt = read_json(receipt_path)
    code_receipt = read_json(code_receipt_path)

    checks = {
        "schema": receipt.get("schema_version") == SCHEMA,
        "status": receipt.get("status") == "completed",
        "machine": receipt.get("machine") == platform.machine() == "aarch64",
        "runtime_root": receipt.get("runtime_root") == str(root),
        "python_executable": Path(sys.executable).resolve() == (root / "bin/python").resolve(),
        "python_version": receipt.get("python_version") == platform.python_version(),
        "uenv_image": receipt.get("uenv_image") == "pytorch/v2.9.1:v2",
        "torch_version": receipt.get("torch_version") == importlib.metadata.version("torch")
        and importlib.metadata.version("torch").startswith("2.9.1"),
        "requirements_path": receipt.get("requirements_path") == str(requirements),
        "requirements_sha256": receipt.get("requirements_sha256") == sha256_file(requirements),
        "code_root": receipt.get("code_bundle", {}).get("root") == str(code_root),
        "code_tree": receipt.get("code_bundle", {}).get("tree_sha256")
        == code_receipt.get("tree_sha256"),
        "code_kind": code_receipt.get("kind") == "scientific",
        "code_status": code_receipt.get("status") == "frozen",
        "packages": receipt.get("packages") == REQUIRED_DISTRIBUTIONS,
    }

    imported = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        imported[name] = str(getattr(module, "__file__", None))
        checks[f"import:{name}"] = bool(getattr(module, "__file__", None))
    checks["live_packages"] = {
        name: importlib.metadata.version(name) for name in REQUIRED_DISTRIBUTIONS
    } == REQUIRED_DISTRIBUTIONS
    checks["pyarrow_version"] = importlib.import_module("pyarrow").__version__ == "21.0.0"

    recorded_files = receipt.get("required_files", {})
    checks["required_files_shape"] = isinstance(recorded_files, dict) and bool(recorded_files)
    if checks["required_files_shape"]:
        for relative, expected in recorded_files.items():
            path = root / relative
            checks[f"file:{relative}"] = path.is_file() and sha256_file(path) == expected

    failed = sorted(name for name, value in checks.items() if value is not True)
    payload = {
        "ok": not failed,
        "schema_version": SCHEMA,
        "runtime_root": str(root),
        "checks": checks,
        "imports": imported,
        "failed": failed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"data runtime verification failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

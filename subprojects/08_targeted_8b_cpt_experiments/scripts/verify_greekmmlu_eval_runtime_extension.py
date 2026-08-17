#!/usr/bin/env python3
"""Verify the immutable GreekMMLU evaluator-runtime extension.

This is an experiment-owned workaround for
fffoivos/apertus-cscs-efficiency#88.  It leaves the verified data runtime
unchanged and adds only the loader dependency that its frozen Transformers
scorer requires.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


SCHEMA = "apertus_hard_h_to_g_greekmmlu_eval_runtime_extension_v1"
EXTENSION_PACKAGES = {"accelerate": "1.14.0"}
REQUIRED_IMPORTS = ("accelerate", "datasets", "torch", "transformers")


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
    parser.add_argument("--base-runtime-root", type=Path, required=True)
    parser.add_argument("--base-runtime-receipt", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.runtime_root.resolve()
    receipt = read_object(args.receipt.resolve())
    requirements = args.requirements.resolve()
    base_root = args.base_runtime_root.resolve()
    base_receipt_path = args.base_runtime_receipt.resolve()
    base_receipt = read_object(base_receipt_path)
    code_root = args.code_root.resolve()
    code_receipt = read_object(args.code_receipt.resolve())
    site = root / "lib" / "python3.12" / "site-packages"
    pth = site / "h2g_base_data_runtime.pth"

    checks = {
        "schema": receipt.get("schema_version") == SCHEMA,
        "status": receipt.get("status") == "completed",
        "machine": receipt.get("machine") == platform.machine() == "aarch64",
        "runtime_root": receipt.get("runtime_root") == str(root),
        "python_executable": Path(sys.executable).resolve() == (root / "bin/python").resolve(),
        "python_version": receipt.get("python_version") == platform.python_version(),
        "uenv_image": receipt.get("uenv_image") == "pytorch/v2.9.1:v2",
        "torch_version": importlib.metadata.version("torch").startswith("2.9.1"),
        "requirements_path": receipt.get("requirements_path") == str(requirements),
        "requirements_sha256": receipt.get("requirements_sha256") == sha256_file(requirements),
        "packages": receipt.get("packages") == EXTENSION_PACKAGES,
        "live_packages": {
            name: importlib.metadata.version(name) for name in EXTENSION_PACKAGES
        }
        == EXTENSION_PACKAGES,
        "base_runtime_root": receipt.get("base_runtime", {}).get("root") == str(base_root),
        "base_runtime_receipt": receipt.get("base_runtime", {}).get("receipt_sha256")
        == sha256_file(base_receipt_path),
        "base_runtime_status": base_receipt.get("status") == "completed",
        "base_runtime_schema": base_receipt.get("schema_version")
        == "apertus_hard_h_to_g_data_runtime_v1",
        "code_root": receipt.get("code_bundle", {}).get("root") == str(code_root),
        "code_tree": receipt.get("code_bundle", {}).get("tree_sha256")
        == code_receipt.get("tree_sha256"),
        "code_kind": code_receipt.get("kind") == "scientific",
        "code_status": code_receipt.get("status") == "frozen",
        "base_site_pth": pth.is_file()
        and pth.read_text(encoding="utf-8").strip()
        == str(base_root / "lib" / "python3.12" / "site-packages"),
    }

    imports = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        origin = str(getattr(module, "__file__", ""))
        imports[name] = origin
        checks[f"import:{name}"] = bool(origin)
    checks["accelerate_from_extension"] = imports["accelerate"].startswith(str(root))
    checks["datasets_from_base"] = imports["datasets"].startswith(str(base_root))

    failed = sorted(name for name, value in checks.items() if value is not True)
    print(
        json.dumps(
            {
                "ok": not failed,
                "schema_version": SCHEMA,
                "runtime_root": str(root),
                "checks": checks,
                "imports": imports,
                "failed": failed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failed:
        raise SystemExit(f"GreekMMLU evaluator runtime verification failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

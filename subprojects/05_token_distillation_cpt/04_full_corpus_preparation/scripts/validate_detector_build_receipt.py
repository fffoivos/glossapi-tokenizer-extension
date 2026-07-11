#!/usr/bin/env python3
"""Fail closed unless a detector binary matches its exact Clariden build receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def elf_machine(path: Path) -> int:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2:
        raise ValueError(f"detector is not a 64-bit ELF binary: {path}")
    byteorder = "little" if header[5] == 1 else "big" if header[5] == 2 else None
    if byteorder is None:
        raise ValueError(f"detector ELF has an invalid byte-order marker: {path}")
    return int.from_bytes(header[18:20], byteorder)


def validate_receipt(
    *,
    receipt_path: Path,
    repo: Path,
    binary: Path,
    cargo_lock: Path,
    cargo_toml: Path,
    current_commit: str,
    current_architecture: str,
) -> dict:
    receipt = load_object(receipt_path)
    errors: list[str] = []
    if receipt.get("schema_version") != "full_cpt_detector_build_receipt_v1":
        errors.append("unsupported detector-build receipt schema")
    if receipt.get("status") != "passed":
        errors.append("detector-build receipt is not passed")
    if receipt.get("code_commit") != current_commit:
        errors.append("detector was not built from the current checkout commit")
    if receipt.get("architecture") != current_architecture or current_architecture != "aarch64":
        errors.append(
            f"detector architecture mismatch: receipt={receipt.get('architecture')!r}, current={current_architecture!r}"
        )
    if Path(str(receipt.get("repo", ""))).resolve() != repo.resolve():
        errors.append("detector-build receipt repository path mismatch")

    checks = (
        ("binary", binary, True),
        ("cargo_lock", cargo_lock, False),
        ("cargo_toml", cargo_toml, False),
    )
    for label, path, check_size in checks:
        recorded = receipt.get(label, {})
        if not isinstance(recorded, dict):
            errors.append(f"detector-build receipt {label} entry is invalid")
            continue
        if not path.is_file():
            errors.append(f"current {label} file is missing: {path}")
            continue
        if Path(str(recorded.get("path", ""))).resolve() != path.resolve():
            errors.append(f"detector-build receipt {label} path mismatch")
        if recorded.get("sha256") != sha256_file(path):
            errors.append(f"detector-build receipt {label} hash mismatch")
        if check_size and recorded.get("size") != path.stat().st_size:
            errors.append("detector-build receipt binary size mismatch")
        if check_size and (recorded.get("elf_machine") != "EM_AARCH64" or elf_machine(path) != 183):
            errors.append("detector binary is not an AArch64 ELF artifact")

    if errors:
        raise ValueError("detector-build receipt validation failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": "full_cpt_detector_build_check_v1",
        "ok": True,
        "code_commit": current_commit,
        "architecture": current_architecture,
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "binary_sha256": receipt["binary"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cargo-lock", type=Path, required=True)
    parser.add_argument("--cargo-toml", type=Path, required=True)
    args = parser.parse_args()

    current_commit = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    result = validate_receipt(
        receipt_path=args.receipt,
        repo=args.repo,
        binary=args.binary,
        cargo_lock=args.cargo_lock,
        cargo_toml=args.cargo_toml,
        current_commit=current_commit,
        current_architecture=platform.machine(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

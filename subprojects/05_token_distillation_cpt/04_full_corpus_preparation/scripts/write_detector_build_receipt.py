#!/usr/bin/env python3
"""Write an immutable receipt for a clean, exact detector build."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def elf_machine(path: Path) -> int:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF" or header[4] != 2:
        raise ValueError(f"detector is not a 64-bit ELF binary: {path}")
    if header[5] == 1:
        byteorder = "little"
    elif header[5] == 2:
        byteorder = "big"
    else:
        raise ValueError(f"detector ELF has an invalid byte-order marker: {path}")
    return int.from_bytes(header[18:20], byteorder)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_receipt(
    *,
    repo: Path,
    binary: Path,
    cargo_lock: Path,
    cargo_toml: Path,
    code_commit: str,
    architecture: str,
    cargo_version: str,
    rustc_version: str,
    published_binary_path: Path | None = None,
) -> dict:
    repo = repo.resolve()
    binary = binary.resolve()
    cargo_lock = cargo_lock.resolve()
    cargo_toml = cargo_toml.resolve()
    published_binary_path = (published_binary_path or binary).resolve()
    for label, path in (
        ("repository", repo),
        ("detector binary", binary),
        ("Cargo.lock", cargo_lock),
        ("Cargo.toml", cargo_toml),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} missing: {path}")
    if not binary.is_file() or binary.stat().st_size <= 0:
        raise ValueError(f"detector binary is empty or not a regular file: {binary}")
    if len(code_commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in code_commit):
        raise ValueError("code commit must be a lowercase full Git object ID")
    if architecture != "aarch64":
        raise ValueError(f"Clariden detector must be built for aarch64, got {architecture!r}")
    machine = elf_machine(binary)
    if machine != 183:
        raise ValueError(f"detector ELF machine is {machine}, expected EM_AARCH64 (183)")

    return {
        "schema_version": "full_cpt_detector_build_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": code_commit,
        "repo": str(repo),
        "architecture": architecture,
        "toolchain": {"cargo": cargo_version, "rustc": rustc_version},
        "cargo_lock": {"path": str(cargo_lock), "sha256": sha256_file(cargo_lock)},
        "cargo_toml": {"path": str(cargo_toml), "sha256": sha256_file(cargo_toml)},
        "binary": {
            "path": str(published_binary_path),
            "size": binary.stat().st_size,
            "sha256": sha256_file(binary),
            "elf_machine": "EM_AARCH64",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cargo-lock", type=Path, required=True)
    parser.add_argument("--cargo-toml", type=Path, required=True)
    parser.add_argument("--cargo-bin", type=Path, required=True)
    parser.add_argument("--rustc-bin", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--published-binary-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable detector-build receipt: {args.output}")
    actual_commit = command_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"])
    if actual_commit != args.code_commit:
        raise ValueError(f"checkout moved during detector build: expected {args.code_commit}, got {actual_commit}")
    receipt = build_receipt(
        repo=args.repo,
        binary=args.binary,
        cargo_lock=args.cargo_lock,
        cargo_toml=args.cargo_toml,
        code_commit=args.code_commit,
        architecture=platform.machine(),
        cargo_version=command_output([str(args.cargo_bin), "--version"]),
        rustc_version=command_output([str(args.rustc_bin), "-Vv"]),
        published_binary_path=args.published_binary_path,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps({"ok": True, "output": str(args.output), "binary_sha256": receipt["binary"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Close and verify Agent 1 v3's protected anonymization ledger.

The anonymizer intentionally writes a private, shard-aligned Parquet ledger
with raw span values.  Its public manifest contains only receipts, not those
values.  This helper reopens every receipt and produces a compact, private
ledger-closure manifest that later stages can bind as a *file* contract input.
It never copies a ledger row or a raw identifier into its output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping


ANONYMIZATION_MANIFEST_SCHEMA = "agent1_full_corpus_v3_anonymization_manifest_v1"
LEDGER_CLOSURE_SCHEMA = "agent1_full_corpus_v3_protected_anonymization_ledger_closure_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"required non-empty JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def relative_receipt_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: missing relative receipt path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}: ledger receipt path must be a safe relative path")
    if path.suffix != ".parquet":
        raise ValueError(f"{label}: protected-ledger receipt must name a .parquet shard")
    return path


def require_private_ledger_root(path: Path, *, label: str) -> Path:
    """Return an existing, non-symlink protected-ledger root with exact mode."""

    raw = path.absolute()
    try:
        metadata = raw.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {raw}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FileNotFoundError(f"{label} is missing/unsafe: {raw}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError(f"{label} must have mode 0700: {raw}")
    return raw.resolve(strict=True)


def protected_ledger_path(root: Path, relative: Path, *, label: str) -> Path:
    """Join a receipt path without permitting a symlinked path component."""

    path = root
    for part in relative.parts:
        path = path / part
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label}: protected ledger shard is missing: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FileNotFoundError(f"{label}: protected ledger path contains a symlink: {path}")
    return path


def protected_ledger_parquet_inventory(root: Path) -> dict[str, Path]:
    """Enumerate real Parquet shards and reject symlinks anywhere in the tree."""

    inventory: dict[str, Path] = {}
    pending: list[tuple[Path, Path]] = [(root, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = relative_directory / entry.name
                if entry.is_symlink():
                    raise ValueError(f"protected ledger contains a symlink: {relative.as_posix()}")
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append((path, relative))
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() == ".parquet":
                    inventory[relative.as_posix()] = path
    return inventory


def verify_receipt(path: Path, receipt: Mapping[str, object], *, label: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    required = {"path", "sha256", "bytes", "rows", "row_groups"}
    if set(receipt) != required:
        raise ValueError(f"{label}: protected-ledger receipt key drift")
    try:
        file_metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label}: protected ledger shard is missing/unsafe: {path}") from exc
    if (
        stat.S_ISLNK(file_metadata.st_mode)
        or not stat.S_ISREG(file_metadata.st_mode)
        or file_metadata.st_size < 1
    ):
        raise FileNotFoundError(f"{label}: protected ledger shard is missing/unsafe: {path}")
    if stat.S_IMODE(file_metadata.st_mode) != 0o600:
        raise ValueError(f"{label}: protected ledger shard must have mode 0600: {path}")
    expected_bytes = receipt.get("bytes")
    expected_sha256 = receipt.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise ValueError(f"{label}: invalid protected-ledger byte receipt")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label}: invalid protected-ledger SHA-256 receipt")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label}: protected-ledger bytes/SHA-256 drift")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != receipt.get("rows") or metadata.num_row_groups != receipt.get("row_groups"):
        raise ValueError(f"{label}: protected-ledger Parquet metadata drift")
    return {
        "relative_path": str(receipt["path"]),
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def write_no_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_closure(*, manifest_path: Path, ledger_root: Path) -> dict[str, Any]:
    manifest = read_object(manifest_path)
    if manifest.get("schema_version") != ANONYMIZATION_MANIFEST_SCHEMA or manifest.get("status") != "completed":
        raise ValueError("a completed Agent 1 v3 anonymization manifest is required")
    protected = manifest.get("protected_ledger")
    if not isinstance(protected, Mapping):
        raise ValueError("anonymization manifest lacks protected-ledger metadata")
    expected_root_value = protected.get("path")
    if not isinstance(expected_root_value, str) or not expected_root_value:
        raise ValueError("anonymization manifest lacks protected-ledger root path")
    declared_root = Path(expected_root_value)
    if declared_root.is_symlink():
        raise ValueError("anonymization manifest declares a symlinked protected-ledger root")
    expected_root = declared_root.resolve(strict=True)
    actual_root = require_private_ledger_root(ledger_root, label="protected ledger root")
    if actual_root != expected_root:
        raise ValueError("provided protected-ledger root differs from anonymization manifest")
    if protected.get("contains_raw_span_values") is not True or protected.get("public_training_output") is not False:
        raise ValueError("anonymization manifest does not declare a non-public raw-span protected ledger")
    if protected.get("directory_mode") != "0700" or protected.get("file_mode") != "0600":
        raise ValueError("anonymization manifest does not declare the required 0700/0600 protected-ledger modes")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("anonymization manifest has no shard receipts")
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise ValueError(f"anonymization manifest file receipt {index} is not an object")
        receipt = row.get("protected_ledger")
        if not isinstance(receipt, Mapping):
            raise ValueError(f"anonymization manifest file receipt {index} lacks protected ledger")
        relative = relative_receipt_path(receipt.get("path"), label=f"protected ledger receipt {index}")
        relative_name = relative.as_posix()
        if relative_name in seen:
            raise ValueError(f"duplicate protected ledger receipt: {relative}")
        seen.add(relative_name)
        path = protected_ledger_path(actual_root, relative, label=f"protected ledger receipt {index}")
        observed.append(verify_receipt(path, receipt, label=f"protected ledger receipt {index}"))

    actual_inventory = protected_ledger_parquet_inventory(actual_root)
    expected_inventory = set(seen)
    if set(actual_inventory) != expected_inventory:
        raise ValueError("protected-ledger Parquet inventory does not exactly match anonymization manifest receipts")

    expected_rows = manifest.get("counts", {}).get("protected_ledger_rows") if isinstance(manifest.get("counts"), Mapping) else None
    actual_rows = sum(int(item["rows"]) for item in observed)
    if not isinstance(expected_rows, int) or expected_rows != actual_rows:
        raise ValueError("protected ledger rows do not close against anonymization manifest")
    return {
        "schema_version": LEDGER_CLOSURE_SCHEMA,
        "status": "passed",
        "anonymization_manifest": binding(manifest_path),
        "protected_ledger": {
            "path": str(actual_root),
            "contains_raw_span_values": True,
            "public_training_output": False,
            "directory_mode": protected.get("directory_mode"),
            "file_mode": protected.get("file_mode"),
        },
        "counts": {"shards": len(observed), "protected_ledger_rows": actual_rows},
        "files": sorted(observed, key=lambda item: str(item["relative_path"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymization-manifest", type=Path, required=True)
    parser.add_argument("--protected-ledger-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite ledger closure: {args.output}")
    payload = build_closure(
        manifest_path=args.anonymization_manifest.resolve(),
        ledger_root=args.protected_ledger_root,
    )
    write_no_replace(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "shards": payload["counts"]["shards"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

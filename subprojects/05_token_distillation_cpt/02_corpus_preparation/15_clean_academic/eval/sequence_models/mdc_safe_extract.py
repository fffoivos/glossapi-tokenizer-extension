#!/usr/bin/env python3
"""Safely and reproducibly extract the quarantined MDC Greek-PhD archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence


class SafeExtractionError(ValueError):
    """The archive or an existing extraction violates the extraction contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_is_within(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise SafeExtractionError(f"immutable output differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise SafeExtractionError("archive member has an empty/NUL path")
    raw_parts = name.split("/")
    if ".." in raw_parts:
        raise SafeExtractionError(f"archive member contains '..': {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise SafeExtractionError(f"archive member path is absolute/empty: {name!r}")
    normalized = PurePosixPath(*(part for part in path.parts if part not in {"", "."}))
    if not normalized.parts:
        raise SafeExtractionError(f"archive member path normalizes empty: {name!r}")
    return normalized


def _copy_exact(source: BinaryIO, destination: Path, expected_bytes: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    copied = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                copied += len(block)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if copied != expected_bytes:
        destination.unlink(missing_ok=True)
        raise SafeExtractionError(
            f"archive member byte count drift for {destination}: {copied} != {expected_bytes}"
        )


def _extract_fresh(archive_path: Path, destination: Path) -> None:
    seen: set[str] = set()
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise SafeExtractionError(f"cannot open gzip/tar archive {archive_path}: {exc}") from exc
    with archive:
        members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        for member in archive.getmembers():
            relative = _safe_member_path(member.name)
            key = relative.as_posix()
            if key in seen:
                raise SafeExtractionError(f"duplicate normalized archive member: {key}")
            seen.add(key)
            if not (member.isdir() or member.isreg()):
                raise SafeExtractionError(
                    f"archive member is a link/device/fifo/special entry: {key}"
                )
            if member.size < 0 or (member.isdir() and member.size != 0):
                raise SafeExtractionError(f"archive member has invalid size: {key}")
            members.append((member, relative))
        for member, relative in members:
            target = destination.joinpath(*relative.parts)
            try:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise SafeExtractionError(
                        f"regular archive member has no readable payload: {relative.as_posix()}"
                    )
                with source:
                    _copy_exact(source, target, member.size)
            except (FileExistsError, NotADirectoryError, OSError) as exc:
                raise SafeExtractionError(
                    f"archive member conflicts with another path: {relative.as_posix()}"
                ) from exc


def tree_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.is_dir() or root_path.is_symlink():
        raise SafeExtractionError(f"extraction root is not a real directory: {root_path}")
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for current, dirnames, filenames in os.walk(root_path, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(dirnames):
            path = current_path / name
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise SafeExtractionError(f"non-directory/link in directory inventory: {path}")
            directories.append(path.relative_to(root_path).as_posix())
        for name in sorted(filenames):
            path = current_path / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                raise SafeExtractionError(f"non-regular/link in file inventory: {path}")
            files.append(
                {
                    "path": path.relative_to(root_path).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    directories.sort()
    files.sort(key=lambda row: str(row["path"]))
    return {
        "schema_version": "mdc_safe_extraction_manifest_v1",
        "directories": directories,
        "files": files,
        "directory_count": len(directories),
        "file_count": len(files),
        "total_file_bytes": sum(int(row["bytes"]) for row in files),
    }


def safe_extract(
    archive_path: str | Path,
    extracted_root: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    extracted = Path(extracted_root).resolve()
    manifest_output = Path(manifest_path).resolve()
    receipt_output = Path(receipt_path).resolve()
    if not archive.is_file():
        raise SafeExtractionError(f"archive is not a regular file: {archive}")
    if (
        manifest_output == receipt_output
        or _path_is_within(extracted, manifest_output)
        or _path_is_within(extracted, receipt_output)
    ):
        raise SafeExtractionError("extraction outputs collide")
    extracted.parent.mkdir(parents=True, exist_ok=True)
    fresh: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{extracted.name}.", suffix=".fresh", dir=extracted.parent)
    )
    try:
        _extract_fresh(archive, fresh)
        fresh_manifest = tree_manifest(fresh)
        if extracted.exists():
            if tree_manifest(extracted) != fresh_manifest:
                raise SafeExtractionError(
                    "existing extraction differs from a fresh archive-derived tree"
                )
        else:
            os.replace(fresh, extracted)
            fresh = None
        _immutable_write(manifest_output, _json_bytes(fresh_manifest))
        manifest_sha = sha256_file(manifest_output)
        receipt = {
            "schema_version": "mdc_safe_extraction_receipt_v1",
            "status": "passed_fresh_archive_tree_matches",
            "archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            "extraction": {
                "root": str(extracted),
                "manifest_path": str(manifest_output),
                "manifest_sha256": manifest_sha,
                "inventory_sha256": canonical_json_sha256(fresh_manifest),
                "file_count": fresh_manifest["file_count"],
                "directory_count": fresh_manifest["directory_count"],
                "total_file_bytes": fresh_manifest["total_file_bytes"],
            },
            "safety_policy": {
                "fresh_extract_every_run": True,
                "reject_absolute_or_parent_paths": True,
                "reject_duplicate_normalized_paths": True,
                "reject_links_devices_fifos_and_special_entries": True,
                "compare_existing_tree_to_fresh_archive_tree": True,
            },
            "tool": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        _immutable_write(receipt_output, _json_bytes(receipt))
        return receipt
    finally:
        if fresh is not None and fresh.exists():
            shutil.rmtree(fresh)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    receipt = safe_extract(
        args.archive, args.extracted_root, args.manifest, args.receipt
    )
    print(
        json.dumps(
            {
                "receipt": str(Path(args.receipt).resolve()),
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

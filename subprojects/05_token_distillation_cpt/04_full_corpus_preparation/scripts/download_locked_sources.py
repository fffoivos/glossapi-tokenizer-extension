#!/usr/bin/env python3
"""Download selected files from an immutable Phase-04 HF source lock."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path, algorithm: str = "sha1", chunk_size: int = 16 * 1024 * 1024) -> str:
    """Compute Git's object ID for a regular file without loading it in memory."""

    digest = hashlib.new(algorithm)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", action="append", help="download only selected source_id values")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-lfs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install huggingface_hub in the Phase-04 runtime") from exc

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "full_cpt_sources_lock_v1":
        raise ValueError("unsupported source lock schema")
    selected_ids = set(args.source or [])
    sources = [row for row in lock["sources"] if not selected_ids or row["source_id"] in selected_ids]
    unknown = selected_ids - {row["source_id"] for row in sources}
    if unknown:
        raise ValueError(f"unknown or unresolved source ids: {sorted(unknown)}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be supplied in the job environment; it is never read from repo files")

    completed = []
    for source in sources:
        local_dir = args.destination / source["source_id"] / source["revision"]
        local_dir.mkdir(parents=True, exist_ok=True)
        patterns = [row["path"] for row in source["selected_files"]]
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            revision=source["revision"],
            allow_patterns=patterns,
            local_dir=local_dir,
            max_workers=args.workers,
            token=token,
        )
        lfs_verified = 0
        git_blobs_verified = 0
        verified_files: list[dict] = []
        for expected in source["selected_files"]:
            path = local_dir / expected["path"]
            if not path.is_file():
                raise FileNotFoundError(f"download missing locked file: {path}")
            pre_hash_stat = path.stat()
            if pre_hash_stat.st_size != expected["size"]:
                raise ValueError(
                    f"{path}: size mismatch, expected {expected['size']}, got {pre_hash_stat.st_size}"
                )
            if args.verify_lfs and expected.get("lfs_sha256"):
                actual = sha256_file(path)
                if actual != expected["lfs_sha256"]:
                    raise ValueError(
                        f"{path}: LFS SHA-256 mismatch, expected {expected['lfs_sha256']}, got {actual}"
                    )
                lfs_verified += 1
                hash_kind = "lfs_sha256"
                expected_hash = expected["lfs_sha256"]
            elif expected.get("lfs_sha256"):
                hash_kind = "lfs_sha256"
                expected_hash = expected["lfs_sha256"]
                actual = None
            elif expected.get("blob_id"):
                expected_hash = str(expected["blob_id"])
                if len(expected_hash) not in {40, 64}:
                    raise ValueError(f"{path}: unsupported Git blob ID length: {expected_hash}")
                algorithm = "sha1" if len(expected_hash) == 40 else "sha256"
                actual = git_blob_id(path, algorithm=algorithm)
                if actual != expected_hash:
                    raise ValueError(
                        f"{path}: Git blob mismatch, expected {expected_hash}, got {actual}"
                    )
                git_blobs_verified += 1
                hash_kind = "git_blob_id"
            else:
                raise ValueError(f"{path}: locked file has no immutable payload identifier")
            post_hash_stat = path.stat()
            stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(pre_hash_stat, field) != getattr(post_hash_stat, field) for field in stat_fields):
                raise ValueError(f"{path}: file metadata changed while payload verification was running")
            verified_files.append(
                {
                    "path": expected["path"],
                    "size": expected["size"],
                    "hash_kind": hash_kind,
                    "expected_hash": expected_hash,
                    "actual_hash": actual,
                    "device": post_hash_stat.st_dev,
                    "inode": post_hash_stat.st_ino,
                    "mtime_ns": post_hash_stat.st_mtime_ns,
                    "ctime_ns": post_hash_stat.st_ctime_ns,
                }
            )
        completed.append(
            {
                "source_id": source["source_id"],
                "repo_id": source["repo_id"],
                "repo_type": source["repo_type"],
                "revision": source["revision"],
                "local_dir": str(local_dir),
                "files": len(source["selected_files"]),
                "bytes": source["selected_bytes"],
                "lfs_sha256_verified": lfs_verified,
                "git_blob_ids_verified": git_blobs_verified,
                "verified_files": verified_files,
            }
        )
        print(f"downloaded {source['source_id']} -> {local_dir}", flush=True)

    manifest = {
        "schema_version": "full_cpt_download_manifest_v1",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_lock": str(args.lock.resolve()),
        "source_lock_sha256": sha256_file(args.lock),
        "destination": str(args.destination.resolve()),
        "sources": completed,
    }
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite immutable download manifest: {args.manifest}")
    write_json_atomic(args.manifest, manifest)
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
